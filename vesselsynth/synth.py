import torch
from torch import nn as tnn
from nitorch import nn
from nitorch.core import py, optionals, linalg, utils
from nitorch.spatial._curves import BSplineCurve, draw_curves
from torch import distributions
import math as pymath


def _get_colormap_depth(colormap, n=256, dtype=None, device=None):
    plt = optionals.try_import_as('matplotlib.pyplot')
    mcolors = optionals.try_import_as('matplotlib.colors')
    if colormap is None:
        if not plt:
            raise ImportError('Matplotlib not available')
        colormap = plt.get_cmap('rainbow')
    elif plt and isinstance(colormap, str):
        colormap = plt.get_cmap(colormap)
    if mcolors and isinstance(colormap, mcolors.Colormap):
        colormap = [colormap(i/(n-1))[:3] for i in range(n)]
    else:
        raise ImportError('Matplotlib not available')
    colormap = torch.as_tensor(colormap, dtype=dtype, device=device)
    return colormap


def mip_depth(x, dim=-1, colormap='rainbow'):
    cmap = _get_colormap_depth(colormap, x.shape[dim],
                               dtype=x.dtype, device=x.device)
    x = utils.movedim(x, dim, -1)
    d = linalg.dot(x.unsqueeze(-2), cmap.T)
    d /= x.sum(-1, keepdim=True)
    d *= x.max(-1, keepdim=True).values
    return d


def setup_lognormal_sampler(value):
    if callable(value):
        return value
    else:
        exp, scale = py.make_list(value, 2, default=0)
        if scale:
            var = scale * scale
            var_log = pymath.log(1 + var / (exp*exp))
            exp_log = pymath.log(exp) - var_log / 2
            scale_log = pymath.sqrt(var_log)
            return distributions.LogNormal(exp_log, scale_log).sample
        else:
            return lambda n=0: (torch.full([n], exp) if n else exp)


class SynthSplineBlock(tnn.Module):

    def __init__(
            self,
            shape,
            voxel_size=0.1,             # 100 mu
            tree_density=(0.5, 0.5),    # trees/mm3 (should be 8 according to known_stats)
            tortuosity=(0.7, 0.2),      # expected jitter in mm
            radius=(0.07, 0.01),        # mean radius
            radius_change=(1., 0.1),    # radius variation along the vessel
            device=None):
        super().__init__()
        self.shape = shape
        self.vx = voxel_size
        self.device = device or 'cpu'
        self.tree_density = setup_lognormal_sampler(tree_density)
        self.tortuosity = setup_lognormal_sampler(tortuosity)
        self.radius = setup_lognormal_sampler(radius)
        self.radius_change = setup_lognormal_sampler(radius_change)

    def forward(self, batch=1):
        if batch > 1:
            prob, lab = [], []
            for _ in range(batch):
                prob1, lab1 = self()
                prob += [prob1]
                lab += [lab1]
            return torch.cat(prob), torch.cat(lab)

        import time
        dim = len(self.shape)

        # sample vessels
        volume = py.prod(self.shape) * (self.vx ** dim)
        density = self.tree_density()
        nb_trees = int(volume * density // 1)

        def clamp(x):
            # ensure point is inside FOV
            x = x.clamp_min(0)
            mx = torch.as_tensor(self.shape, dtype=x.dtype)
            x = torch.min(x, mx.to(x))
            return x

        def length(a, b):
            return (a-b).square().sum().sqrt()

        def linspace(a, b, n):
            vector = (b-a) / (n-1)
            return a + vector * torch.arange(n).unsqueeze(-1)

        start = time.time()
        curves = []
        max_radius = 0
        print(nb_trees)
        for n_tree in range(nb_trees):

            # sample initial point and length
            n = 0
            while n < 3:
                side1 = torch.randint(2*dim, [])
                a = torch.cat([torch.rand([1]) * (s-1) for s in self.shape])
                if side1 // dim:
                    a[side1 % dim] = 0
                else:
                    a[side1 % dim] = self.shape[side1 % dim] - 1
                side2 = side1
                while side2 == side1:
                    side2 = torch.randint(2*dim, [])
                b = torch.cat([torch.rand([1]) * (s-1) for s in self.shape])
                if side2 // dim:
                    b[side2 % dim] = 0
                else:
                    b[side2 % dim] = self.shape[side2 % dim] - 1
                l = length(a, b)                        # true length
                # n = (l / 5).ceil().int().item()         # number of discrete points
                n = torch.randint(3, (l / 5).ceil().int().clamp_min_(4), [])

            waypoints = linspace(a, b, n)
            sigma = (self.tortuosity() - 1) * l / (2 * dim * (n - 1))
            sigma = sigma.clamp_min_(0)
            if sigma:
                waypoints[1:-1] += sigma * torch.randn([n-2, dim])
            radii = self.radius() * self.radius_change([n]) / self.vx
            curve = BSplineCurve(waypoints.to(self.device),
                                 radius=radii.to(self.device))
            max_radius = max(max_radius, radii.max())
            curves.append(curve)
        print('sample curves: ', time.time() - start)

        # draw vessels
        start = time.time()
        true_vessels, true_labels = draw_curves(
            self.shape, curves, fast=10*max_radius, mode='binary')
        print('draw curves: ', time.time() - start)

        return true_vessels[None, None], true_labels[None, None]


class SynthVesselMicro(SynthSplineBlock):

    def __init__(
            self,
            shape=(256, 256, 256),      # ~16 mm3
            voxel_size=0.01,            # 10 mu
            tree_density=(8, 1),        # trees/mm3
            tortuosity=(2, 1),          # expected jitter in mm
            radius=(0.07, 0.01),        # mean radius
            radius_change=(1., 0.1),    # radius variation along the vessel
            device=None):
        super().__init__(shape, voxel_size, tree_density, tortuosity,
                         radius, radius_change, device)


class SynthVesselHiResMRI(SynthSplineBlock):

    def __init__(
            self,
            shape=(256, 256, 256),      # ~16 mm3
            voxel_size=0.1,             # 100 mu
            tree_density=(0.01, 0.01),  # trees/mm3
            tortuosity=(5, 3),          # expected jitter in mm
            radius=(0.1, 0.02),         # mean radius
            radius_change=(1., 0.1),    # radius variation along the vessel
            device=None):
        super().__init__(shape, voxel_size, tree_density, tortuosity,
                         radius, radius_change, device)