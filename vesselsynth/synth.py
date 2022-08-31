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
            return torch.cat([self() for _ in range(batch)])

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
        l0 = (py.prod(self.shape) ** (1 / dim))  # typical length
        curves = []
        max_radius = 0
        print(nb_trees)
        for n_tree in range(nb_trees):

            # sample initial point and length
            n = 0
            while n < 3:
                a = clamp(torch.randn([dim]) * l0)      # initial point
                l = torch.rand([dim]) * l0              # length
                b = clamp(a + l)                        # end point
                l = length(a, b)                        # true length
                n = (l / 5).ceil().int().item()         # number of discrete points

            waypoints = linspace(a, b, n)
            waypoints += self.tortuosity() * torch.randn([n, dim])
            radii = self.radius() * self.radius_change([n])
            curve = BSplineCurve(waypoints.to(self.device),
                                 radius=radii.to(self.device))
            max_radius = max(max_radius, radii.max())
            curves.append(curve)
        print('sample curves: ', time.time() - start)

        # draw vessels
        start = time.time()
        true_vessels, _ = draw_curves(self.shape, curves, fast=10*max_radius)
        print('draw curves: ', time.time() - start)

        import matplotlib.pyplot as plt
        plt.imshow(mip_depth(true_vessels.squeeze()))
        plt.show()

        return true_vessels[None, None]








