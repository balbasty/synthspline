import torch
from torch import nn as tnn
from nitorch import nn
from nitorch.core import py, optionals, linalg, utils
from nitorch.spatial._curves import BSplineCurve, draw_curves
from nitorch.spatial import identity_grid
from torch import distributions
import math as pymath
from . import random


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


def setup_sampler(value):
    if isinstance(value, random.Sampler):
        return value
    elif not isinstance(value, (list, tuple)):
        return random.Dirac(value)
    else:
        return random.Uniform(*value)


class SynthSplineBlock(tnn.Module):

    def __init__(
            self,
            shape,
            voxel_size=0.1,                             # 100 mu
            tree_density=random.LogNormal(0.5, 0.5),    # trees/mm3 (should be 8 according to known_stats)
            tortuosity=random.LogNormal(0.7, 0.2),      # expected jitter in mm
            radius=random.LogNormal(0.07, 0.01),        # mean radius
            radius_change=random.LogNormal(1., 0.1),    # radius variation along the vessel
            nb_levels=1,                                # number of hierarchical level in the tree
            nb_children=random.LogNormal(5, 5),         # mean number of children
            radius_ratio=random.LogNormal(0.7, 0.1),    # Radius ratio child/parent
            device=None):
        super().__init__()
        self.shape = shape
        self.vx = voxel_size
        self.device = device or 'cpu'
        self.tree_density = setup_sampler(tree_density)
        self.tortuosity = setup_sampler(tortuosity)
        self.radius = setup_sampler(radius)
        self.radius_change = setup_sampler(radius_change)
        self.nb_children = setup_sampler(nb_children)
        self.radius_ratio = setup_sampler(radius_ratio)
        self.nb_levels = setup_sampler(nb_levels)

    def sample_curve(self, first=None, last=None, radius=None):

        dim = len(self.shape)
        def length(a, b):
            return (a-b).square().sum().sqrt()

        def linspace(a, b, n):
            vector = (b-a) / (n-1)
            return a + vector * torch.arange(n).unsqueeze(-1)

        # sample initial point and length
        n = 0
        while n < 3:

            # sample initial point
            if first is None:
                side1 = torch.randint(2 * dim, [])
                a = torch.cat([torch.rand([1]) * (s - 1) for s in self.shape])
                if side1 // dim:
                    a[side1 % dim] = 0
                else:
                    a[side1 % dim] = self.shape[side1 % dim] - 1
                side2 = side1
                while side2 == side1:
                    side2 = torch.randint(2 * dim, [])
            else:
                a = first
                side2 = torch.randint(2 * dim, [])

            # sample final point
            if last is None:
                b = torch.cat([torch.rand([1]) * (s - 1) for s in self.shape])
                if side2 // dim:
                    b[side2 % dim] = 0
                else:
                    b[side2 % dim] = self.shape[side2 % dim] - 1
            else:
                b = last

            # initial straight line
            l = length(a, b)  # true length
            n = torch.randint(3, (l / 5).ceil().int().clamp_min_(4), [])

        # deform curve + sample radius
        waypoints = linspace(a, b, n)
        sigma = (self.tortuosity() - 1) * l / (2 * dim * (n - 1))
        sigma = sigma.clamp_min_(0)
        if sigma:
            waypoints[1:-1] += sigma * torch.randn([n - 2, dim])
        radius = radius or self.radius
        radii = radius() / self.vx
        radii = self.radius_change([n]) * radii
        radii.clamp_min_(0.5)
        curve = BSplineCurve(waypoints.to(self.device),
                             radius=radii.to(self.device))
        return curve

    def sample_tree(self, first=None, n_level=0, max_level=0, radius=None):
        radius = radius or self.radius
        root = self.sample_curve(first, radius=radius)
        curves = [root]
        levels = [n_level+1]
        if n_level >= max_level - 1:
            return curves, levels, []

        nb_children = self.nb_children().floor().int()
        branchings = []
        for c in range(nb_children):
            t = torch.rand([])
            first = root.eval_position(t)
            root_radius = root.eval_radius(t)
            branchings += [(first, root_radius.item())]
            root_radius *= self.vx
            radius_ratio = self.radius_ratio()
            radius_sampler = radius * radius_ratio
            cuves1, levels1, branchings1 \
                = self.sample_tree(first, n_level + 1, max_level, radius_sampler)
            curves += cuves1
            levels += levels1
            branchings += branchings1

        return curves, levels, branchings

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
        print(nb_trees)

        start = time.time()
        curves = []
        levels = []
        branchings = []
        for n in range(nb_trees):
            curves1, levels1, branchings1 = self.sample_tree(max_level=self.nb_levels())
            curves += curves1
            levels += levels1
            branchings += branchings1
        print('sample curves: ', time.time() - start)

        # draw vessels
        start = time.time()
        vessels, labels = draw_curves(
            self.shape, curves, fast=True, mode='cosine')

        levelmap = torch.zeros_like(labels)
        for i, l in enumerate(levels):
            levelmap.masked_fill_(labels == i+1, l)

        branchmap = torch.zeros_like(vessels)
        id = identity_grid(branchmap.shape, device=branchmap.device)
        for branch in branchings:
            loc, radius = branch
            mask = (id - loc).square_().sum(-1).sqrt_() < radius + 0.5
            if mask.any():
                branchmap.masked_fill_(mask, True)
            else:
                loc = loc.round().long().tolist()
                if all(0 <= l < s for l, s in zip(loc, branchmap.shape)):
                    branchmap[tuple(loc)] = True

        print('draw curves: ', time.time() - start)

        vessels = vessels[None, None]
        labels = labels[None, None]
        levelmap = levelmap[None, None]
        branchmap = branchmap[None, None]
        return vessels, labels, levelmap, branchmap


class SynthVesselMicro(SynthSplineBlock):

    def __init__(
            self,
            shape=(256, 256, 256),                      # ~16 mm3
            voxel_size=0.01,                            # 10 mu
            tree_density=random.LogNormal(8, 1),        # trees/mm3
            tortuosity=random.LogNormal(2, 1),          # expected jitter in mm
            radius=random.LogNormal(0.07, 0.01),        # mean radius
            radius_change=random.LogNormal(1., 0.1),    # radius variation along the vessel
            nb_levels=5,                                # number of hierarchical level in the tree
            nb_children=random.LogNormal(5, 5),         # mean number of children
            radius_ratio=random.LogNormal(0.7, 0.1),    # Radius ratio child/parent
            device=None):
        super().__init__(shape, voxel_size, tree_density, tortuosity,
                         radius, radius_change, nb_levels, nb_children,
                         radius_ratio, device)


class SynthVesselHiResMRI(SynthSplineBlock):

    def __init__(
            self,
            shape=(256, 256, 256),                      # ~16 mm3
            voxel_size=0.1,                             # 100 mu
            tree_density=random.LogNormal(0.01, 0.01),  # trees/mm3
            tortuosity=random.LogNormal(5, 3),          # expected jitter in mm
            radius=random.LogNormal(0.1, 0.02),         # mean radius
            radius_change=random.LogNormal(1., 0.1),    # radius variation along the vessel
            nb_levels=2,                                # number of hierarchical level in the tree
            nb_children=random.LogNormal(5, 5),         # mean number of children
            radius_ratio=random.LogNormal(0.7, 0.1),    # Radius ratio child/parent
            device=None):
        super().__init__(shape, voxel_size, tree_density, tortuosity,
                         radius, radius_change, nb_levels, nb_children,
                         radius_ratio, device)


class SynthAxon(SynthSplineBlock):

    def __init__(
            self,
            shape=(256, 256, 256),
            voxel_size=1e-3,                            # 1 mu
            axon_density=random.LogNormal(3e4, 5e3),    # axons/mm3
            tortuosity=random.LogNormal(1.1, 2),        # expected jitter in mm
            radius=random.LogNormal(1e-3, 2e-4).clamp_max(2e-3),  # mean radius in mm
            radius_change=random.LogNormal(1., 0.1),    # radius variation along the axon
            nb_levels=2,                                # number of hierarchical level in the tree
            nb_children=random.LogNormal(0.5, 1),       # mean number of children
            radius_ratio=random.LogNormal(1, 0.1),      # Radius ratio child/parent
            device=None):
        super().__init__(shape, voxel_size, axon_density, tortuosity,
                         radius, radius_change, nb_levels, nb_children,
                         radius_ratio, device)
