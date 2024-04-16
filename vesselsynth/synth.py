import torch
from torch import nn as tnn
from .curves import BSplineCurve, BSplineCurves
from . import random
from interpol import identity_grid


def setup_sampler(value):
    if isinstance(value, random.Sampler):
        return value
    elif not isinstance(value, (list, tuple)):
        return random.Dirac(value)
    else:
        return random.Uniform(*value)


class SynthSplineBlock(tnn.Module):

    class defaults:
        shape: int = (64, 64, 64)
        voxel_size: float = 0.1
        nb_levels: int = 1
        tree_density = random.LogNormal(mean=0.5, std=0.5)
        tortuosity = random.LogNormal(mean=0.7, std=0.2)
        radius = random.LogNormal(mean=0.07, std=0.01)
        radius_change = random.LogNormal(mean=1., std=0.1)
        nb_children = random.LogNormal(mean=5, std=5)
        radius_ratio = random.LogNormal(mean=0.7, std=0.1)
        device = None

    def __init__(self, *args, **kwargs):
        """

        Parameters
        ----------
        shape : list[int]
            Number of voxels in the synthesized patch
        voxel_size : float
            Voxel size of the patch
        tree_density : Sampler
            Number of trees per mm^3
            For vessels, should be 8 (in the cortex) according to known_stats
            Default: LogNormal(0.5, 0.5), 95% of samples in [0.6, 4.5]
        tortuosity : Sampler
            Tortuosity ~= cord / length
            Default: LogNormal(0.7, 0.2), 95% of samples in [1.3, 3.0]
        radius : Sampler
            Mean radius at the first (coarsest) level
            Default: LogNormal(0.07, 0.01), 95% of samples in [1.05, 1.09]
        radius_change : Sampler
            Radius variation along the length of the spline
            Default : LogNormal(1., 0.1), 95% of samples in [2.2, 3.3]
        nb_levels : Sampler
            Number of hierarchical levels
        nb_children :
            Number of children per spline
            Default: LogNormal(5, 5)
        radius_ratio : Sampler
            Ratio between the mean radius at child and parent levels
            Default : LogNormal(0.7, 0.1), 95% of samples in [1.6, 2.5]
        """
        super().__init__()
        params = {key: val for key, val in self.defaults.__dict__.items()
                  if key[0] != '_'}
        params.update({key: val for key, val in zip(params.keys(), args)})
        params.update(kwargs)
        for key, val in params.items():
            if key not in ('shape', 'voxel_size', 'device'):
                val = setup_sampler(val)
            setattr(self, key, val)

    def sample_curve(self, first=None, last=None, radius=None):
        """
        Sample a single spline

        Parameters
        ----------
        first : (ndim,) tensor, optional
            Coordinates of the first endpoint.
            Randomly sampled on a face of the cube if not provided.
        last : (ndim,) tensor, optional
            Coordinates of the last endpoint.
            Randomly sampled on a (different) face of the cube if not provided.
        radius : Sampler, default=self.radius
            Mean radius sampler.

        Returns
        -------
        spline : BSplineCurve
        """

        shape = self.shape
        ndim = len(shape)

        def length(a, b):
            # length of a straight line
            return (a-b).square().sum().sqrt()

        def linspace(a, b, n):
            # Generate a straight line with `n` control points
            vector = (b-a) / (n-1)
            return a + vector * torch.arange(n).unsqueeze(-1)

        # Sample initial endpoint and length
        # We keep trying until we get a segment that's at least 3 voxel long
        segment_length = 0
        first_side = None
        while segment_length < 3:

            # If not provided, sample first endpoint
            #
            #   It is provided when we are branching from another spline,
            #   or when we're on a retry du to a too short spline, in which
            #   case we keep, the original first endpoint.
            if first is None:
                # Sample a 3D coordinate inside the cube
                first = torch.rand([ndim]) * (torch.as_tensor(shape) - 1)
                # Project it onto one of the faces of the cube
                first_side = torch.randint(2 * ndim, [])
                if first_side // ndim:
                    first[first_side % ndim] = 0
                else:
                    first[first_side % ndim] = shape[first_side % ndim] - 1

            # If not provided, sample last endpoint
            if last is None:
                # Sample a 3D coordinate inside the cube
                last = torch.rand([ndim]) * (torch.as_tensor(shape) - 1)
                # Sample the face of the second endpoint
                last_side = torch.randint(2 * ndim, [])
                if first_side:
                    while last_side == first_side:
                        last_side = torch.randint(2 * ndim, [])
                # Project the point onto the face
                if last_side // ndim:
                    last[last_side % ndim] = 0
                else:
                    last[last_side % ndim] = self.shape[last_side % ndim] - 1

            # Sample intermediate control points every 5 voxels
            # TODO: make `n` depend on tortuosity
            segment_length = length(first, last)
            n = (segment_length / 5).ceil().int()
            n = torch.randint(3, n.clamp_min_(4), [])

        # Initial straight line with control points
        waypoints = linspace(first, last, n)

        # Sample tortuosity and jitter intermediate points accordingly
        # (tortuosity ~= cord_length / curvilinear_length)
        tortuosity = self.tortuosity()
        sigma = (tortuosity - 1) * segment_length / (2 * ndim * (n - 1))
        sigma = sigma.clamp_min_(0)
        if sigma:
            waypoints[1:-1] += sigma * torch.randn([n - 2, ndim])
        # Sample average radius
        radius = radius or self.radius
        radius = radius() / self.voxel_size  # sample and convert to voxels
        # Sample radius that changes across the length of the spline
        radii = self.radius_change([n]) * radius
        radii.clamp_min_(0.5)
        # Return B-spline
        return BSplineCurve(waypoints, radius=radii)

    def sample_tree(self, first=None, n_level=0, max_level=0, radius=None):
        """
        Sample a single tree.
        This function is called recursively to build a tree.

        Parameters
        ----------
        first : (ndim,) tensor, optional
            Coordinates of the first endpoint.
            Randomly sampled on a face of the cube if not provided.
        n_level : int
            Index of the current level in the tree
        max_level : int
            Maximum number of levels in the tree.
            Stop the recursion if current level is larger than this.
        radius : Sampler, default=self.radius
            Mean radius sampler.

        Returns
        -------
        curves : list[BSplineCurve]
            All individual splines in the tree
        levels : list[int]
            The level of each spline in the tree
        branchings : list[(tensor, float)]
            The first endpoint and radius at endpoint of each spline.
            This list has one fewer elements than the other two, since the
            root spline has no initial branching point.
        """
        radius = radius or self.radius
        root = self.sample_curve(first, radius=radius)
        curves = [root]
        levels = [n_level+1]
        if n_level >= max_level - 1:
            return curves, levels, []

        nb_children = self.nb_children().floor().int()
        branchings = []
        for _ in range(nb_children):
            t = torch.rand([])
            first = root.eval_position(t)
            root_radius = root.eval_radius(t)
            branchings += [(first, root_radius.item())]
            root_radius *= self.voxel_size
            radius_ratio = self.radius_ratio()
            radius_sampler = radius * radius_ratio
            cuves1, levels1, branchings1 = self.sample_tree(
                first, n_level + 1, max_level, radius_sampler)
            curves += cuves1
            levels += levels1
            branchings += branchings1

        return curves, levels, branchings

    def forward(self, batch=1):
        """
        Sample everything

        Parameters
        ----------
        batch : int
            Number of patches to generate

        Returns
        -------
        vessels : (batch, 1, *shape) tensor[float]
            Vessels partial volume map.
        labels : (batch, 1, *shape) tensor[int]
            Unique label of each spline.
            Every voxel that has a nonzero partial volume has a nonzero index.
        levelmap : (batch, 1, *shape) tensor[int]
            Hierarchical level of each spline (starting at one).
            Every voxel that has a nonzero partial volume has a nonzero label.
        nblevelmap : (batch, 1, *shape) tensor[int]
            Number of levels in the tree to which each spline belongs.
        branchmap : (batch, 1, *shape) tensor[bool]
            Mask of all branching points (they have the side of their radius).
        skeleton : (batch, 1, *shape) tensor[bool]
            Mask of all voxels that are traversed by a centerline.
        dist : (batch, 1, *shape) tensor[float]
            Distance from eahc voxel to its nearest centerline.
        """
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
        volume = 1
        for s in self.shape:
            volume *= s
        volume *= (self.voxel_size ** dim)
        density = self.tree_density()
        nb_trees = max(int(volume * density // 1), 1)
        print('Sampling', nb_trees, 'trees')

        start = time.time()
        curves = []
        levels = []
        branchings = []
        nb_levels = []
        for _ in range(nb_trees):
            nb_levels1 = self.nb_levels()
            curves1, levels1, branchings1 \
                = self.sample_tree(max_level=nb_levels1)
            nb_levels += [max(levels1)] * len(curves1)
            curves += curves1
            levels += levels1
            branchings += branchings1
        print(f'Curves sampled in {time.time() - start:.2f} sec')

        # draw vessels
        start = time.time()
        curves = BSplineCurves(curves)
        curves.to(self.device)
        vessels, labels, dist = curves.rasterize(self.shape, mode='cosine')

        levelmap = torch.zeros_like(labels)
        for i, l in enumerate(levels):
            levelmap.masked_fill_(labels == i+1, l)

        nblevelmap = torch.zeros_like(labels)
        for i, l in enumerate(nb_levels):
            nblevelmap.masked_fill_(labels == i+1, l)

        skeleton = torch.zeros_like(labels)
        for i, curve in enumerate(curves):
            ind = curve.evaluate_equidistant(curve, 0.1)
            ind = ind.round().long()
            ind = ind[(ind[:, 0] >= 0) & (ind[:, 0] < skeleton.shape[0])]
            ind = ind[(ind[:, 1] >= 0) & (ind[:, 1] < skeleton.shape[1])]
            ind = ind[(ind[:, 2] >= 0) & (ind[:, 2] < skeleton.shape[2])]
            ind = ind[:, 2] \
                + ind[:, 1] * skeleton.shape[2] \
                + ind[:, 0] * skeleton.shape[2] * skeleton.shape[1]
            skeleton.view([-1])[ind] = i+1

        branchmap = torch.zeros_like(vessels)
        id = identity_grid(branchmap.shape, device=branchmap.device)
        for branch in branchings:
            loc, radius = branch
            loc = loc.to(id)
            mask = (id - loc).square_().sum(-1).sqrt_() < radius + 0.5
            if mask.any():
                branchmap.masked_fill_(mask, True)
            else:
                loc = loc.round().long().tolist()
                if all(0 <= c < s for c, s in zip(loc, branchmap.shape)):
                    branchmap[tuple(loc)] = True

        print(f'Curves rasterized in {time.time() - start:.3f} sec')

        vessels = vessels[None, None]
        labels = labels[None, None]
        levelmap = levelmap[None, None]
        branchmap = branchmap[None, None]
        skeleton = skeleton[None, None]
        return vessels, labels, levelmap, nblevelmap, branchmap, skeleton, dist


class SynthVesselMicro(SynthSplineBlock):

    class defaults:
        shape = (256, 256, 256)
        """Number of voxels in the patch (256 -> ~16 mm3 @ 10 micron)"""
        voxel_size = 0.01
        """10 micron"""
        nb_levels = 5
        """Number of hierarchical level in the tree. Could be random."""
        tree_density = random.LogNormal(mean=8, std=1)
        """Trees/mm3"""
        tortuosity = random.LogNormal(mean=2, std=1)
        """Tortuosity ~= cord / length"""
        radius = random.LogNormal(mean=0.07, std=0.01)
        """Mean radius, in mm, at coarsest level"""
        radius_change = random.LogNormal(mean=1., std=0.1)
        """Radius variation along the spline"""
        nb_children = random.LogNormal(mean=5, std=5)
        """Number of branches per spline"""
        radius_ratio = random.LogNormal(mean=0.7, std=0.1)
        """Radius ratio branch/parent"""
        device = None


class SynthVesselHiResMRI(SynthSplineBlock):

    class defaults:
        shape = (256, 256, 256)
        """Number of voxels in the patch (256 -> ~16 cm3 @ 100 micron)"""
        voxel_size = 0.1
        """100 micron"""
        nb_levels = 2
        """Number of hierarchical level in the tree. Could be random."""
        tree_density = random.LogNormal(mean=0.01, std=0.01)
        """Trees/mm3"""
        tortuosity = random.LogNormal(mean=5, std=3)
        """Tortuosity ~= cord / length"""
        radius = random.LogNormal(mean=0.1, std=0.02)
        """Mean radius, in mm, at coarsest level"""
        radius_change = random.LogNormal(mean=1., std=0.1)
        """Radius variation along the spline"""
        nb_children = random.LogNormal(mean=5, std=5)
        """Number of branches per spline"""
        radius_ratio = random.LogNormal(mean=0.7, std=0.1)
        """Radius ratio branch/parent"""
        device = None


class SynthVesselOCT(SynthSplineBlock):

    class defaults:
        shape = (128, 128, 128)
        """Number of voxels in the patch (256 -> ~2 mm3 @ 10 micron)"""
        voxel_size = 0.01
        """10 micron"""
        nb_levels = 4
        """Number of hierarchical level in the tree. Could be random."""
        tree_density = random.LogNormal(mean=0.01, std=0.01)
        """Trees/mm3"""
        tortuosity = random.LogNormal(mean=1.5, std=5)
        """Tortuosity ~= cord / length"""
        radius = random.LogNormal(mean=0.1, std=0.02)
        """Mean radius, in mm, at coarsest level"""
        radius_change = random.LogNormal(mean=1., std=0.2)
        """Radius variation along the spline"""
        nb_children = random.LogNormal(mean=2, std=3)
        """Number of branches per spline"""
        radius_ratio = random.LogNormal(mean=0.5, std=0.1)
        """Radius ratio branch/parent"""
        device = None


class SynthVesselPhoto(SynthSplineBlock):

    class defaults:
        shape = (256, 256, 256)
        """Number of voxels in the patch (256 -> ~450 mm3 @ 30 micron)"""
        voxel_size = 0.03
        """30 micron"""
        nb_levels = random.RandInt(min=1, max=5)
        """Number of hierarchical level in the tree. Could be random."""
        tree_density = random.LogNormal(mean=0.1, std=0.2)
        """Trees/mm3"""
        tortuosity = random.LogNormal(mean=1, std=5)
        """Tortuosity ~= cord / length"""
        radius = random.LogNormal(mean=0.1, std=0.02)
        """Mean radius, in mm, at coarsest level"""
        radius_change = random.LogNormal(mean=1., std=0.2)
        """Radius variation along the spline"""
        nb_children = random.LogNormal(mean=2, std=3)
        """Number of branches per spline"""
        radius_ratio = random.LogNormal(mean=0.5, std=0.1)
        """Radius ratio branch/parent"""
        device = None


class SynthAxon(SynthSplineBlock):

    class defaults:
        shape = (256, 256, 256)
        """Number of voxels in the patch (256 -> 16E-3 mm3 @ 1 micron)"""
        voxel_size = 1e-3
        """1 micron"""
        nb_levels = 2
        """Number of hierarchical level in the tree. Could be random."""
        tree_density = 10 ** random.Uniform(min=4, max=6)
        """Trees/mm3"""
        tortuosity = random.LogNormal(mean=1.1, std=2)
        """Tortuosity ~= cord / length"""
        radius = random.LogNormal(mean=1e-3, std=5e-4).clamp_max(2e-3)
        """Mean radius, in mm, at coarsest level"""
        radius_change = random.LogNormal(mean=1., std=0.1)
        """Radius variation along the spline"""
        nb_children = random.LogNormal(mean=0.5, std=1)
        """Number of branches per spline"""
        radius_ratio = random.LogNormal(mean=1, std=0.1)
        """Radius ratio branch/parent"""
        device = None
