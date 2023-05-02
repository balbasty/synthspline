import torch
from torch import nn as tnn
from .curves import BSplineCurve, draw_curves, discretize_equidistant
from . import random
from interpol import identity_grid


def setup_sampler(value):
    if isinstance(value, random.Sampler):  # Check if value is an instance of the random.Sampler class
        return value # If yes, return as is
    elif not isinstance(value, (list, tuple)): # Check if value is NOT an instance of a list or tuple
        return random.Dirac(value) # If not a list or tuple, return a fixed parameter distribution with single mode at value
    else: # If it IS a list or a tuple
        return random.Uniform(*value) # Return a random.Uniform object with value as its arguments (unpacked using the * operator)


class SynthSplineBlock(tnn.Module):

    def __init__(
            self,
            shape,                                      # default to 128, size of image
            voxel_size=0.1,                             # 0.1 mm = 100 mu --> needs to have the same units as spatial dimension of tree_density
            tree_density=random.LogNormal(0.5, 0.5),    # trees/mm3 (should be 8 according to known_stats)
            tortuosity=random.LogNormal(0.7, 0.2),      # expected jitter in mm
            radius=random.LogNormal(0.07, 0.01),        # mean radius
            radius_change=random.LogNormal(1., 0.1),    # radius variation along the vessel
            nb_levels=1,                                # number of hierarchical level in the tree
            nb_children=random.LogNormal(5, 5),         # mean number of children
            radius_ratio=random.LogNormal(0.7, 0.1),    # Radius ratio child/parent
            device=None):
        """

        Parameters
        ----------
        shape : list[int]
        voxel_size : float
        tree_density : Sampler
            Number of trees per mm3
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
            '''Makes vectors on range [a, b] with n steps'''
            vector = (b-a) / (n-1)
            return a + vector * torch.arange(n).unsqueeze(-1)

        # sample initial point and length
        n = 0
        while n < 3:

            # sample initial point
            if first is None: # Sanity check: generates point if not given
                side1 = torch.randint(2 * dim, [])
                a = torch.cat([torch.rand([1]) * (s - 1) for s in self.shape]) # generate random point a on random coordinate betweem [0, 0, 0] and [127, 127, 127]
                if side1 // dim:
                    a[side1 % dim] = 0 # set the coordinate in that dimension to 0
                else:
                    a[side1 % dim] = self.shape[side1 % dim] - 1 # set the coordinate in that dimension to 127 (max)
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
            n = torch.randint(3, (l / 5).ceil().int().clamp_min_(4), []) # number of points in spline, min number of points is 3. Not sure we div by 5 (5 px / point)

        # deform curve + sample radius
        waypoints = linspace(a, b, n)   # make vectors on range [a, b] with n steps
        sigma = (self.tortuosity() - 1) * l / (2 * dim * (n - 1)) # distort vectors according to sampled tortuosity
        sigma = sigma.clamp_min_(0)
        if sigma:
            waypoints[1:-1] += sigma * torch.randn([n - 2, dim]) # applying tortuosity manipulation to all points besides points a (waypoints[1]) and b (waypoints[-1])
        radius = radius or self.radius
        radii = radius() / self.vx
        radii = self.radius_change([n]) * radii # why are we putting the number of points into the uniform sampler? Won't this just create a distrobution around n??
        radii.clamp_min_(0.5)
        curve = BSplineCurve(waypoints, radius=radii)
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
            #MAX = torch.randint(1, 6, [])
            #print(MAX)

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
        dim = len(self.shape) # [128, 128, 128]

        # sample vessels
        volume = 1
        for s in self.shape:
            volume *= s
        volume *= (self.vx ** dim)
        density = self.tree_density()
        nb_trees = 1 #max(int(volume * density // 1), 1)
        print(f"number of trees: {nb_trees}")

        start = time.time()
        curves = []
        levels = []
        branchings = []
        nb_levels = []
        print(nb_trees)
        for n in range(nb_trees):
            nb_levels1 = torch.randint(1, 6, []) #self.nb_levels() #torch.randint(2, 6, [])
            print("nb_levels1: ", nb_levels1)
            curves1, levels1, branchings1 = self.sample_tree(max_level=nb_levels1)
            nb_levels += [max(levels1)] * len(curves1)
            curves += curves1
            levels += levels1
            branchings += branchings1
        print('sample curves: ', time.time() - start)

        # draw vessels
        start = time.time()
        curves = [c.to(self.device) for c in curves]
        vessels, labels = draw_curves(
            self.shape, curves, fast=True, mode='cosine') # sum prob, label

        levelmap = torch.zeros_like(labels)
        for i, l in enumerate(levels):
            levelmap.masked_fill_(labels == i+1, l)

        nblevelmap = torch.zeros_like(labels)
        for i, l in enumerate(nb_levels):
            nblevelmap.masked_fill_(labels == i+1, l)

        skeleton = torch.zeros_like(labels)
        for i, curve in enumerate(curves):
            ind = discretize_equidistant(curve, 0.1)
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
                if all(0 <= l < s for l, s in zip(loc, branchmap.shape)):
                    branchmap[tuple(loc)] = True

        print('draw curves: ', time.time() - start)

        vessels = vessels[None, None]
        labels = labels[None, None]
        levelmap = levelmap[None, None]                                                 # I want this to be a binary mask. Binary masks are needed for the training of the unet
        branchmap = branchmap[None, None]
        skeleton = skeleton[None, None]
        return vessels, labels, levelmap, nblevelmap, branchmap, skeleton


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
        """

        Parameters
        ----------
        shape : list[int], default=256
        voxel_size : float, default=0.1
        tree_density : Sampler
            Number of trees per mm3
            For vessels, should be 8 (in the cortex) according to known_stats
            Default: LogNormal(0.01, 0.01), 95% of samples in [0.99, 1.03]
        tortuosity : Sampler
            Expected jitter, in mm
            Default: LogNormal(5, 3), 95% of samples in [1.3, 60E3]
        radius : Sampler
            Mean radius at the first (coarsest) level
            Default: LogNormal(0.1, 0.02), 95% of samples in [1.06, 1.15]
        radius_change : Sampler
            Radius variation along the length of the spline
            Default : LogNormal(1., 0.1), 95% of samples in [2.2, 3.3]
        nb_levels : Sampler, default=2
            Number of hierarchical levels
        nb_children :
            Number of children per spline
            Default: LogNormal(5, 5)
        radius_ratio : Sampler
            Ratio between the mean radius at child and parent levels
            Default : LogNormal(0.7, 0.1), 95% of samples in [1.6, 2.5]
        """
        super().__init__(shape, voxel_size, tree_density, tortuosity,
                         radius, radius_change, nb_levels, nb_children,
                         radius_ratio, device)


class SynthVesselOCT(SynthSplineBlock):

    def __init__(
            self,
            shape=(128, 128, 128),                      # ~0.2 mm3
            voxel_size=0.01,                            # 10 um
            tree_density=random.LogNormal(0.01, 0.01),  # trees/mm3
            tortuosity=random.LogNormal(1.5, 5),        # expected jitter in mm
            radius=random.LogNormal(0.1, 0.02),         # mean radius
            radius_change=random.LogNormal(1., 0.2),    # radius variation along the vessel
            nb_levels= random.LogNormal(3, 2), #4,      # number of hierarchical level in the tree
            nb_children= 2, #random.LogNormal(5, 5),    # mean number of children
            radius_ratio=random.LogNormal(0.5, 0.1),    # Radius ratio child/parent
            device=None):

        """

        Parameters
        ----------
        shape : list[int], default=256
        voxel_size : float, default=0.1
        tree_density : Sampler
            Number of trees per mm3
            For vessels, should be 8 (in the cortex) according to known_stats
            Default: LogNormal(0.01, 0.01), 95% of samples in [0.99, 1.03]
        tortuosity : Sampler
            Expected jitter, in mm
            Default: LogNormal(5, 3), 95% of samples in [1.3, 60E3]
        radius : Sampler
            Mean radius at the first (coarsest) level
            Default: LogNormal(0.1, 0.02), 95% of samples in [1.06, 1.15]
        radius_change : Sampler
            Radius variation along the length of the spline
            Default : LogNormal(1., 0.1), 95% of samples in [2.2, 3.3]
        nb_levels : Sampler, default=2
            Number of hierarchical levels
        nb_children :
            Number of children per spline
            Default: LogNormal(5, 5)
        radius_ratio : Sampler
            Ratio between the mean radius at child and parent levels
            Default : LogNormal(0.7, 0.1), 95% of samples in [1.6, 2.5]
        """
        super().__init__(shape, voxel_size, tree_density, tortuosity,
                         radius, radius_change, nb_levels, nb_children,
                         radius_ratio, device)


class SynthAxon(SynthSplineBlock):

    def __init__(
            self,
            shape=(256, 256, 256),
            voxel_size=1e-3,                             # 1 mu
            axon_density=random.Uniform(4, 6).rpow(10),  # axons/mm3
            tortuosity=random.LogNormal(1.1, 2),         # expected jitter in mm
            radius=random.LogNormal(1e-3, 5e-4).clamp_max(2e-3),  # mean radius in mm
            radius_change=random.LogNormal(1., 0.1),     # radius variation along the axon
            nb_levels=2,                                 # number of hierarchical level in the tree
            nb_children=random.LogNormal(0.5, 1),        # mean number of children
            radius_ratio=random.LogNormal(1, 0.1),       # Radius ratio child/parent
            device=None):
        super().__init__(shape, voxel_size, axon_density, tortuosity,
                         radius, radius_change, nb_levels, nb_children,
                         radius_ratio, device)
