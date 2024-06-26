__all__ = ['SynthSplineBlock', 'SynthSplineParameters']
import torch
from torch import nn
from typing import Union, List
from collections import namedtuple
from interpol import identity_grid
from synthspline.curves import BSplineCurve, BSplineCurves
from synthspline.random import AnyVar, Sampler
from synthspline import random


def setup_sampler(value):
    if isinstance(value, random.Sampler):
        return value
    elif not isinstance(value, (list, tuple)):
        return random.Dirac(value)
    else:
        return random.Uniform(*value)


class SynthSplineParameters:
    """
    A class to hold parameters (roughly modeled on `@dataclass`)

    In order to define savable parameters, one can inherit from
    `SynthSplineParameters` and define typed attributes with default
    values in the subclass. Every typed attribute can also be instantiated
    through the constructor. Hence, the class
    ```python
    class MyParams(SynthSplineParameters):
        x: int
        y: float = 3.0
    ```
    has an `__init__` method defined automatically of the form
    ```python
    def __init__(self, x: int, y: float = 3.0):
        self.x = x
        self.y = y
    ```

    Note that parameters typed `AnyVar` have a special treatment as
    they are passed through the function `setup_sampler` which ensures
    that the value is a `Sampler` (a `Dirac` sampler as a fallback).

    An object of type `SynthSplineParameters` also has a `to_dict()`
    method that returns a dictionary whose keys are its typed parameters.
    The values in the `to_dict()` dictionary can be of any type and are
    not necessarily JSON-serializable.

    If JSON serialization is needed, the object also implements a
    `serialize()` method that further serializes each value in the
    dictionary. Serializable values are recognized because they also have
    a `serialize()` method (this is the case of all `Sampler` subclasses).
    It is the user's responsibility that all values in a
    `SynthSplineParameters` either have JSON-compatible types or implement
    a `serialize()` method.
    """

    def __init__(self, *args, **kwargs):

        def is_sampler(annot):
            if annot is AnyVar:
                return True
            if isinstance(annot, type):
                return issubclass(type, Sampler)
            return False

        known_keys = list(self.__annotations__.keys())
        for key, arg in zip(known_keys, args):
            if is_sampler(self.__annotations__.get(key, object)):
                arg = setup_sampler(arg)
            setattr(self, key, arg)
        for key, val in kwargs.items():
            if is_sampler(self.__annotations__.get(key, object)):
                val = setup_sampler(val)
            setattr(self, key, val)
        for key in known_keys[len(args):]:
            if key not in kwargs:
                val = getattr(self, key)
                if is_sampler(self.__annotations__.get(key, object)):
                    val = setup_sampler(val)
                setattr(self, key, val)

    def to_dict(self) -> dict:
        """
        Returns the object in dictionary form.
        Only typed attributes are present in the dictionary.
        """
        return {
            key: val
            for key, val in self.__dict__.items()
            if key in self.__annotations__
        }

    def serialize(self, keep_tensors=False) -> dict:
        """
        Returns the object in JSON-compatible form.
        Only typed attributes are present in the dictionary.
        """
        obj = self.to_dict()
        for key, val in obj.items():
            if isinstance(val, random.Sampler):
                obj[key] = val.serialize(keep_tensors)
            if isinstance(val, torch.device):
                obj[key] = val.type
        return obj

    @classmethod
    def unserialize(cls, obj: dict):
        """
        Builds an object from its JSON-compatible form.
        """
        obj = dict(obj)
        for key, val in obj.items():
            if isinstance(val, str) and \
                    issubclass(cls.__annotations__[key], AnyVar):
                obj[key] = Sampler.unserialize(val)
        return cls(**obj)


class SynthSplineBlockV1(nn.Module):
    """
    A generic synthesis machine for spline-based trees.

    Parameters and their default values are defined in the `defaults` subclass.
    """

    ReturnedType = namedtuple('ReturnedType', [
        'prob',
        'labels',
        'levelmap',
        'nblevelmap',
        'branchmap',
        'skeleton',
        'dist',
    ])

    class defaults(SynthSplineParameters):
        shape: List[int] = (64, 64, 64)
        """Number of voxels in the synthesized patch"""
        voxel_size: float = 0.1
        """Voxel size of the patch"""
        nb_levels: AnyVar = 1
        """Number of hierarchical levels"""
        tree_density: AnyVar = random.LogNormal(mean=0.5, std=0.5)
        """
        Number of trees per mm^3
        For vessels, should be 8 (in the cortex) according to known_stats
        Default: 95% of samples in [0.6, 4.5]
        """
        tortuosity: AnyVar = random.LogNormal(mean=0.7, std=0.2)
        """
        Tortuosity ~= cord / length
        Default: 95% of samples in [1.3, 3.0]
        """
        radius: AnyVar = random.LogNormal(mean=0.07, std=0.01)
        """
        Mean radius at the first (coarsest) level
        Default: 95% of samples in [1.05, 1.09]
        """
        radius_change: AnyVar = random.LogNormal(mean=1., std=0.1)
        """
        Radius variation along the length of the spline
        Default : 95% of samples in [2.2, 3.3]
        """
        nb_children: AnyVar = random.LogNormal(mean=5, std=5)
        """
        Number of children per spline
        """
        radius_ratio: AnyVar = random.LogNormal(mean=0.7, std=0.1)
        """
        Ratio between the mean radius at child and parent levels
        Default : 95% of samples in [1.6, 2.5]
        """
        device: Union[torch.device, str] = None
        """Device to use during rasterization: "cpu" or "cuda"."""

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.params = self.defaults(*args, **kwargs)
        for key, val in self.params.to_dict().items():
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
        prob : (batch, 1, *shape) tensor[float]
            Partial volume map.
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
            Distance from each voxel to its nearest centerline.
        """
        if batch > 1:
            out = list(map(lambda x: [x], self()))
            for _ in range(batch-1):
                for i, x in enumerate(self()):
                    out[i].append(x)
            return self.ReturnedType(*map(torch.cat, out))

        import time
        dim = len(self.shape)

        # sample splines
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

        # draw splines
        start = time.time()
        curves = BSplineCurves(curves)
        curves.to(self.device)
        prob, labels, dist = curves.rasterize(self.shape, mode='cosine')
        print(f'Curves rasterized in {time.time() - start:.3f} sec')

        start = time.time()
        levelmap = torch.zeros_like(labels)
        for i, l in enumerate(levels):
            levelmap.masked_fill_(labels == i+1, l)

        nblevelmap = torch.zeros_like(labels)
        for i, l in enumerate(nb_levels):
            nblevelmap.masked_fill_(labels == i+1, l)
        print(f'Level maps computed in {time.time() - start:.3f} sec')

        start = time.time()
        skeleton = torch.zeros_like(labels)
        for i, curve in enumerate(curves):
            ind = curve.evaluate_equidistant(0.1)
            ind = ind.round().long()
            ind = ind[(ind[:, 0] >= 0) & (ind[:, 0] < skeleton.shape[0])]
            ind = ind[(ind[:, 1] >= 0) & (ind[:, 1] < skeleton.shape[1])]
            ind = ind[(ind[:, 2] >= 0) & (ind[:, 2] < skeleton.shape[2])]
            ind = ind[:, 2] \
                + ind[:, 1] * skeleton.shape[2] \
                + ind[:, 0] * skeleton.shape[2] * skeleton.shape[1]
            skeleton.view([-1])[ind] = i+1
        print(f'Skeleton computed in {time.time() - start:.3f} sec')

        start = time.time()
        branchmap = torch.zeros_like(prob)
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
        print(f'Branch map computed in {time.time() - start:.3f} sec')

        return self.ReturnedType(
            prob[None, None],
            labels[None, None],
            levelmap[None, None],
            nblevelmap[None, None],
            branchmap[None, None],
            skeleton[None, None],
            dist[None, None],
        )


SynthSplineBlock = SynthSplineBlockV1


class SynthSplineBlockV2(nn.Module):
    """
    A generic synthesis machine for spline-based trees.

    Parameters and their default values are defined in the `defaults` subclass.

    The difference with V1 is that:

    - we can sample spline orientations from a Bingham or Uniform distribution
    - instead of sampling endpoints unformely on the cube's faces, we
      sample a point uniformely inside the cube and sample an orientation
      from the Bingham distribution, and then find the intersection of the
      given line with the cube's faces.
    """

    ReturnedType = namedtuple('ReturnedType', [
        'prob',
        'labels',
        'levelmap',
        'nblevelmap',
        'branchmap',
        'skeleton',
        'dist',
    ])

    class defaults(SynthSplineParameters):
        shape: List[int] = (64, 64, 64)
        """Number of voxels in the synthesized patch"""
        voxel_size: float = 0.1
        """Voxel size of the patch"""
        nb_levels: AnyVar = 1
        """Number of hierarchical levels"""
        tree_density: AnyVar = random.LogNormal(mean=0.5, std=0.5)
        """
        Number of trees per mm^3
        For vessels, should be 8 (in the cortex) according to known_stats
        Default: 95% of samples in [0.6, 4.5]
        """
        tortuosity: AnyVar = random.LogNormal(mean=0.7, std=0.2)
        """
        Tortuosity ~= cord / length
        Default: 95% of samples in [1.3, 3.0]
        """
        radius: AnyVar = random.LogNormal(mean=0.07, std=0.01)
        """
        Mean radius at the first (coarsest) level
        Default: 95% of samples in [1.05, 1.09]
        """
        radius_change: AnyVar = random.LogNormal(mean=1., std=0.1)
        """
        Radius variation along the length of the spline
        Default : 95% of samples in [2.2, 3.3]
        """
        nb_children: AnyVar = random.LogNormal(mean=5, std=5)
        """
        Number of children per spline
        """
        radius_ratio: AnyVar = random.LogNormal(mean=0.7, std=0.1)
        """
        Ratio between the mean radius at child and parent levels
        Default : 95% of samples in [1.6, 2.5]
        """
        orient_distribution: str = 'Bingham'
        """
        Distribution from which to sample orientations:
        Bingham or Watson or Uniform
        """
        orient_variance: AnyVar = random.LogNormal(mean=10, std=50)
        """
        Variance of spline orientations about the main axes
        (Inverse of the Bingham/Watson concentration parameter).
        """
        orient_mixture: AnyVar = random.RandInt(1, 3)
        """
        Number of components in the mixture of Watson/Bingham
        """
        device: Union[torch.device, str] = None
        """Device to use during rasterization: "cpu" or "cuda"."""

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.params = self.defaults(*args, **kwargs)
        for key, val in self.params.to_dict().items():
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
        project_first_to_side = first is None
        while segment_length <= 0:

            # If not provided, sample first endpoint
            #
            #   It is provided when we are branching from another spline,
            #   or when we're on a retry du to a too short spline, in which
            #   case we keep, the original first endpoint.
            if first is None:
                # Sample a 3D coordinate inside the cube
                first = torch.rand([ndim]) * (torch.as_tensor(shape) - 1)

            # If not provided, sample orientation + last endpoint
            if last is None:
                # sample an orientation
                orient_mix = self.orient_mix_sampler().item()
                orient = self.orient_sampler[orient_mix]()
                # find intersection between line and cube
                corners = torch.as_tensor([
                    [0, 0, 0],
                    [shape[0], 0, 0],
                    [0, shape[1], 0],
                    [0, 0, shape[2]],
                ]).to(first)
                entrypoint, last = intersect_line_cube(first, orient, corners)
                if project_first_to_side:
                    first = entrypoint

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
            Randomly sampled if not provided.
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

        corners = torch.as_tensor([
            [0, 0, 0],
            [self.shape[0], 0, 0],
            [0, self.shape[1], 0],
            [0, 0, self.shape[2]],
        ])

        nb_children = self.nb_children().floor().int()
        branchings = []
        for _ in range(nb_children):
            t = torch.rand([])
            first = root.eval_position(t)
            if not point_is_inside_cube(first, corners.to(first)):
                continue
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
        prob : (batch, 1, *shape) tensor[float]
            Partial volume map.
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
            Distance from each voxel to its nearest centerline.
        """
        if batch > 1:
            out = list(map(lambda x: [x], self()))
            for _ in range(batch-1):
                for i, x in enumerate(self()):
                    out[i].append(x)
            return self.ReturnedType(*map(torch.cat, out))

        import time
        dim = len(self.shape)

        orient_mix = self.orient_mixture()
        orient_prob = random.Dirichlet([1]*orient_mix.item()).sample()
        self.orient_mix_sampler = random.Categorical(orient_prob)
        orient_sampler = self.orient_distribution.lower()[0]
        self.orient_sampler = []
        for _ in range(orient_mix):
            if orient_sampler == 'u':
                self.orient_sampler += [random.UniformSphere()]
            elif orient_sampler == 'w':
                mu = random.UniformSphere()()
                kappa = 1. / self.orient_concentration()
                self.orient_sampler += [random.Watson(mu, kappa)]
            elif orient_sampler == 'b':
                # sample orthogonal directions
                # (might be easier to sample on SO(3) from Lie algebra)
                mu = random.UniformSphere()(3)
                mu = mu.transpose(-1, -2).matmul(mu)
                _, mu = torch.linalg.eigh(mu)
                # sample concentrations
                kappa = 1. / self.orient_variance(3)
                kappa = kappa.sort(-1, descending=True).values
                kappa[..., -1] = 0
                # create sampler
                self.orient_sampler += [random.Bingham(R=mu, Z=kappa)]
            else:
                raise ValueError('Unknown orientation sampler')
        print('Orientation distribution:')
        print(self.orient_mix_sampler)
        for sampler in self.orient_sampler:
            print(sampler)

        # sample splines
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

        # draw splines
        start = time.time()
        curves = BSplineCurves(curves)
        curves.to(self.device)
        prob, labels, dist = curves.rasterize(self.shape, mode='cosine')
        print(f'Curves rasterized in {time.time() - start:.3f} sec')

        start = time.time()
        levelmap = torch.zeros_like(labels)
        for i, l in enumerate(levels):
            levelmap.masked_fill_(labels == i+1, l)

        nblevelmap = torch.zeros_like(labels)
        for i, l in enumerate(nb_levels):
            nblevelmap.masked_fill_(labels == i+1, l)
        print(f'Level maps computed in {time.time() - start:.3f} sec')

        start = time.time()
        skeleton = torch.zeros_like(labels)
        for i, curve in enumerate(curves):
            ind = curve.evaluate_equidistant(0.1)
            ind = ind.round().long()
            ind = ind[(ind[:, 0] >= 0) & (ind[:, 0] < skeleton.shape[0])]
            ind = ind[(ind[:, 1] >= 0) & (ind[:, 1] < skeleton.shape[1])]
            ind = ind[(ind[:, 2] >= 0) & (ind[:, 2] < skeleton.shape[2])]
            ind = ind[:, 2] \
                + ind[:, 1] * skeleton.shape[2] \
                + ind[:, 0] * skeleton.shape[2] * skeleton.shape[1]
            skeleton.view([-1])[ind] = i+1
        print(f'Skeleton computed in {time.time() - start:.3f} sec')

        start = time.time()
        branchmap = torch.zeros_like(prob)
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
        print(f'Branch map computed in {time.time() - start:.3f} sec')

        return self.ReturnedType(
            prob[None, None],
            labels[None, None],
            levelmap[None, None],
            nblevelmap[None, None],
            branchmap[None, None],
            skeleton[None, None],
            dist[None, None],
        )


def intersect_line_plane(line_point, line_orient, plane_point, plane_normal):
    """
    Return the intersection of a line and a plane

    !!! quote "Reference"
        https://en.wikipedia.org/wiki/Line%E2%80%93plane_intersection

    Parameters
    ----------
    line_point : (*, 3) tensor
        A point on the line
    line_orient : (*, 3) tensor
        Direction of the line
    plane_point : (*, 3) tensor
        A point on the plane
    plane_normal : (*, 3) tensor

    Returns
    -------
    intersect : (*, 3) tensor
        Point at the intersection of the line and plane
    """
    def dot(x, y):
        return x.unsqueeze(-2).matmul(y.unsqueeze(-1)).squeeze(-1).squeeze(-1)

    p0 = plane_point
    v0 = line_point
    n = plane_normal
    v = line_orient

    ln = dot(v, n)
    d = dot(p0 - v0, n) / ln
    p = v0 + v * d

    z = torch.full([], float('nan')).expand(p.shape)
    p = torch.where(ln == 0, z, p)
    return p


def point_is_inside_cube(point, corners):
    def dot(x, y):
        return x.unsqueeze(-2).matmul(y.unsqueeze(-1)).squeeze(-1).squeeze(-1)

    x = point
    p0, p1, p2, p3 = corners
    u, v, w, x = p1 - p0, p2 - p0, p3 - p0, x - p0
    ux, vx, wx = dot(u, x), dot(v, x), dot(w, x)
    return (
        (0 <= ux) & (ux <= dot(u, u)) &
        (0 <= vx) & (vx <= dot(v, v)) &
        (0 <= wx) & (wx <= dot(w, w))
    )


def point_is_inside_square(point, corners):
    # assumes point is on the plane of the square
    def dot(x, y):
        return x.unsqueeze(-2).matmul(y.unsqueeze(-1)).squeeze(-1).squeeze(-1)

    x = point
    p0, p1, p2 = corners
    u, v, x = p1 - p0, p2 - p0, x - p0
    ux, vx = dot(u, x), dot(v, x)
    return (
        (0 <= ux) & (ux <= dot(u, u)) &
        (0 <= vx) & (vx <= dot(v, v))
    )


def intersect_line_cube(line_point, line_orient, corners):
    """
    Return the intersection of a line and the sides of a cube

    Parameters
    ----------
    line_point : (*, 3) tensor
        A point on the line, that must be inside the cube.
    line_orient : (*, 3) tensor
        Direction of the line
    corners : (*, 4, 3) tensor
        Corners of the cube.

    Returns
    -------
    inter1 : (*, 3) tensor
        First intersection point.
        It is always the points that is in the negative direction of the
        line, i.e., the line gets into the cube at this point.
    inter2 : (*, 3) tensor
        Second intersection point.
        It is always the points that is in the positive direction of the
        line, i.e., the line gets out of the cube at this point.
    """
    def dot(x, y):
        return x.unsqueeze(-2).matmul(y.unsqueeze(-1)).squeeze(-1).squeeze(-1)

    x = line_point
    p0, p1, p2, p3 = corners.unbind(-2)
    u, v, w = p1 - p0, p2 - p0, p3 - p0
    p4, p5, p6, p7 = p1 + w, p1 + v, p3 + v, p1 + v + w

    if not point_is_inside_cube(line_point, (p0, p1, p2, p3)).all():
        raise ValueError('Point must be inside the cube')

    batch = torch.broadcast_shapes(
        line_point.shape[:-1],
        line_orient.shape[:-1],
        corners.shape[:-2]
    )

    inter1 = x.new_zeros(batch + x.shape[-1:])
    inter2 = x.new_zeros(batch + x.shape[-1:])

    def update_inter(points, norm):
        inter = intersect_line_plane(x, line_orient, points[0], norm)
        is_inside = point_is_inside_square(inter, points)[..., None]
        is_incoming = (dot(x - inter, w) > 0)[..., None]
        inter1.copy_(torch.where(is_inside & is_incoming, inter, inter1))
        inter2.copy_(torch.where(is_inside & ~is_incoming, inter, inter2))

    update_inter((p0, p1, p2), w)
    update_inter((p0, p1, p3), v)
    update_inter((p0, p2, p3), u)
    update_inter((p7, p4, p5), -u)
    update_inter((p4, p4, p6), -w)
    update_inter((p7, p5, p6), -v)

    return inter1, inter2
