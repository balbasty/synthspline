import torch
from typing import Union, List
from synthspline import random
from synthspline.random import AnyVar
from synthspline.labelsynth import (
    SynthSplineBlock, SynthSplineBlockV2, SynthSplineParameters
)


class SynthVesselMicro(SynthSplineBlock):

    class defaults(SynthSplineParameters):
        shape: List[int] = (256, 256, 256)
        """Number of voxels in the patch (256 -> ~16 mm3 @ 10 micron)"""
        voxel_size: float = 0.01
        """10 micron"""
        nb_levels: AnyVar = 5
        """Number of hierarchical level in the tree. Could be random."""
        tree_density = random.LogNormal(mean=8, std=1)
        """Trees/mm3"""
        tortuosity: AnyVar = random.LogNormal(mean=2, std=1)
        """Tortuosity ~= cord / length"""
        radius: AnyVar = random.LogNormal(mean=0.07, std=0.01)
        """Mean radius, in mm, at coarsest level"""
        radius_change: AnyVar = random.LogNormal(mean=1., std=0.1)
        """Radius variation along the spline"""
        nb_children: AnyVar = random.LogNormal(mean=5, std=5)
        """Number of branches per spline"""
        radius_ratio: AnyVar = random.LogNormal(mean=0.7, std=0.1)
        """Radius ratio branch/parent"""
        device: Union[torch.device, str] = None


class SynthVesselHiResMRI(SynthSplineBlock):

    class defaults(SynthSplineParameters):
        shape: List[int] = (256, 256, 256)
        """Number of voxels in the patch (256 -> ~16 cm3 @ 100 micron)"""
        voxel_size: float = 0.1
        """100 micron"""
        nb_levels: AnyVar = 2
        """Number of hierarchical level in the tree. Could be random."""
        tree_density: AnyVar = random.LogNormal(mean=0.01, std=0.01)
        """Trees/mm3"""
        tortuosity: AnyVar = random.LogNormal(mean=5, std=3)
        """Tortuosity ~= cord / length"""
        radius: AnyVar = random.LogNormal(mean=0.1, std=0.02)
        """Mean radius, in mm, at coarsest level"""
        radius_change: AnyVar = random.LogNormal(mean=1., std=0.1)
        """Radius variation along the spline"""
        nb_children: AnyVar = random.LogNormal(mean=5, std=5)
        """Number of branches per spline"""
        radius_ratio: AnyVar = random.LogNormal(mean=0.7, std=0.1)
        """Radius ratio branch/parent"""
        device: Union[torch.device, str] = None


class SynthVesselOCT(SynthSplineBlock):

    class defaults(SynthSplineParameters):
        shape: List[int] = (128, 128, 128)
        """Number of voxels in the patch (256 -> ~2 mm3 @ 10 micron)"""
        voxel_size: float = 0.01
        """10 micron"""
        nb_levels: AnyVar = 4
        """Number of hierarchical level in the tree. Could be random."""
        tree_density: AnyVar = random.LogNormal(mean=0.01, std=0.01)
        """Trees/mm3"""
        tortuosity: AnyVar = random.LogNormal(mean=1.5, std=5)
        """Tortuosity ~= cord / length"""
        radius: AnyVar = random.LogNormal(mean=0.1, std=0.02)
        """Mean radius, in mm, at coarsest level"""
        radius_change: AnyVar = random.LogNormal(mean=1., std=0.2)
        """Radius variation along the spline"""
        nb_children: AnyVar = random.LogNormal(mean=2, std=3)
        """Number of branches per spline"""
        radius_ratio: AnyVar = random.LogNormal(mean=0.5, std=0.1)
        """Radius ratio branch/parent"""
        device: Union[torch.device, str] = None


class SynthVesselPhoto(SynthSplineBlock):

    class defaults(SynthSplineParameters):
        shape: List[int] = (256, 256, 256)
        """Number of voxels in the patch (256 -> ~450 mm3 @ 30 micron)"""
        voxel_size: float = 0.03
        """30 micron"""
        nb_levels: AnyVar = random.RandInt(min=1, max=5)
        """Number of hierarchical level in the tree. Could be random."""
        tree_density: AnyVar = random.LogNormal(mean=0.1, std=0.2)
        """Trees/mm3"""
        tortuosity: AnyVar = random.LogNormal(mean=1, std=5)
        """Tortuosity ~= cord / length"""
        radius: AnyVar = random.LogNormal(mean=0.1, std=0.02)
        """Mean radius, in mm, at coarsest level"""
        radius_change: AnyVar = random.LogNormal(mean=1., std=0.2)
        """Radius variation along the spline"""
        nb_children: AnyVar = random.LogNormal(mean=2, std=3)
        """Number of branches per spline"""
        radius_ratio: AnyVar = random.LogNormal(mean=0.5, std=0.1)
        """Radius ratio branch/parent"""
        device: Union[torch.device, str] = None


class SynthAxon(SynthSplineBlockV2):

    class defaults(SynthSplineParameters):
        shape: List[int] = (256, 256, 256)
        """Number of voxels in the patch (256 -> 16E-3 mm3 @ 1 micron)"""
        voxel_size: float = 1e-3
        """1 micron"""
        nb_levels: AnyVar = 2
        """Number of hierarchical level in the tree. Could be random."""
        tree_density: AnyVar = 10 ** random.Uniform(min=4, max=6)
        """Trees/mm3"""
        tortuosity: AnyVar = random.LogNormal(mean=1.1, std=2)
        """Tortuosity ~= cord / length"""
        radius: AnyVar = random.LogNormal(mean=1e-3, std=5e-4).clamp_max(2e-3)
        """Mean radius, in mm, at coarsest level"""
        radius_change: AnyVar = random.LogNormal(mean=1., std=0.1)
        """Radius variation along the spline"""
        nb_children: AnyVar = random.LogNormal(mean=0.5, std=1)
        """Number of branches per spline"""
        radius_ratio: AnyVar = random.LogNormal(mean=1, std=0.1)
        """Radius ratio branch/parent"""
        orient_distribution: str = 'Bingham'
        """Distribution from which to sample orientations"""
        orient_variance: AnyVar = random.LogNormal(mean=10, std=50)
        """Orientation variance (inverse of concetration)"""
        orient_mixture: AnyVar = random.RandInt(1, 3)
        """Number of components in the mixture of Watson/Bingham"""
        device: Union[torch.device, str] = None
