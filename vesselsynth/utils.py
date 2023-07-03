import torch


def default_affine(shape, voxel_size=1, **backend):
    """Generate a RAS affine matrix

    Parameters
    ----------
    shape : list[int]
        Lattice shape
    voxel_size : [sequence of] float
        Voxel size
    dtype : torch.dtype, optional
    device : torch.device, optional

    Returns
    -------
    affine : (D+1, D+1) tensor
        Affine matrix

    """
    ndim = len(shape)
    aff = torch.eye(ndim+1, **backend)
    backend = dict(dtype=aff.dtype, device=aff.device)

    # set voxel size
    voxel_size = torch.as_tensor(voxel_size, **backend).flatten()
    pad = max(0, ndim - len(voxel_size))
    pad = [voxel_size[-1:]] * pad
    voxel_size = torch.cat([voxel_size, *pad])
    voxel_size = voxel_size[:ndim]
    aff[:-1, :-1] *= voxel_size[None, :]

    # set center fov
    shape = torch.as_tensor(shape, **backend)
    aff[:-1, -1] = -voxel_size * (shape - 1) / 2

    return aff