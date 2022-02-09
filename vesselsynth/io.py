import nibabel.freesurfer.io as fsio
from nitorch.io.transforms import conversions
import nitorch.core.dtypes as ni_dtype
from nitorch import spatial
import torch


def load_mesh(fname, return_space=False, numpy=False):
    """Load a mesh in memory

    Parameters
    ----------
    fname : str
        Path to surface file

    return_space : bool, default=False
        Return the affine matrix and shape of the original volume

    numpy : bool, default=False
        Return numpy array instead of torch tensor

    Returns
    -------
    coord : (N, D) tensor
        Node coordinates.
        Each node has a coordinate in an ambient space.

    faces : (M, K) tensor
        Faces.
        Each face is made of K nodes, whose indices are stored in this tensor.
        For triangular meshes, K = 3.

    affine : (D+1, D+1) tensor, if `return_space`
        Mapping from the `coord`'s ambient space to a standard space.
        In Freesurfer surfaces, edges coordinates are also expressed in
        voxels of the original volumetric file, in which case the affine
        maps these voxel coordinates to millimetric RAS coordinates.

    shape : (D,) list[int], if `return_space`
        Shape of the original volume.

    """
    c, f, *meta = fsio.read_geometry(fname, read_metadata=return_space)

    if not numpy:
        if not ni_dtype.dtype(c.dtype).is_native:
            c = c.newbyteorder().byteswap(inplace=True)
        if not ni_dtype.dtype(f.dtype).is_native:
            f = f.newbyteorder().byteswap(inplace=True)
        c = torch.as_tensor(c, dtype=ni_dtype.dtype(c.dtype).torch_upcast)
        f = torch.as_tensor(f, dtype=ni_dtype.dtype(f.dtype).torch_upcast)

    if not return_space:
        return c, f

    shape = None
    if 'volume' in meta:
        shape = torch.as_tensor(meta['volume']).tolist()
    if 'cras' in meta:
        x, y, z, c = meta['xras'], meta['yras'], meta['zras'], meta['cras']
        aff = conversions.XYZC(x, y, z, c).affine().matrix
        aff = torch.as_tensor(aff, dtype=torch.float32)
    else:
        aff = torch.eye(c.shape[-1])
    if 'voxelsize' in meta:
        vx = torch.as_tensor(meta['voxelsize'])
        aff = spatial.affine_matmul(aff, vx)

    return c, f, aff, shape


def load_overlay(fname, numpy=False):
    """Load an overlay (= map from vertex to scalar/vector value)

    Parameters
    ----------
    fname : str
        Path to overlay file

    numpy : bool, default=False
        Return numpy array instead of torch tensor

    Returns
    -------
    overlay : (N, [K]) tensor
        N is the number of vertices
        K is the dimension of the value space (or number of frames)

    """
    o = fsio.read_morph_data(fname)
    if not numpy:
        if not ni_dtype.dtype(o.dtype).is_native:
            o = o.newbyteorder().byteswap(inplace=True)
        o = torch.as_tensor(o, dtype=ni_dtype.dtype(o.dtype).torch_upcast)
    return o