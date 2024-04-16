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


def make_list(x):
    if not isinstance(x, (list, tuple)):
        x = [x]
    x = list(x)
    return x


def make_tuple(x):
    if not isinstance(x, (list, tuple)):
        x = (x,)
    x = tuple(x)
    return x


def pyflatten(x):
    if isinstance(x, (list, tuple)):
        out = []
        for elem in x:
            out += pyflatten(elem)
        return out
    return [x]


_dtype_order = (
    torch.complex128, torch.complex64, torch.complex32,
    torch.float64, torch.float32, torch.float16,
    torch.int64, torch.int32, torch.int16, torch.int8, torch.uint8,
    torch.bool, type(None), None
)
_pytype_order = (complex, float, int, bool, type(None), None)


def _pytype_map(pytype):
    torch_default = torch.get_default_dtype()
    if pytype is complex:
        return (torch.complex32 if torch_default is torch.float16 else
                torch.complex64 if torch_default is torch.float32 else
                torch.complex128)
    if pytype is float:
        return torch.get_default_dtype()
    if pytype is int:
        return torch.long
    if pytype is bool:
        return torch.bool
    return None


def to_tensor(*x, dtype=None, device=None):
    """
    Convert one or more values to a tensor.

    !!! tip "Upcasting"
        * If no data type is provided (`dtype=None`), this function will
          convert all variables to the "largest" data type in the list,
          with priority given to tensors over builtin python types.
        * If a `torch.dtype` is passed, it converts all variables to tensors
          of this exact type.
        * If instead a python builtin type (`complex`, `float`, `int`, `bool`)
          is passed, the function ensures that the output data type has the
          correct "meta-type". That is, if `dtype=float`, and all inputs
          are integers, it will convert them to `torch.get_default_dtype()`,
          but if at least one input is of type `torch.float64`, the normal
          upcasting behaviour is preserved.
        * A similar strategy is used to convert all variables to tensors
          that live on the same device.

    !!! warning "This function does not handle numpy arrays"

    Parameters
    ----------
    *x : tensor or [sequence of] number
        One or several values to convert to tensor.
    dtype : type or torch.dtype
        Target data type
    device : torch.device
        Target device

    Returns
    -------
    *x : tensor
        One or several tensors
    """
    def has_complex(x):
        if torch.is_tensor(x):
            return x.dtype.is_complex
        if isinstance(x, complex):
            return True
        if isinstance(x, (list, tuple)):
            return any(map(has_complex, x))
        return False

    def has_float(x):
        if torch.is_tensor(x):
            return x.dtype.is_floating_point
        if isinstance(x, float):
            return True
        if isinstance(x, (list, tuple)):
            return any(map(has_float, x))
        return False

    def has_signed(x):
        if torch.is_tensor(x):
            return x.dtype.is_signed
        if isinstance(x, int):
            return True
        if isinstance(x, (list, tuple)):
            return any(map(has_signed, x))
        return False

    def has_number(x):
        if torch.is_tensor(x):
            return x.dtype is not torch.bool
        if isinstance(x, (complex, float, int)):
            return True
        if isinstance(x, (list, tuple)):
            return any(map(has_number, x))
        return False

    def get_metatype(x):
        return (
            complex if has_complex(x) else
            float if has_float(x) else
            int if has_signed(x) else
            bool if not has_number else
            None
        )

    metatype = None
    if isinstance(dtype, type):
        dtype, metatype = None, dtype

    if len(x) == 0:
        return None

    # pass 0: specified
    if dtype and device:
        out = tuple(torch.as_tensor(elem, dtype=dtype, device=device)
                    for elem in x)
        return out[0] if len(out) == 1 else out

    # pass 1: find device
    if not device:
        for elem in x:
            if not torch.is_tensor(elem):
                continue
            maybe_device = elem.device
            if not device or device.type == 'cpu':
                device = maybe_device
            elif maybe_device.type != 'cpu':
                if device.index is None:
                    device = maybe_device
                elif maybe_device.index is None:
                    pass
                elif maybe_device.index > device.index:
                    device = maybe_device

    # pass 2: find dtype
    if not dtype:
        for elem in x:
            if not torch.is_tensor(elem):
                continue
            maybe_dtype = elem.dtype
            if _dtype_order.index(maybe_dtype) < _dtype_order.index(dtype):
                dtype = maybe_dtype

    # pass 3: find python objects
    if not dtype:
        dtypes = list(map(type, pyflatten(x)))
        for maybe_dtype in dtypes:
            if maybe_dtype not in _pytype_order:
                continue
            if _pytype_order.index(maybe_dtype) < _pytype_order.index(dtype):
                dtype = maybe_dtype
        dtype = _pytype_map(dtype)

    # still nothing: error
    if not dtype:
        raise TypeError('Could not find an object with a known type')

    # pass 4: check type family
    metatype = metatype or get_metatype(x)
    if metatype is complex and not dtype.is_complex:
        dtype = _pytype_map(complex)
    elif metatype is float and not dtype.is_floating_point:
        dtype = _pytype_map(float)
    elif metatype is int and not dtype.is_signed:
        dtype = _pytype_map(int)
    elif metatype is not bool and dtype is torch.bool:
        dtype = torch.uint8

    out = tuple(torch.as_tensor(elem, dtype=dtype, device=device)
                for elem in x)
    return out[0] if len(out) == 1 else out
