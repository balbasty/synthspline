__all__ = [
    'Sampler',
    'Op1', 'Op2', 'OpN',
    'Dirac',
    'Uniform',
    'RandInt',
    'Normal',
    'LogNormal',
]
import math
import torch
from torch import distributions, Tensor, get_default_dtype
from typing import Union, Callable, Optional, Tuple
from numbers import Number
from synthspline.utils import make_tuple, to_tensor, import_fullname


pymin, pymax = min, max


def _get_aliases(kwargs, keys, default=None):
    for key in keys:
        if key in kwargs:
            return kwargs[key]
    return default


def min(*args):
    """
    Return a random variable that is the minimum of multiple variables.
    """
    if len(args) == 0:
        return None
    if len(args) == 1:
        return args[0]
    x, *args = args
    while args:
        y, *args = args
        if isinstance(x, Sampler):
            x = x.minimum(y)
        elif isinstance(y, Sampler):
            x = y.minimum(x)
        else:
            x = pymin(x, y)
    return x


def max(*args):
    """
    Return a random variable that is the maximum of multiple variables.
    """
    if len(args) == 0:
        return None
    if len(args) == 1:
        return args[0]
    x, *args = args
    while args:
        y, *args = args
        if isinstance(x, Sampler):
            x = x.maximum(y)
        elif isinstance(y, Sampler):
            x = y.maximum(x)
        else:
            x = pymax(x, y)
    return x


class RandomVar:
    """
    Base class for random variables.

    Nothing is defined here, mostly used for type hints.
    """
    pass


AnyVar = Union[Number, Tensor, RandomVar]
"""
Either a deterministic variable (a number) or a random variable (a sampler).
"""


class Sampler(RandomVar):
    """
    Base class for all random samplers.

    !!! note
        - Most samplers have a set of fixed hyper-parameters, for example,
          the mean and standard deviation of the Normal distribution.

        - All hyper-parameters should be tensors, or will be converted to
          tensors. All samplers return a tensor, even when the
          hyper-parameters passed by the user are not. The type and device
          of the samples are generally the same as those of the
          hyper-parameters, but can be overrided at call-time.

        - All samplers should be thought of as random variables, and new
          random variables can be derived by applying determinstic or random
          functions to other random variables. All supported functions are
          defined in this class.

    !!! tip "Sampling"
        All samplers generate values by calling them.

        - If they are called without arguments, they return a scalar tensor.
        - If they are called with an integer, they return a vector tensor
          with this length.
        - If they are called with a list of integers, they return a tensor
          of values with this shape.

        ```python
        sampler = Normal()
        sampler()           # ~ tensor(0.3)
        sampler(3)          # ~ tensor([0.3, -0.2, 1.5])
        sampler([1, 3])     # ~ tensor([[0.3, -0.2, 1.5]])
        ```

    !!! warning "Equality"
        The `==` and `!=` operations are functions that transform the
        random variable, like any others. They do not test if the
        _distributions are identical_ but generate a random variable
        that returns `True` if _the realizations from its parents are
        identical_.

    Attributes
    ----------
    param : Parameters
        The sampler parameters and derived metrics.

        It has methods `mean()`, `var()`, `std()`, `min()` and `max()`
        and attribute `shape`. Each specialized sampler also adds attributes
        specific to its parameters (for example, the `Normal` sampler
        adds `mu` and `sigma` attributes). Many samplers also implement
        the attributes `loc` and `scale`, whose use is standard in
        distributions from the exponential family.
    """

    def serialize(self, keep_tensors=False):
        """
        Serialize object into a dictionary

        If `keep_tensors`, the dictionary must be written to disk with
        `torch.save`. Otherwise, it can be written to disk with `json.dump`.
        """
        return {type(self).__name__: self.param.serialize(keep_tensors)}

    @classmethod
    def unserialize(cls, obj):
        """
        Unserialize a Sampler.

        If `cls` is `Sampler`, assumes that `obj` contains the sampler
        name as key and its arguments as value.

        If `cls` is a `Sampler` subclass, assumes that obj contains
        its arguments.
        """
        if cls is Sampler:
            if not isinstance(obj, dict) or len(obj) != 1:
                raise ValueError('Cannot interpret this object as a Sampler')
            (sampler, args), = obj.items()
            if '.' not in sampler:
                sampler = 'vesselsynth.random.' + sampler
            sampler = import_fullname(sampler)
            return sampler.unserialize(args)
        elif isinstance(obj, dict):
            return cls(**obj)
        elif isinstance(obj, (list, tuple)):
            return cls(*obj)
        else:
            return cls(obj)

    class Parameters:
        """
        An object that contains the parameters of a sampler, and knows
        how to compute derived metrics such as the mean or variance.

        Attributes
        ----------
        ref
            Reference to the Sampler object.
        shape
            Shape of the parameter tensor (broadcasted if multiple parameters).
        loc
            Location parameter.
        scale
            Scale parameter.
        """
        ref: "Sampler" = None
        shape: torch.Size = None
        loc: Tensor = None
        scale: Tensor = None
        _keys: Tuple[str] = tuple()

        def __init__(self, ref, **kwargs) -> None:
            self.ref = ref
            for key, value in kwargs.items():
                setattr(self, key, value)
            self._keys = tuple(kwargs.keys())

        def serialize(self, keep_tensors=False):
            """
            Serialize object into a dictionary

            If `keep_tensors`, the dictionary must be written to disk
            with `torch.save`. Otherwise, it can be written to disk
            with `json.dump`.
            """
            json = {}
            for key in self._keys:
                val = getattr(self, key)
                if torch.is_tensor(val):
                    # Tensor, which we may or may not serialize as a list
                    if not keep_tensors:
                        val = val.tolist()
                elif hasattr(val, 'serialize'):
                    # Serializable object
                    val = val.serialize(keep_tensors)
                elif callable(val):
                    # Let's hope it's importable
                    val = val.__module__ + '.' + val.__name__
                json[key] = val
            return json

        def mean(self,
                 nb_samples: Optional[int] = None,
                 max_samples: Optional[int] = None) -> Tensor:
            """
            Compute the expected value of the distribution.

            Parameters
            ----------
            nb_samples : int
                Number of samples used for monte-carlo estimation.
                If `None` (default), use either the exact value if it exists,
                or assume a peaked distribution.
            max_samples : int
                Maximum number of samples to hold in memory.
                If `None` (default), generate all samples at once, which
                may use up a lot of memory.
            """
            if nb_samples:
                if max_samples and max_samples < nb_samples:
                    x, n = 0, 0
                    while n < nb_samples:
                        n1 = min(max_samples, nb_samples - n)
                        x += self.ref(n1).sum(0)
                        n += n1
                    x /= n
                else:
                    x = self.ref(nb_samples).mean(0)
                return x
            return NotImplemented

        def var(self,
                nb_samples: Optional[int] = None,
                max_samples: Optional[int] = None) -> Tensor:
            """
            Compute the variance of the distribution.

            Parameters
            ----------
            nb_samples : int
                Number of samples used for monte-carlo estimation.
                If `None` (default), use either the exact value if it exists,
                or assume a peaked distribution.
            max_samples : int
                Maximum number of samples to hold in memory.
                If `None` (default), generate all samples at once, which
                may use up a lot of memory.
            """
            if nb_samples:
                if max_samples and max_samples < nb_samples:
                    x, n = 0, 0
                    while n < nb_samples:
                        n1 = min(max_samples, nb_samples - n)
                        x += self.ref(n1).square_().sum(0)
                        n += n1
                    x /= n
                else:
                    x = self.ref(nb_samples).var(0)
                return x
            return NotImplemented

        def std(self,
                nb_samples: Optional[int] = None,
                max_samples: Optional[int] = None) -> Tensor:
            """
            Compute the standard deviation of the distribution.

            Parameters
            ----------
            nb_samples : int
                Number of samples used for monte-carlo estimation.
                If `None` (default), use either the exact value if it exists,
                or assume a peaked distribution.
            max_samples : int
                Maximum number of samples to hold in memory.
                If `None` (default), generate all samples at once, which
                may use up a lot of memory.
            """
            return self.var(nb_samples, max_samples).sqrt_()

        def min(self,
                nb_samples: Optional[int] = None,
                max_samples: Optional[int] = None) -> Tensor:
            """
            Compute the minimum of the distribution's support.

            Parameters
            ----------
            nb_samples : int
                Number of samples used for monte-carlo estimation.
                If `None` (default), use either the exact value if it exists,
                or assume a peaked distribution.
            max_samples : int
                Maximum number of samples to hold in memory.
                If `None` (default), generate all samples at once, which
                may use up a lot of memory.
            """
            if nb_samples:
                if max_samples and max_samples < nb_samples:
                    x = self.ref(max_samples).min(0).values
                    n = max_samples
                    while n < nb_samples:
                        n1 = min(max_samples, nb_samples - n)
                        x = x.minimum(self.ref(n1).min(0).values)
                        n += n1
                    x /= n
                else:
                    x = self.ref(nb_samples).min(0).values
                return x
            return NotImplemented

        def max(self,
                nb_samples: Optional[int] = None,
                max_samples: Optional[int] = None) -> Tensor:
            """
            Compute the maximum of the distribution's support.

            Parameters
            ----------
            nb_samples : int
                Number of samples used for monte-carlo estimation.
                If `None` (default), use either the exact value if it exists,
                or assume a peaked distribution.
            max_samples : int
                Maximum number of samples to hold in memory.
                If `None` (default), generate all samples at once, which
                may use up a lot of memory.
            """
            if nb_samples:
                if max_samples and max_samples < nb_samples:
                    x = self.ref(max_samples).max(0).values
                    n = max_samples
                    while n < nb_samples:
                        n1 = min(max_samples, nb_samples - n)
                        x = x.maximum(self.ref(n1).max(0).values)
                        n += n1
                    x /= n
                else:
                    x = self.ref(nb_samples).max(0).values
                return x
            return NotImplemented

    param: Parameters

    def __call__(self, batch=tuple(), **backend) -> Tensor:
        """
        Sample from the distribution

        Parameters
        ----------
        batch : tuple[int], optional
            Batch size
        dtype : torch.dtype, optional
            Data type
        device : torch.dtype, optional
            Device

        Returns
        -------
        sample : tensor
            Tensor of values, with shape `(*batch, *shape)`, where
            `shape` is the broadcasted shape of the parameter(s).
        """
        return NotImplemented

    def to_str(self):
        args = []
        for key in self.param._keys:
            val = getattr(self.param, key)
            if hasattr(val, 'to_str'):
                val = val.to_str()
            args += [f'{key}={val}']
        args = ', '.join(args)
        return f'{type(self).__name__}({args})'

    __repr__ = __str__ = to_str

    def exp(self) -> RandomVar:
        """
        Exponential
        """
        return Op1(self, torch.exp)

    def log(self) -> RandomVar:
        """
        Natural logarithm
        """
        return Op1(self, torch.log)

    def sqrt(self) -> RandomVar:
        """
        Square root. Can also be called via `x ** 0.5`.
        """
        return Op1(self, torch.sqrt)

    def square(self) -> RandomVar:
        """
        Square. Can also be called via `x ** 2`.
        """
        return Op1(self, torch.square)

    def pow(self, other: AnyVar) -> RandomVar:
        """
        Take the power of the left var. Can also be called via `x ** other`.
        """
        return Op2(self, other, torch.pow)

    def rpow(self, other: AnyVar) -> "Sampler":
        """
        Take the power of the right var. Can also be called via `other ** x`.
        """
        return Op2(other, self, torch.pow)

    def add(self, other: AnyVar) -> RandomVar:
        """
        Sum of two variables. Can also be called via `x + other`.
        """
        return Op2(self, other, torch.add)

    def sub(self, other: AnyVar) -> RandomVar:
        """
        Difference of two variables. Can also be called via `x - other`.
        """
        return Op2(self, other, torch.sub)

    def mul(self, other: AnyVar) -> RandomVar:
        """
        Product of two variables. Can also be called via `x * other`.
        """
        return Op2(self, other, torch.mul)

    def div(self, other: AnyVar) -> RandomVar:
        """
        Ratio of two variables. Can also be called via `x / other`.
        """
        return Op2(self, other, torch.div)

    def floordiv(self, other: AnyVar) -> RandomVar:
        """
        Floor division. Can also be called via `x // other`.
        """
        return Op2(self, other, torch.floor_divide)

    def matmul(self, other: AnyVar) -> RandomVar:
        """
        Matrix product. Can also be called via `x @ other`.
        """
        return Op2(self, other, torch.matmul)

    def minimum(self, other: AnyVar) -> RandomVar:
        """
        Minimum of two variables.
        """
        return Op2(self, other, torch.minimum)

    def maximum(self, other: AnyVar) -> RandomVar:
        """
        Maximum of two variables.
        """
        return Op2(self, other, torch.maximum)

    def equal(self, other: AnyVar) -> RandomVar:
        """
        Whether left values are equal to the right values.
        Can also be called via `x == other`.
        """
        return Op2(self, other, torch.less)

    def not_equal(self, other: AnyVar) -> RandomVar:
        """
        Whether left values are different from the right values.
        Can also be called via `x != other`.
        """
        return Op2(self, other, torch.less)

    def less(self, other: AnyVar) -> RandomVar:
        """
        Whether left values are less than right values (exclusive).
        Can also be called via `x < other`.
        """
        return Op2(self, other, torch.less)

    def less_equal(self, other: AnyVar) -> RandomVar:
        """
        Whether left values are less than right values (inclusive).
        Can also be called via `x <= other`.
        """
        return Op2(self, other, torch.less_equal)

    def greater(self, other: AnyVar) -> RandomVar:
        """
        Whether left values are greater than right values (exclusive).
        Can also be called via `x > other`.
        """
        return Op2(self, other, torch.greater)

    def greater_equal(self, other: AnyVar) -> RandomVar:
        """
        Whether left values are greater than right values (inclusive).
        Can also be called via `x >= other`.
        """
        return Op2(self, other, torch.greater_equal)

    def clamp(self, min: AnyVar, max: AnyVar) -> RandomVar:
        """
        Clamp values outside a range.
        """
        return self.minimum(max).maximum(min)

    def clamp_min(self, other: AnyVar) -> RandomVar:
        """
        Clamp values below a threshold.
        """
        return self.maximum(other)

    def clamp_max(self, other: AnyVar) -> RandomVar:
        """
        Clamp values above a threshold.
        """
        return self.minimum(other)

    def __pow__(self, other: AnyVar, modulo=None) -> RandomVar:
        if modulo is not None:
            raise NotImplementedError('pow+modulo not implemented')
        return self.pow(other)

    def __rpow__(self, other: AnyVar) -> RandomVar:
        return self.rpow(other)

    def __add__(self, other: AnyVar) -> RandomVar:
        return self.add(other)

    def __sub__(self, other: AnyVar) -> RandomVar:
        return self.sub(other)

    def __mul__(self, other: AnyVar) -> RandomVar:
        return self.mul(other)

    def __truediv__(self, other: AnyVar) -> RandomVar:
        return self.div(other)

    def __floordiv__(self, other: AnyVar) -> RandomVar:
        return self.floordiv(other)

    def __matmul__(self, other: AnyVar) -> RandomVar:
        return self.matmul(other)

    def __eq__(self, other: AnyVar) -> RandomVar:
        return self.equal(other)

    def __ne__(self, other: AnyVar) -> RandomVar:
        return self.not_equal(other)

    def __lt__(self, other: AnyVar) -> RandomVar:
        return self.less(other)

    def __le__(self, other: AnyVar) -> RandomVar:
        return self.less_equal(other)

    def __gt__(self, other: AnyVar) -> RandomVar:
        return self.greater(other)

    def __ge__(self, other: AnyVar) -> RandomVar:
        return self.greater_equal(other)


class Op1(Sampler):
    """
    A sampler obtained by applying a function to another sampler.
    """

    @classmethod
    def unserialize(cls, obj):
        if isinstance(obj, dict):
            sampler = obj['sampler']
            op = obj['op']
        else:
            sampler, op = obj
        if isinstance(sampler, dict):
            sampler = Sampler.unserialize(sampler)
        if isinstance(op, str):
            op = import_fullname(op)
        return cls(sampler, op)

    class Parameters(Sampler.Parameters):

        @property
        def shape(self) -> torch.Size:
            return self.sampler.param.shape

        def mean(self, nb_samples=None, *args, **kwargs):
            if nb_samples:
                return super().mean(nb_samples, *args, **kwargs)
            # approximation: assumes peaked sampler
            return self.op(self.sampler.param.mean())

        def min(self, nb_samples=None, *args, **kwargs):
            if nb_samples:
                return super().min(nb_samples, *args, **kwargs)
            # approximation: assumes monotonic function
            return torch.minimum(self.op(self.sampler.param.min()),
                                 self.op(self.sampler.param.max()))

        def max(self, nb_samples=None, *args, **kwargs):
            if nb_samples:
                return super().max(nb_samples, *args, **kwargs)
            # approximation: assumes monotonic function
            return torch.maximum(self.op(self.sampler.param.min()),
                                 self.op(self.sampler.param.max()))

    def __init__(self, sampler: AnyVar, op: Callable[[Tensor], Tensor]):
        """
        Parameters
        ----------
        sampler : number or Sampler
            A (random) variable
        op : callable
            A function that takes a tensor as input and returns a tensor
        """
        if not isinstance(sampler, Sampler):
            sampler = Dirac(sampler)
        self.param = self.Parameters(self, sampler=sampler, op=op)

    def __call__(self, batch=tuple(), **backend):
        return self.param.op(self.param.sampler(batch, **backend))


class Op2(Sampler):
    """
    A sampler obtained by applying a function to two other samplers.
    """

    @classmethod
    def unserialize(cls, obj):
        if isinstance(obj, dict):
            sampler1 = obj['sampler1']
            sampler2 = obj['sampler2']
            op = obj['op']
        else:
            sampler1, sampler2, op = obj
        if isinstance(sampler1, dict):
            sampler1 = Sampler.unserialize(sampler1)
        if isinstance(sampler2, dict):
            sampler2 = Sampler.unserialize(sampler2)
        if isinstance(op, str):
            op = import_fullname(op)
        return cls(sampler1, sampler2, op)

    class Parameters(Sampler.Parameters):

        @property
        def shape(self) -> torch.Size:
            return torch.broadcast_shapes(self.sampler1.param.shape,
                                          self.sampler2.param.shape)

        def mean(self, nb_samples=None, *args, **kwargs):
            if nb_samples:
                return super().mean(nb_samples, *args, **kwargs)
            # approximation: assumes peaked sampler
            return self.op(self.sampler1.param.mean(),
                           self.sampler2.param.mean())

    def __init__(self, sampler1: AnyVar, sampler2: AnyVar,
                 op: Callable[[Tensor, Tensor], Tensor]):
        """
        Parameters
        ----------
        sampler1 : number or Sampler
            A (random) variable
        sampler2 : number or Sampler
            A (random) variable
        op : callable
            A function that takes two tensors as input and returns a tensor
        """
        if not isinstance(sampler1, Sampler):
            sampler1 = Dirac(sampler1)
        if not isinstance(sampler2, Sampler):
            sampler2 = Dirac(sampler2)
        self.param = self.Parameters(
            self, sampler1=sampler1, sampler2=sampler2, op=op)

    def __call__(self, batch=tuple(), **backend):
        x = self.param.sampler1(batch, **backend)
        y = self.param.sampler2(batch, **backend)
        x, y = to_tensor(x, y, **backend)
        return self.param.op(x, y)


class SamplerList(list):
    """Utility class to hold a serializable list of Samplers"""
    def serialize(self, keep_tensors=False):
        return [x.serialize(keep_tensors) if isinstance(x, Sampler) else x
                for x in self]


class OpN(Sampler):
    """
    A sampler obtained by applying a function to `N` other samplers.
    """

    @classmethod
    def unserialize(cls, obj):
        if isinstance(obj, dict):
            samplers = obj['samplers']
            op = obj['op']
        else:
            samplers, op = obj
        samplers = [
            Sampler.unserialize(sampler) if isinstance(sampler, dict)
            else sampler for sampler in samplers
        ]
        if isinstance(op, str):
            op = import_fullname(op)
        return cls(samplers, op)

    class Parameters(Sampler.Parameters):

        @property
        def shape(self) -> torch.Size:
            return torch.broadcast_shapes(*[
                f.param.shape for f in self.samples
            ])

        def mean(self, nb_samples=None, *args, **kwargs):
            if nb_samples:
                return super().mean(nb_samples, *args, **kwargs)
            # approximation: assumes peaked sampler
            return self.op([f.param.mean() for f in self.samplers])

    def __init__(self, samplers, op: Callable[..., Tensor]):
        """
        Parameters
        ----------
        samplers : list[number or Sampler]
            A list of (random) variables
        op : callable
            A function that takes N tensors as input and returns a tensor
        """
        samplers = SamplerList([
            Dirac(f) if not isinstance(f, Sampler) else f for f in samplers
        ])
        self.param = self.Parameters(self, samplers=samplers, op=op)

    def __call__(self, batch=tuple(), **backend):
        x = [f(batch, **backend) for f in self.param.samplers]
        x = to_tensor(*x, **backend)
        return self.param.op(*x)


class Dirac(Sampler):
    """
    A fixed value

    !!! example "Signatures"
        ```python
        Dirac()               # default (0)
        Dirac(0)              # positional variant
        Dirac(value=0)        # keyword variant
        Dirac(loc=0)          # alias
        Dirac(mean=0)         # alias
        ```
    """

    class Parameters(Sampler.Parameters):

        @property
        def shape(self) -> torch.Size:
            return self.value.shape

        def mean(self, *args, **kwargs):
            return self.value

        def var(self, *args, **kwargs):
            return self.value.new_zeros([])

        scale = std = var
        loc = min = max = mean

    def __init__(self, *args, **kwargs):
        if len(args) > 1:
            raise ValueError('Too many positional arguments')
        if len(kwargs) > 1:
            raise ValueError('Too many keyword arguments')
        if len(args) == 1 and kwargs:
            raise ValueError('Positional and keyword arguments')
        if any(map(lambda x: x not in ['mean', 'loc'], kwargs)):
            raise ValueError('Unknown keyword argument')
        if args:
            kwargs.setdefault('value', args[0])
        value = _get_aliases(kwargs, ['value', 'mean', 'loc'], 0)
        value = to_tensor(value, dtype=get_default_dtype())
        self.param = self.Parameters(self, value=value)

    def __call__(self, batch=tuple(), **backend):
        batch = make_tuple(batch or [])
        value = self.param.value.to(**backend)
        return (value.new_empty([batch + self.param.shape]).copy_(value)
                if batch else value.clone())


class Uniform(Sampler):
    """
    A uniform distribution

    !!! example "Signatures"
        ```python
        Uniform()                   # default: (a=0, b=1)
        Uniform(b)                  # default min: (a=0)
        Uniform(a, b)               # positional variant
        Uniform(a=0, b=1)           # keyword variant

        # Aliases
        Uniform(*, min=VALUE)       # Alias for a
        Uniform(*, max=VALUE)       # Alias for b
        Uniform(*, mean=VALUE)      # Specify mean (default width is 1)
        Uniform(*, fwhm=VALUE)      # Specify width. Must also set mean or loc
        Uniform(*, std=VALUE)       # Specify standard deviation
        Uniform(*, var=VALUE)       # Specify variance
        Uniform(*, loc=VALUE)       # Alias for mean
        Uniform(*, scale=VALUE)     # Alias for std
        ```
    """

    @classmethod
    def unserialize(cls, obj):
        if isinstance(obj, dict):
            return cls(**obj)
        elif isinstance(obj, (list, tuple)):
            return cls(*obj)
        else:
            return cls(obj)

    class Parameters(Sampler.Parameters):

        @property
        def shape(self) -> torch.Size:
            return torch.broadcast_shapes(self.a.shape, self.b.shape)

        def min(self, *args, **kwargs):
            return self.a

        def max(self, *args, **kwargs):
            return self.b

        def mean(self, *args, **kwargs):
            return (self.a + self.b) / 2

        def fwhm(self, *args, **kwargs):
            """Full-width at half-maximum"""
            return (self.b - self.a).abs()

        def var(self, *args, **kwargs):
            return self.fwhm().square() / 12

        def std(self, *args, **kwargs):
            return self.fwhm() / (12 ** 0.5)

        scale = std

    def __init__(self, *args, **kwargs):
        keys = list(kwargs.keys())
        keys_min = ('min', 'a')
        keys_max = ('max', 'b')
        keys_loc = ('mean', 'loc')
        keys_scl = ('fwhm', 'std', 'var', 'scale')
        keys_all = keys_min + keys_max + keys_loc + keys_scl
        nb_min = (len(args) > 1) + sum(map(lambda x: keys.count(x), keys_min))
        nb_max = (len(args) > 0) + sum(map(lambda x: keys.count(x), keys_max))
        nb_loc = sum(map(lambda x: keys.count(x), keys_loc))
        nb_scl = sum(map(lambda x: keys.count(x), keys_scl))
        if any(key not in keys_all for key in kwargs):
            raise ValueError('Unknown arguments')
        if (nb_min + nb_max) and (nb_loc + nb_scl):
            raise ValueError('Bound and nonbound arguments')
        if (nb_min > 1) or (nb_max > 1) or (nb_loc > 1) or (nb_scl > 1):
            raise ValueError('Incompatible arguments')
        if nb_scl and not nb_loc:
            raise ValueError('Location argument without scale argument')

        if nb_loc:
            loc = _get_aliases(kwargs, ['mean', 'loc'])
            if 'fwhm' in kwargs:
                fwhm = kwargs['fwhm']
            elif 'var' in kwargs:
                fwhm = (12 * kwargs['var']) ** 0.5
            elif 'std' in kwargs or 'scale' in kwargs:
                std = _get_aliases(kwargs, ['std', 'scale'])
                fwhm = (12 ** 0.5) * std
            else:
                fwhm = 1
            loc, fwhm = to_tensor(loc, fwhm, dtype=float)
            a = loc - fwhm / 2
            b = loc + fwhm / 2
        else:
            if len(args) == 2:
                a, b = args
            elif len(args) == 1:
                if 'max' in kwargs or 'b' in kwargs:
                    a = args[0]
                    b = _get_aliases(kwargs, ['max', 'b'], 1)
                else:
                    b = args[0]
                    a = _get_aliases(kwargs, ['min', 'a'], 0)
            else:
                b = _get_aliases(kwargs, ['max', 'b'], 1)
                a = _get_aliases(kwargs, ['min', 'a'], 0)
            a, b = to_tensor(a, b, dtype=float)
        self.param = self.Parameters(self, a=a, b=b)

    def kernel(self, x):
        return (self.param.a < x & x <= self.param.b).to(x)

    def pdf(self, x):
        return self.kernel(x) / (self.b - self.a)

    def __call__(self, batch=tuple(), **backend):
        batch = make_tuple(batch or [])
        a = to_tensor(self.param.a, **backend)
        b = to_tensor(self.param.b, **backend)
        return distributions.Uniform(a, b).sample(batch)


class RandInt(Sampler):
    """
    A discrete uniform distribution, with both bounds included.

    !!! example "Signatures"
        ```python
        RandInt()                   # default: (a=0, b=1)
        RandInt(b)                  # default min: (a=0)
        RandInt(a, b)               # positional variant
        RandInt(a=0, b=1)           # keyword variant

        # Aliases
        RandInt(*, min=VALUE)       # Alias for a
        RandInt(*, max=VALUE)       # Alias for b
        RandInt(*, mean=VALUE)      # Specify mean (default width is 1)
        RandInt(*, fwhm=VALUE)      # Specify width. Must also set mean or loc
        RandInt(*, std=VALUE)       # Specify standard deviation
        RandInt(*, var=VALUE)       # Specify variance
        RandInt(*, loc=VALUE)       # Alias for mean
        RandInt(*, scale=VALUE)     # Alias for std
        ```
    """

    class Parameters(Sampler.Parameters):

        @property
        def shape(self) -> torch.Size:
            return torch.broadcast_shapes(self.a.shape, self.b.shape)

        def min(self, *args, **kwargs):
            return self.a

        def max(self, *args, **kwargs):
            return self.b

        def mean(self, *args, **kwargs):
            return (self.a + self.b) / 2

        def fwhm(self, *args, **kwargs):
            """Full-width at half-maximum"""
            return (self.b - self.a).abs() + 1

        def var(self, *args, **kwargs):
            return (self.fwhm().square() - 1) / 12

        def std(self, *args, **kwargs):
            return self.var().sqrt()

        scale = std

    def __init__(self, *args, **kwargs):
        keys = list(kwargs.keys())
        keys_min = ('min', 'a')
        keys_max = ('max', 'b')
        keys_loc = ('mean', 'loc')
        keys_scl = ('fwhm', 'std', 'var', 'scale')
        keys_all = keys_min + keys_max + keys_loc + keys_scl
        nb_min = (len(args) > 1) + sum(map(lambda x: keys.count(x), keys_min))
        nb_max = (len(args) > 0) + sum(map(lambda x: keys.count(x), keys_max))
        nb_loc = sum(map(lambda x: keys.count(x), keys_loc))
        nb_scl = sum(map(lambda x: keys.count(x), keys_scl))
        if any(key not in keys_all for key in kwargs):
            raise ValueError('Unknown arguments')
        if (nb_min + nb_max) and (nb_loc + nb_scl):
            raise ValueError('Bound and nonbound arguments')
        if (nb_min > 1) or (nb_max > 1) or (nb_loc > 1) or (nb_scl > 1):
            raise ValueError('Incompatible arguments')
        if nb_scl and not nb_loc:
            raise ValueError('Location argument without scale argument')

        if nb_loc:
            loc = _get_aliases(kwargs, ['mean', 'loc'])
            if 'fwhm' in kwargs:
                fwhm = kwargs['fwhm']
            elif 'var' in kwargs:
                fwhm = (12 * kwargs['var'] + 1) ** 0.5 - 1
            elif 'std' in kwargs or 'scale' in kwargs:
                std = _get_aliases(kwargs, ['std', 'scale'])
                var = std * std
                fwhm = (12 * var + 1) ** 0.5 - 1
            else:
                fwhm = 1
            loc, fwhm = to_tensor(loc, fwhm, dtype=float)
            a = loc - fwhm / 2
            b = loc + fwhm / 2
            a, b = a.round().long(), b.round().long()
        else:
            if len(args) == 2:
                a, b = args
            elif len(args) == 1:
                if 'max' in kwargs or 'b' in kwargs:
                    a = args[0]
                    b = _get_aliases(kwargs, ['max', 'b'])
                else:
                    b = args[0]
                    a = _get_aliases(kwargs, ['min', 'a'], 0)
            else:
                b = _get_aliases(kwargs, ['max', 'b'])
                a = _get_aliases(kwargs, ['min', 'a'], 0)
            a, b = to_tensor(a, b, dtype=int)
        self.param = self.Parameters(self, a=a, b=b)

    def __call__(self, batch=tuple(), **backend):
        batch = make_tuple(batch or [])
        dtype = backend.pop('dtype', None)
        if not dtype:
            dtype = self.param.a.dtype
        if dtype.is_floating_point:
            dtype = torch.long
        if not self.param.a.dtype.is_floating_point:
            backend['dtype'] = torch.get_default_dtype()
        a = to_tensor(self.param.a, **backend)
        b = to_tensor(self.param.b, **backend)
        x = distributions.Uniform(a-0.5, b+0.5).sample(batch)
        return x.round().to(dtype)


class Normal(Sampler):
    """
    A Normal/Gaussian distribution

    !!! example "Signatures"
        ```python
        Normal()                    # default: (mu=0, sigma=1)
        Normal(sigma)               # default mean: (mu=0)
        Normal(mu, sigma)           # positional variant
        Normal(mu=0, sigma=1)       # keyword variant

        # Aliases
        Normal(*, mean=VALUE)       # Alias for mu
        Normal(*, std=VALUE)        # Alias for sigma
        Normal(*, var=VALUE)        # Specify variance (sigma**2)
        Normal(*, fwhm=VALUE)       # Specify width (~ 2.355*sigma)
        Normal(*, loc=VALUE)        # Alias for mu
        Normal(*, scale=VALUE)      # Alias for sigma
        ```
    """

    class Parameters(Sampler.Parameters):

        @property
        def shape(self) -> torch.Size:
            return torch.broadcast_shapes(self.mu.shape, self.sigma.shape)

        def min(self, *args, **kwargs):
            return self.mu.new_full(self.shape, -float('inf'))

        def max(self, *args, **kwargs):
            return self.mu.new_full(self.shape, float('inf'))

        def mean(self, *args, **kwargs):
            return self.mu

        def std(self, *args, **kwargs):
            return self.sigma

        def var(self, *args, **kwargs):
            return self.sigma.square()

        def fwhm(self, *args, **kwargs):
            """Full-width at half-maximum"""
            return self.sigma * (2 * math.sqrt(2 * math.log(2)))

        loc = mean
        scale = std

    def __init__(self, *args, **kwargs):
        keys = list(kwargs.keys())
        keys_loc = ('mu', 'mean', 'loc')
        keys_scl = ('sigma', 'fwhm', 'std', 'var', 'scale')
        nb_loc = (len(args) > 1) + sum(map(lambda x: keys.count(x), keys_loc))
        nb_scl = (len(args) > 0) + sum(map(lambda x: keys.count(x), keys_scl))
        if any(key not in keys_loc + keys_scl for key in kwargs):
            raise ValueError('Unknown arguments')
        if (nb_loc > 1) or (nb_scl > 1):
            raise ValueError('Incompatible arguments')

        mu, sigma = 0, 1
        if len(args) == 2:
            mu = args[0]
            sigma = args[1]
        elif len(args) == 1:
            sigma = args[0]
        mu = _get_aliases(kwargs, ['mu', 'mean', 'loc'], mu)
        sigma = _get_aliases(kwargs, ['sigma', 'std', 'scale'], sigma)
        if 'var' in kwargs:
            sigma = kwargs['var'] ** 0.5
        elif 'fwhm' in kwargs:
            sigma = kwargs['fwhm'] / (2 * math.sqrt(2 * math.log(2)))

        mu, sigma = to_tensor(mu, sigma, dtype=float)
        self.param = self.Parameters(self, mu=mu, sigma=sigma)

    def __call__(self, batch=tuple(), **backend):
        batch = make_tuple(batch or [])
        mu = to_tensor(self.param.mu, **backend)
        sigma = to_tensor(self.param.sigma, **backend)
        return distributions.Normal(mu, sigma).sample(batch)


class LogNormal(Sampler):
    """
    A log-Normal distribution

    !!! example "Signatures"
        ```python
        LogNormal()                    # default: (mu=0, sigma=1)
        LogNormal(sigma)               # default mean: (mu=0)
        LogNormal(mu, sigma)           # positional variant
        LogNormal(mu=0, sigma=1)       # keyword variant

        # Aliases
        LogNormal(*, logmean=VALUE)    # Alias for mu (mean of log)
        LogNormal(*, logstd=VALUE)     # Alias for sigma (std of log)
        LogNormal(*, logvar=VALUE)     # Specify variance of log (sigma**2)
        LogNormal(*, mean=VALUE)       # Specify the mean
        LogNormal(*, var=VALUE)        # Specify variance
        LogNormal(*, std=VALUE)        # Specify the standard deviation
        LogNormal(*, loc=VALUE)        # Alias for mu
        LogNormal(*, scale=VALUE)      # Alias for sigma
        ```
    """

    class Parameters(Sampler.Parameters):

        @property
        def shape(self) -> torch.Size:
            return torch.broadcast_shapes(self.mu.shape, self.sigma.shape)

        def min(self, *args, **kwargs):
            return self.mu.new_zeros(self.shape, -float('inf'))

        def max(self, *args, **kwargs):
            return self.mu.new_full(self.shape, float('inf'))

        def logmean(self, *args, **kwargs):
            return self.mu

        def logstd(self, *args, **kwargs):
            return self.sigma

        def logvar(self, *args, **kwargs):
            return self.sigma.square()

        def mean(self, *args, **kwargs):
            return (self.mu + 0.5 * self.sigma.square()).exp()

        def var(self, *args, **kwargs):
            return self.mean().square() * (self.sigma.square().exp() - 1)

        loc = logmean
        scale = logstd

    def __init__(self, *args, **kwargs):
        keys = list(kwargs.keys())
        keys_loc = ('mu', 'mean', 'loc', 'logmean')
        keys_scl = ('sigma', 'fwhm', 'std', 'var', 'scale', 'logstd', 'logvar')
        nb_loc = (len(args) > 1) + sum(map(lambda x: keys.count(x), keys_loc))
        nb_scl = (len(args) > 0) + sum(map(lambda x: keys.count(x), keys_scl))
        if any(key not in keys_loc + keys_scl for key in kwargs):
            raise ValueError('Unknown arguments')
        if (nb_loc > 1) or (nb_scl > 1):
            raise ValueError('Incompatible arguments')

        # log-parameters
        mu, sigma = 0, 1
        if len(args) == 2:
            mu = args[0]
            sigma = args[1]
        elif len(args) == 1:
            sigma = args[0]
        mu = _get_aliases(kwargs, ['mu', 'logmean', 'loc'], mu)
        sigma = _get_aliases(kwargs, ['sigma', 'logstd', 'scale'], sigma)
        if 'logvar' in kwargs:
            sigma = kwargs['logvar'] ** 0.5
        if 'std' in kwargs:
            kwargs['var'] = kwargs.pop('std') ** 2

        # exp-parameters
        if 'mean' in kwargs and 'var' in kwargs:
            mean, var = to_tensor(kwargs['mean'], kwargs['var'], dtype=float)
            sigma2 = (1 + var / mean.square()).log().clamp_min(0)
            mu = mean.log() - sigma2 / 2
            sigma = sigma2.sqrt()
        elif 'mean' in kwargs:
            mean, sigma = to_tensor(kwargs['mean'], sigma, dtype=float)
            mu = mean.log() - 0.5 * self.sigma.square()
        elif 'var' in kwargs:
            mu, var = to_tensor(mu, kwargs['var'])
            var = var / mu.exp().square()
            delta = (1 - 4 * var).clamp_min_(0)
            sigma = (1 + delta.sqrt()) / 2

        mu, sigma = to_tensor(mu, sigma, dtype=float)
        self.param = self.Parameters(self, mu=mu, sigma=sigma)

    def __call__(self, batch=tuple(), **backend):
        batch = make_tuple(batch or [])
        mu = to_tensor(self.param.mu, **backend)
        sigma = to_tensor(self.param.sigma, **backend)
        return distributions.LogNormal(mu, sigma).sample(batch)


class UniformSphere(Sampler):
    """
    Uniform distribution on the (d-1)-sphere
    """
    def __init__(self, ndim=3, **backend):
        """
        Parameters
        ----------
        ndim : int
            Number of dimensions of the embedding space.
            If `ndim=3`, generate sample that lie on the 2-sphere.
        """
        super().__init__()
        backend.setdefault('dtype', torch.get_default_dtype())
        backend.setdefault('device', 'cpu')
        self.param = self.Parameters(ndim=ndim, **backend)

    class Parameters(Sampler.Parameters):

        @property
        def shape(self):
            return torch.Size([self.ndim])

    def __call__(self, batch=tuple(), **backend):
        backend.setdefault('dtype', self.param.dtype)
        backend.setdefault('device', self.param.device)
        x = torch.randn(tuple(batch) + (self.param.ndim,), **backend)
        mask = x.norm(2, dim=1) < 1e-3
        while mask.any():
            x = torch.where(
                mask,
                torch.randn(tuple(batch) + (self.param.ndim,), **backend),
                x,
            )
            mask = x.norm(2, dim=1) < 1e-3
        x /= x.norm(2, dim=1, keepdim=True)
        return x


class AngularCentralGaussian(Sampler):
    """
    Angular Central Gaussian (ACG) distribution.

    !!! note "Parameters"
            - `A`: `(*, 3, 3)` precision

    !!! example "Signatures"
        ```python
        AngularCentralGaussian()       # default (A=I)
        AngularCentralGaussian(A)      # positional variant
        AngularCentralGaussian(A=0)    # keyword variant
        # aliases
        AngularCentralGaussian(*, lam=VALUE)        # alias for A
        AngularCentralGaussian(*, sigma=VALUE)      # alias for inv(A)
        AngularCentralGaussian(*, U=VALUE)          # upper-triangular matrix
        AngularCentralGaussian(*, triu=VALUE)       # alias for U
        AngularCentralGaussian(*, L=VALUE)          # lower-triangular matrix
        AngularCentralGaussian(*, tril=VALUE)       # alias for L
        AngularCentralGaussian(*, R=VALUE, Z=1)     # SVD decomposition of A
        AngularCentralGaussian(*, rot=VALUE)        # alias for R
        AngularCentralGaussian(*, axes=VALUE)       # alias for R
        ```
        Note that the rows of `R` are the principle axes, not its columns,
        i.e., `R = [axis1, axis2, axis3]`.
    """

    class Parameters(Sampler.Parameters):

        @property
        def shape(self) -> torch.Size:
            return self.A.shape[:-1]

    def __init__(self, *args, **kwargs):
        super().__init__()

        keys = list(kwargs.keys())
        keys_scl = (
            'A', 'lam', 'sigma', 'U', 'triu', 'L', 'tril', 'R', 'rot', 'axes'
        )
        nb_scl = (len(args) > 1) + sum(map(lambda x: keys.count(x), keys_scl))
        if any(key not in keys_scl for key in kwargs):
            raise ValueError('Unknown arguments')
        if nb_scl > 1:
            raise ValueError('Incompatible arguments')
        if len(args) > 1:
            raise ValueError('Too many arguments')

        # log-parameters
        A = None
        R, Z = ((1, 0, 0), (0, 1, 0), (0, 0, 1)), (1, 1, 1)
        if len(args) == 1:
            A = args[0]
        A = _get_aliases(kwargs, ['A', 'lam', 'prec'], A)
        R = _get_aliases(kwargs, ['R', 'rot', 'axes'], R)
        Z = _get_aliases(kwargs, ['Z'], Z)
        if 'sigma' in kwargs:
            A = torch.linalg.inv(kwargs['sigma'])

        R = torch.as_tensor(R)
        Z = torch.as_tensor(Z)
        if not Z.shape or Z.shape[-1] == 1:
            if not Z.shape:
                Z = Z[None]
            Z = Z.repeat_interleave(3, -1)

        if A is None:
            if Z.shape[-1] != 3:
                raise ValueError('Last dimension of Z must be 3')
            if R.shape[-2:] != (3, 3):
                raise ValueError('Last two dimensions of R must be 3')
            A = (R * Z.unsqueeze(-2)).matmul(R.transpose(-1, -2))
        elif A.shape[-2:] != (3, 3):
            raise ValueError('Last two dimensions of A must be 3')
        else:
            A = (A + A.transpose(-1, -2)) / 2

        # all `A + c*I`, whatever the value of `c`, yield the same .
        # distribution. We therefore subtract the smallest eigenvalue of A.
        # We also add a little epsilon so that it's pos-def
        A = A.clone()
        A.diagonal(0, -1, -2).sub_(torch.linalg.eigvalsh(A)[..., :1] - 1e-3)

        self.param = self.Parameters(self, A=A)

    def logkernel(self, x):
        """
        Return the unnormalized log-PDF
        """
        ndim = x.shape[-1]
        # ensure data lie on the sphere
        x = x / x.norm(2, dim=-1, keepdim=True)
        # compute main term
        logp = self.param.A.matmul(x.unsqueeze(-1))
        logp = x.unsqueeze(-2).matmul(logp).squeeze(-1).squeeze(-1)
        logp = -0.5 * ndim * logp.log()
        return logp

    def kernel(self, x):
        """
        Return the unnormalized PDF
        """
        return self.logkernel(x).exp()

    def __call__(self, batch=tuple(), **backend):
        """
        !!! warning
            We are using a Metropolis-Hastings samplers and samples
            (within call, or across successive calls) are correlated.
        """
        batch = make_tuple(batch)
        A = to_tensor(self.param.A, **backend)
        sample = torch.distributions.MultivariateNormal(
            loc=torch.zeros_like(A[..., 0]),
            precision_matrix=A,
        ).sample(batch)
        sample = sample / sample.norm(2, dim=-1, keepdim=True)
        return sample


class FisherBingham(Sampler):
    r"""
    Fisher-Bingham distribution.

    !!! "PDF"
        p(x) = Z(A) exp(\kappa * x'\mu + x'Ax)

    !!! note "Parameters"
        - `mu`:     `(*, 3)`    principal axis with unit norm
        - `kappa`:  `(*)`       concentration
        - `A`:      `(*, 3, 3)` precision

    !!! example "Signatures"
        ```python
        FisherBingham(mu=[1, 0, 0], kappa=1, A=0)
        ```

    !!! tip "Special cases"
        ```python
        FisherBingham(mu, kappa, 0) === VonMisesFisher(mu, kappa)
        FisherBingham(_, 0, A)      === Bingham(A)
        A @ mu == 0                 === Kent(mu, kappa, A)
        ```
    """

    class Parameters(Sampler.Parameters):

        @property
        def shape(self) -> torch.Size:
            return torch.broadcast_shapes(
                self.mu.shape, self.kappa[..., None].shape, self.A.shape[:-1])

        def mean(self, *args, **kwargs):
            return self.mu

        loc = mean

    def __init__(self, mu=(1, 0, 0), kappa=1, A=0):
        super().__init__()

        mu, kappa, A = to_tensor(mu, kappa, A, dtype=float)
        if not mu.shape or mu.shape[-1] == 1:
            if not mu.shape:
                mu = mu[None]
            mu = mu.repeat_interleave(3, -1)
        if not A.shape or A.shape[-1] == 1:
            while A.ndim < 2:
                A = A[None]
            if A.shape[-1] == 1:
                A = A.repeat_interleave(3, -1)
            if A.shape[-2] == 1:
                A = A.repeat_interleave(3, -2)
        if mu.shape[-1] != 3:
            raise ValueError('Last dimension of mu must be 3')
        if A.shape[-2:] != (3, 3):
            raise ValueError('Last two dimensions of lam must be 3')

        # Ensure mean is unit norm
        mu = mu / mu.norm(2, dim=-1, keepdim=True)

        # all `A + c*I`, whatever the value of `c`, yield the same .
        # distribution. We therefore subtract the smallest eigenvalue of -A.
        A = A.neg()
        A.diagonal(0, -1, -2).sub_(torch.linalg.eigvalsh(A)[..., :1])
        A.neg_()

        self.param = self.Parameters(self, mu=mu, kappa=kappa, A=A)

    def logkernel(self, x):
        """
        Return the unnormalized log-PDF
        """
        # ensure data lie on the sphere
        x = x / x.norm(2, dim=-1, keepdim=True)
        # compute main term
        A = self.param.A
        mu = self.param.mu.unsqueeze(-1)
        kappa = self.param.kappa.unsqueeze(-1).unsqueeze(-1)
        logp = A.matmul(x.unsqueeze(-1))
        logp = x.unsqueeze(-2).matmul(logp)
        logp += x.unsqueeze(-2).matmul(mu) * kappa
        logp = logp.squeeze(-1).squeeze(-1)
        return logp

    def kernel(self, x):
        """
        Return the unnormalized PDF
        """
        return self.logkernel(x).exp()

    def __call__(self, batch=tuple(), **backend):
        # Rejection method with a Bingham envelope
        # https://arxiv.org/pdf/1310.8110
        batch = make_tuple(batch)
        kappa = to_tensor(self.param.kappa, **backend)
        mu = to_tensor(self.param.mu, **backend)
        A = to_tensor(self.param.A, **backend).neg()

        # Bingham envelope of the FisherBingham distribution
        A1 = mu[..., :, None] * mu[..., :, None]
        A1 = A + 0.5 * kappa[..., None, None] * A1

        # ACG envelope of the Bingham envelope
        logM, b = _bacg_logbound(A1)
        Omega = 0.5 * A / b[..., None, None]
        Omega.diagonal(0, -1, -2).add_(1)
        acg = AngularCentralGaussian(Omega)
        return rejection_sampling(
            pdf=self.logkernel,
            pdf_sampler=acg.logkernel,
            log=True,
            sampler=acg,
            shape=batch,
            sup=0.5 * kappa + logM,
        )


class Bingham(Sampler):
    """
    Bingham distribution.

    !!! "PDF"
        p(x) = Z(A) exp(x'Ax)

    !!! note "Parameters"
            - `A`: `(*, 3, 3)` precision matrix

    !!! example "Signatures"
        ```python
        Bingham()       # default (A=0)
        Bingham(A)      # positional variant
        Bingham(A=0)    # keyword variant
        # aliases
        Bingham(*, lam=VALUE)           # alias for A
        Bingham(*, sigma=VALUE)         # alias for inv(A)
        Bingham(*, U=VALUE)             # upper-triangular matrix: A = U.T@U
        Bingham(*, triu=VALUE)          # alias for U
        Bingham(*, L=VALUE)             # lower-triangular matrix: A = L@L.T
        Bingham(*, tril=VALUE)          # alias for L
        Bingham(*, R=VALUE, Z=1)        # SVD decomposition of A: A = R@Z@R.T
        Bingham(*, rot=VALUE)           # alias for R
        Bingham(*, axes=VALUE)          # alias for R
        ```
        Note that the rows of `R` are the principle axes, not its columns,
        i.e., `R = [axis1, axis2, axis3]`.
    """

    class Parameters(Sampler.Parameters):

        @property
        def shape(self) -> torch.Size:
            return self.A.shape[:-1]

    def __init__(self, *args, **kwargs):
        super().__init__()

        keys = list(kwargs.keys())
        keys_scl = (
            'A', 'lam', 'sigma', 'U', 'triu', 'L', 'tril', 'R', 'rot', 'axes'
        )
        nb_scl = (len(args) > 1) + sum(map(lambda x: keys.count(x), keys_scl))
        if any(key not in keys_scl for key in kwargs):
            raise ValueError('Unknown arguments')
        if nb_scl > 1:
            raise ValueError('Incompatible arguments')
        if len(args) > 1:
            raise ValueError('Too many arguments')

        # log-parameters
        A = None
        R, Z = ((1, 0, 0), (0, 1, 0), (0, 0, 1)), (1, 1, 1)
        if len(args) == 1:
            A = args[0]
        A = _get_aliases(kwargs, ['A', 'lam', 'prec'], A)
        R = _get_aliases(kwargs, ['R', 'rot', 'axes'], R)
        Z = _get_aliases(kwargs, ['Z'], Z)
        if 'sigma' in kwargs:
            A = torch.linalg.inv(kwargs['sigma'])

        R = torch.as_tensor(R)
        Z = torch.as_tensor(Z)
        if not Z.shape or Z.shape[-1] == 1:
            if not Z.shape:
                Z = Z[None]
            Z = Z.repeat_interleave(3, -1)

        if A is None:
            if Z.shape[-1] != 3:
                raise ValueError('Last dimension of Z must be 3')
            if R.shape[-2:] != (3, 3):
                raise ValueError('Last two dimensions of R must be 3')
            A = (R * Z.unsqueeze(-2)).matmul(R.transpose(-1, -2))
        elif A.shape[-2:] != (3, 3):
            raise ValueError('Last two dimensions of A must be 3')
        else:
            A = (A + A.transpose(-1, -2)) / 2

        # all `A + c*I`, whatever the value of `c`, yield the same .
        # distribution. We therefore subtract the smallest eigenvalue of A.
        A = A.neg()
        A.diagonal(0, -1, -2).sub_(torch.linalg.eigvalsh(A)[..., :1])
        A.neg_()

        self.param = self.Parameters(self, A=A)

    def logkernel(self, x):
        """
        Return the unnormalized log-PDF
        """
        # ensure data lie on the sphere
        x = x / x.norm(2, dim=-1, keepdim=True)
        # compute main term
        logp = self.param.A.matmul(x.unsqueeze(-1))
        logp = x.unsqueeze(-2).matmul(logp).squeeze(-1).squeeze(-1)
        return logp

    def kernel(self, x):
        """
        Return the unnormalized PDF
        """
        return self.logkernel(x).exp()

    def __call__(self, batch=tuple(), **backend):
        # Rejection method with a ACG envelope
        # https://arxiv.org/pdf/1310.8110
        batch = make_tuple(batch)
        A = to_tensor(self.param.A, **backend).neg()
        logM, b = _bacg_logbound(A)
        Omega = 2 * A / b[..., None, None]
        Omega.diagonal(0, -1, -2).add_(1)
        acg = AngularCentralGaussian(Omega)
        return rejection_sampling(
            pdf=self.logkernel,
            pdf_sampler=acg.logkernel,
            log=True,
            sampler=acg,
            shape=batch,
            sup=logM,
        )


def _bacg_logbound(A, tol=1e-6, max_iter=1024):
    # Compute the rejection bound for for a Bingham rejection
    # sampler with an Angular Central Gaussian envelope (BACG).
    #
    # We wish to find M such as f(x) < M*g(x) \forall x,
    # where f is Bingham and g is ACG. The acceptance probability is
    # then f(x) / M*g(x).

    # This bound is found by minimizing eq (3.5) from Kent et al. (2013)
    # Its log is convex and has a unique minimum.
    #
    # Reference:
    # Kent, Ganeiber, Marda. "A new method to simulate the Bingham and
    # related distributions in directional data analysis with applications."
    # https://arxiv.org/abs/1310.8110
    # https://eprints.whiterose.ac.uk/123206/7/simbingham8.pdf

    lam = torch.linalg.eigvalsh(A)
    q = lam.shape[-1]

    def func(b):
        b = b.unsqueeze(-1)
        f = 0.5 * b - 0.5 * (b + 2 * lam).log().sum(-1, keepdim=True)
        f = f.squeeze(-1)
        return f

    def grad(b):
        b = b.unsqueeze(-1)
        g = 0.5 - 0.5 * (b + 2 * lam).reciprocal().sum(-1, keepdim=True)
        g = g.squeeze(-1)
        return g

    def hess(b):
        # # This is the true Hessian
        # b = b.unsqueeze(-1)
        # h = 0.5 * (b + 2 * lam).square().reciprocal().sum(-1, keepdim=True)
        # h = h.squeeze(-1)
        # # But we instead use a fixed upper bound that ensures we never
        # # overshoot
        h = 0.5 * q
        return h

    b = torch.ones_like(lam[..., 0])
    f = func(b)
    for _ in range(max_iter):
        g, h = grad(b), hess(b)
        delta = g / h
        f0, armijo = f, 1
        b.sub_(delta, alpha=armijo)
        for _ in range(12):
            f = func(b)
            mask = f < f0
            if mask.all():
                break
            armijo /= 2
            b.add_(delta * (~mask), alpha=armijo)
        if not mask.all():
            b.add_(delta * (~mask), alpha=armijo)
        if (f-f0).abs().max() < tol:
            break

    f = func(b)
    f += 0.5 * q * (math.log(q) - 1)
    return f, b


class Watson(Sampler):
    r"""
    Watson distribution.

    !!! "PDF"
        p(x) = Z(A) exp(\kappa * (x'\mu)^2)

    !!! note "Parameters"
        - `mu`:     `(*, 3)`    principal axis with unit norm
        - `kappa`:  `(*)`       concentration

    !!! example "Signatures"
        ```python
        Watson(mu=[1, 0, 0], kappa=1)
        ```
    """

    class Parameters(Sampler.Parameters):

        @property
        def shape(self) -> torch.Size:
            return torch.broadcast_shapes(
                self.mu.shape, self.kappa[..., None].shape)

    def __init__(self, mu=(1, 0, 0), kappa=1):
        super().__init__()

        mu, kappa = to_tensor(mu, kappa, dtype=float)
        if not mu.shape or mu.shape[-1] == 1:
            if not mu.shape:
                mu = mu[None]
            mu = mu.repeat_interleave(3, -1)
        if mu.shape[-1] != 3:
            raise ValueError('Last dimension of mu must be 3')
        mu = mu / mu.norm(2, dim=-1, keepdim=True)
        self.param = self.Parameters(self, mu=mu, kappa=kappa)

    def logkernel(self, x):
        """
        Return the unnormalized log-PDF
        """
        # ensure data lie on the sphere
        x = x / x.norm(2, dim=-1, keepdim=True)
        # compute main term
        mu = self.param.mu.unsqueeze(-1)
        logp = x.unsqueeze(-2).matmul(mu).squeeze(-1).squeeze(-1)
        logp = logp.square().mul(self.param.kappa)
        return logp

    def kernel(self, x):
        """
        Return the unnormalized PDF
        """
        return self.logkernel(x).exp()

    def __call__(self, batch=tuple(), **backend):
        mu = to_tensor(self.param.mu, **backend)
        kappa = to_tensor(self.param.kappa, **backend)
        A = mu[..., None, :] * mu[..., :, None]
        A = A * kappa[..., None, None]
        return Bingham(A)(batch)


class VonMisesFisher(Sampler):
    r"""
    von Mises-Fisher distribution.

    !!! "PDF"
        p(x) = Z(A) exp(\kappa * x'\mu)

    !!! note "Parameters"
        - `mu`:     `(*, 3)`    principal axis with unit norm
        - `kappa`:  `(*)`       concentration

    !!! example "Signatures"
        ```python
        VonMisesFisher(mu=[1, 0, 0], kappa=1)
        ```
    """

    class Parameters(Sampler.Parameters):

        @property
        def shape(self) -> torch.Size:
            return torch.broadcast_shapes(
                self.mu.shape, self.kappa[..., None].shape)

        def mean(self, *args, **kwargs):
            return self.mu

        loc = mean

    def __init__(self, mu=(1, 0, 0), kappa=1):
        super().__init__()

        mu, kappa = to_tensor(mu, kappa, dtype=float)
        if not mu.shape or mu.shape[-1] == 1:
            if not mu.shape:
                mu = mu[None]
            mu = mu.repeat_interleave(3, -1)
        if mu.shape[-1] != 3:
            raise ValueError('Last dimension of mu must be 3')
        mu = mu / mu.norm(2, dim=-1, keepdim=True)
        self.param = self.Parameters(self, mu=mu, kappa=kappa)

    def logkernel(self, x):
        """
        Return the unnormalized log-PDF
        """
        # ensure data lie on the sphere
        x = x / x.norm(2, dim=-1, keepdim=True)
        # compute main term
        mu = self.param.mu.unsqueeze(-1)
        kappa = self.param.kappa.unsqueeze(-1).unsqueeze(-1)
        logp = x.unsqueeze(-2).matmul(mu) * kappa
        logp = logp.squeeze(-1).squeeze(-1)
        return logp

    def kernel(self, x):
        """
        Return the unnormalized PDF
        """
        return self.logkernel(x).exp()

    def __call__(self, batch=tuple(), **backend):
        # Rejection method with a Bingham envelope
        # https://arxiv.org/pdf/1310.8110
        batch = make_tuple(batch)
        kappa = to_tensor(self.param.kappa, **backend)
        mu = to_tensor(self.param.mu, **backend)

        # Bingham envelope of the VonMisesFisher distribution
        A = mu[..., :, None] * mu[..., :, None]
        A.neg_()
        A.diagonal(0, -1, -2).add_(1)

        # ACG envelope of the Bingham envelope
        logM, b = _bacg_logbound(A)
        Omega = 2 * A / b[..., None, None]
        Omega.diagonal(0, -1, -2).add_(1)
        acg = AngularCentralGaussian(Omega)
        return rejection_sampling(
            pdf=self.logkernel,
            pdf_sampler=acg.logkernel,
            log=True,
            sampler=acg,
            shape=batch,
            sup=0.5 * kappa + logM,
        )


def icdf(p, cdf, mn, mx, tol=1e-6, steps=8):
    """
    Compute the inverse of a cumulative distribution functions (CDF) via
    a binary tree search.

    Parameters
    ----------
    p : (*batch) tensor
        Input cummulant
    cdf : callable[tensor] -> tensor
        A function that evaluates the CDF
    mn : float or (*batch) tensor
        The lower bound of the value range to search
    mx : float or (*batch) tensor
        The upper bound of the value range to search
    tol : float
        Target precision, computed over the cumulant
    steps : int
        Number of values at which to evaluate the CDF, per iteration

    Returns
    -------
    x : (*batch) tensor
        Output value, such that `cdf(x) == p`.
    """
    mn = torch.as_tensor(mn, dtype=p.dtype, device=p.device)
    mx = torch.as_tensor(mx, dtype=p.dtype, device=p.device)
    p, mn, mx = torch.broadcast_tensors(p, mn, mx)

    x = torch.linspace(0, 1, steps, dtype=p.dtype, device=p.device)
    x = x * (mx - mn)[..., None] + mn[..., None]
    q = cdf(x.movedim(-1, 0)).movedim(0, -1)
    i = torch.searchsorted(q, p.unsqueeze(-1), side='right')
    batch_shape = torch.broadcast_shapes(
        i.shape[:-1], x.shape[:-1], q.shape[:-1])
    q = q.expand([*batch_shape, q.shape[-1]])
    x = x.expand([*batch_shape, x.shape[-1]])
    oob = (i == steps).squeeze(-1)
    pmn = q.take_along_dim(i-1, -1).squeeze(-1)
    mn = x.take_along_dim(i-1, -1).squeeze(-1)
    i.clamp_max_(steps-1)
    pmx = q.take_along_dim(i, -1).squeeze(-1)
    mx = x.take_along_dim(i, -1).squeeze(-1)
    if not oob.any() and (((pmn+pmx)/2 - p).abs() < tol).all():
        return (mn + mx) / 2
    # golden section search on the right if solution not in current bounds
    gold = (1 + 5**0.5) / 2
    mx[oob] = x[oob, -1] + (x[oob, -1] - x[oob, -2]) * gold
    # TODO: golden section search on the left as well
    return icdf(p, cdf, mn, mx, tol=tol, steps=steps)


def gauss_moment(n, D):
    """
    Compute the n-th generalized moment of the standard Gaussian

        E_{x ~ N(0, I)}[(x'Dx)^n]

    !!! quote "Reference"
        1.  Raymond Kan. “From moments of sum to moments of product”.
            In: Journal of Multivariate Analysis 99.3 (2008), pp. 542-554.
        2.  Rong Ge, et al. "Efficient sampling from the Bingham distribution".
            (Corollary A.2) https://arxiv.org/pdf/2010.00137.pdf

    Parameters
    ----------
    n : int
        Moment
    D : int or (*, k, k) tensor
        - If an integer, it is the dimension of the distribution, and
          E_{x ~ N(0, I)}[(x'x)^n] is returned (closed-form solution)
        - If a tensor, it should be a symmetric matrix, and
          E_{x ~ N(0, I)}[(x'Dx)^n] is returned.

    Returns
    -------
    m : (*) tensor
        Moment: E_{x ~ N(0, I)}[(x'Dx)^n]

    """
    if isinstance(D, int):
        return (D + 2*torch.arange(n)).prod().item()

    if n == 0:
        return D.new_ones(D.shape[:-2])

    acc, S, Di = 0, 1, D
    for i in range(1, n+1):
        Di = D if i == 1 else Di.matmul(D)
        DS = Di.diagonal(0, -1, -2).sum(-1) * S
        acc += DS
        S = acc / (2*i)
    return S


def metropolis_hastings(init, pdf, sampler=1e-3,
                        nburn=0, nthin=1, shape=tuple(), log=False):
    """
    Run a Metropolis-Hastings MCMC chain

    Parameters
    ----------
    init : (*batch, *itemshape) tensor
        Initial sample. If `batch`, build multiple chains in parallel.
    pdf : callable((*batch, *shape) tensor) -> (*batch) tensor
        Functions that computes the target PDF (up to a scaling factor)
        or the log-PDF (up to an additive constant).
    sampler : float or callable(tensor) -> tensor
        Symmetric sampler.
        If a float: N(0, sigma^2)
        If None: N(0, 1)
    nburn : int
        Number of burnin steps (i.e., discard the first `nburn` samples)
    nthin : int
        Number of thinning steps (i.e., only keep one every `nthin` samples)
    shape : int or list[int]
        Number of samples to return.
        If 0, return a single sample (not in a list).
    log : bool
        Whether the function is a PDF or a log-PDF

    Returns
    -------
    samples : (*shape, *batch, *itemshape) tensor
        Computed samples
    state : (*batch, *itemshape) tensor
        Return the current state (i.e., last accepted sample) of the chain
    """
    if log:
        def accept(logf, logf0):
            return torch.rand_like(logf).log() <= (logf - logf0)
    else:
        def accept(f, f0):
            return torch.rand_like(f) <= f/f0

    if not callable(sampler):
        sigma = sampler or 1

        def sampler(x):
            return x + torch.randn_like(x) * sigma

    if shape is not None:
        if not isinstance(shape, (list, tuple)):
            shape = [shape]
        shape = torch.Size(shape)
        nsamples = shape.numel()
    else:
        nsamples = 0

    if nsamples:
        samples = init.new_zeros((nsamples,) + init.shape)
    else:
        samples = None

    p = np = n = m = 0
    f0 = pdf(init)
    while True:
        # get a proposal
        sample = sampler(init)
        # compute its (log)-pdf
        f = pdf(sample)
        # accept/reject
        mask = accept(f, f0)
        # update acceptance rate
        p = (np * p + mask.float().mean()) / (np + 1)
        np += 1
        print(f'accept prob: {p:12.6f}', end='\r')
        # if accept, update chain state and increase total sample count
        init = torch.where(mask, sample, init)
        f0 = torch.where(mask, f, f0)
        n += 1
        # if accept and burnin finished, and not skipped, save sample
        if n > nburn:
            if nsamples and (n-nburn-1) % nthin == 0:
                samples[m] = init
                m += 1
            if m == nsamples:
                break

    print('')
    if nsamples:
        samples = samples.reshape(shape + init.shape)
    return samples, init


def rejection_sampling(pdf, pdf_sampler=None, sampler=1e-3, shape=tuple(),
                       log=False, sup=1):
    """
    Run a rejection sampling scheme

    Parameters
    ----------
    pdf : callable((*batch, *item) tensor) -> (*batch) tensor
        Functions that computes the target PDF (up to a scaling factor)
        or the log-PDF (up to an additive constant).
    pdf_sampler : callable((*batch, *item) tensor) -> (*batch) tensor
        Function that computes the sampler's PDF (up to a scaling factor)
        or its log-PDF (up to an additive constant).
    sampler : float or callable(batch: list[int]) -> (*batch, *item) tensor
        Sampler.  If a float: N(0, sigma^2). If None: N(0, 1).
    shape : int or list[int]
        Number of samples to return.
        If 0, return a single sample (not in a list).
    log : bool
        Whether the function is a PDF or a log-PDF
    sup : float or (*batch) tensor
        Upper bound of `pdf(x)/pdf_sampler(x)`.
        If `log`, must be the log of the bound.

    Returns
    -------
    samples : (*batch, *item) tensor
        Computed samples
    """
    if log:
        def accept(logf, logf0):
            return torch.rand_like(logf).log() <= (logf - logf0 - sup)
    else:
        def accept(f, f0):
            return torch.rand_like(f) <= f/(sup*f0)

    if not callable(sampler):
        sigma = sampler or 1

        def sampler(x):
            return x + torch.randn_like(x) * sigma

        def pdf_sampler(x):
            return (x*x) / (-2 * (sigma*sigma))

    if not isinstance(shape, (list, tuple)):
        shape = [shape]
    shape = torch.Size(shape)
    nsamples = shape.numel()

    sample = sampler([nsamples])
    mask = accept(pdf(sample), pdf_sampler(sample))
    nbatchinner = mask.ndim - 1
    p, np = mask.float().mean(), nsamples
    print(f'accept prob: {p:12.6f}', end='\r')

    if nbatchinner == 0:
        while ~mask.all():
            nsamples = (~mask).sum().item()
            new_sample = sampler([nsamples])
            new_mask = accept(pdf(new_sample), pdf_sampler(new_sample))
            sample[~mask] = new_sample
            mask[~mask] = new_mask
            # update acceptance rate
            p = (np * p + nsamples * new_mask.float().mean()) / (np + nsamples)
            np += nsamples
            print(f'accept prob: {p:12.6f}', end='\r')
    else:
        while ~mask.all():
            new_sample = sampler([nsamples])
            new_mask = ~mask & accept(pdf(new_sample), pdf_sampler(new_sample))
            sample[new_mask] = new_sample[new_mask]
            mask |= new_mask

    print('')
    sample = sample.reshape(shape + sample.shape[1:])
    return sample
