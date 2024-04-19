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
    def unserialize(cls, json):
        """
        Unserialize a Sampler.

        Assumes that `json` contains the sampler name as key and its
        arguments as value.
        """
        if not isinstance(json, dict) or len(json) != 1:
            raise ValueError('Cannot interpret this object as a Sampler')
        (sampler, args), = json.items()
        if '.' not in sampler:
            sampler = 'vesselsynth.random.' + sampler
        sampler = import_fullname(sampler)
        return sampler.unserialize(args)

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
        keys_loc = ('mean', 'loc')
        keys_scl = ('fwhm', 'std', 'var', 'scale')
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
        keys_loc = ('mean', 'loc', 'logmean')
        keys_scl = ('fwhm', 'std', 'var', 'scale', 'logstd', 'logvar')
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
