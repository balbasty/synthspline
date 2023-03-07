import torch
from torch import distributions
import math as pymath


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


def to_tensor(x): # Returns a tensor of floats
    x = torch.as_tensor(x)
    if not x.dtype.is_floating_point:
        x = x.float()
    return x


class Sampler:

    mean = None
    scale = None

    def exp(self):
        return Op1(self, torch.exp)

    def log(self):
        return Op1(self, torch.log)

    def sqrt(self):
        return Op1(self, torch.sqrt)

    def square(self):
        return Op1(self, torch.square)

    def pow(self, other):
        return Op2(self, other, torch.pow)

    def rpow(self, other):
        return Op2(other, self, torch.pow)

    def add(self, other):
        return Op2(self, other, torch.add)

    def sub(self, other):
        return Op2(self, other, torch.sub)

    def mul(self, other):
        return Op2(self, other, torch.mul)

    def div(self, other):
        return Op2(self, other, torch.div)

    def floordiv(self, other):
        return Op2(self, other, torch.floor_divide)

    def matmul(self, other):
        return Op2(self, other, torch.matmul)

    def minimum(self, other):
        return Op2(self, other, torch.minimum)

    def maximum(self, other):
        return Op2(self, other, torch.maximum)

    def clamp(self, min, max):
        return self.minimum(max).maximum(min)

    def clamp_min(self, other):
        return self.maximum(other)

    def clamp_max(self, other):
        return self.minimum(other)

    def __pow__(self, other, modulo=None):
        if modulo is not None:
            raise NotImplementedError('pow+modulo not implemented')
        return self.pow(other)

    def __rpow__(self, other):
        return self.rpow(other)

    def __add__(self, other):
        return self.add(other)

    def __sub__(self, other):
        return self.sub(other)

    def __mul__(self, other):
        return self.mul(other)

    def __truediv__(self, other):
        return self.div(other)

    def __floordiv__(self, other):
        return self.floordiv(other)

    def __matmul__(self, other):
        return self.matmul(other)


class Op1(Sampler):
    def __init__(self, sampler, op):
        if not isinstance(sampler, Sampler):
            sampler = Dirac(sampler)
        self.sampler = sampler
        self.op = op

    def __call__(self, n=tuple()):
        return self.op(self.sampler(n))

    @property
    def mean(self):
        # approximation: assumes peaked sampler
        return self.op(self.sampler.mean)


class Op2(Sampler):
    def __init__(self, sampler1, sampler2, op):
        if not isinstance(sampler1, Sampler):
            sampler1 = Dirac(sampler1)
        if not isinstance(sampler2, Sampler):
            sampler2 = Dirac(sampler2)
        self.sampler1 = sampler1
        self.sampler2 = sampler2
        self.op = op

    def __call__(self, n=tuple()):
        return self.op(self.sampler1(n), self.sampler2(n))

    @property
    def mean(self):
        # approximation: assumes peaked sampler
        return self.op(self.sampler1.mean, self.sampler2.mean)


class OpN(Sampler):
    def __init__(self, samplers, op):
        samplers = [Dirac(f) if not isinstance(f, Sampler) else f
                    for f in samplers]
        self.samplers = samplers
        self.op = op

    def __call__(self, n=tuple()):
        return self.op(*[f(n) for f in self.samplers])

    @property
    def mean(self):
        # approximation: assumes peaked sampler
        return self.op([f.mean for f in self.samplers])


class Dirac(Sampler): #A fixed parameter distribution class with a single mode at the mean parameter.
    """Fixed parameter"""
    def __init__(self, mean): # Constructor taking mean input
        self.mean = to_tensor(mean) # Assigning mean to self.mean after converting it to a tensor.

    def __call__(self, n=tuple()): # Method that allows the object to be called as a function. Takes an optional parameter n representing the shape of the tensor to be returned.
        n = make_tuple(n or []) # Ensures n is a tuple, even if it was not provided.
        mean = to_tensor(self.mean) # Converts self.mean to a tensor.
        return torch.full(n, mean) if n else mean # Returns a tensor filled with the value of mean if n is not an empty tuple. If n is an empty tuple, mean is returned directly.


class Uniform(Sampler):
    '''Generates a uniform distrobution using either the mean, fwhm, min, or max. Returns either PyTorch uniform distribution with the specified bounds or to a Dirac distribution with the specified mean'''
    def __init__(self, *args, **kwargs):
        if 'mean' in kwargs: # If we have a mean, set the self.mean to it and we're done
            self.mean = to_tensor(kwargs['mean'])
            if 'fwhm' in kwargs: # The full width at half maximum (https://en.wikipedia.org/wiki/Full_width_at_half_maximum)
                self.fwhm = to_tensor(kwargs['fwhm'])
                self.scale = self.fwhm / pymath.sqrt(12)
            else:
                self.scale = to_tensor(kwargs.get('scale', 0))
                self.fwhm = self.scale * pymath.sqrt(12)
            self.min = self.mean - self.fwhm / 2
            self.max = self.mean + self.fwhm / 2
        else:
            if len(args) == 1:
                self.max = to_tensor(args[0])
                self.min = torch.zeros([]).to(self.max)
            elif len(args) == 2:
                self.min = to_tensor(args[0])
                self.max = to_tensor(args[1])
            else:
                self.max = kwargs['max']
                self.min = kwargs.get('min', torch.zeros([]).to(self.max))
            self.fwhm = self.max - self.min
            self.scale = self.fwhm / pymath.sqrt(12)
            self.mean = (self.min + self.max) / 2

        if self.scale:
            self.sampler = distributions.Uniform(self.min, self.max).sample
        else:
            self.sampler = Dirac(self.mean)

    def __call__(self, n):
        return self.sampler(make_tuple(n or []))


class Normal(Sampler):
    def __init__(self, mean, scale=0):
        self.mean = to_tensor(mean)
        self.scale = to_tensor(scale)

        if self.scale:
            self.sampler = distributions.Normal(self.mean, self.scale).sample
        else:
            self.sampler = Dirac(mean)

    def __call__(self, n):
        return self.sampler(make_tuple(n or []))


class LogNormal(Sampler): # Define a class LogNormal that inherits from Sampler

    def __init__(self, mean, scale=0): # Define the constructor with arguments mean and scale, convert to tensors
        self.mean = to_tensor(mean)
        self.scale = to_tensor(scale)
        if self.scale: # If scale is not zero
            var = self.scale * self.scale # Calculate the variance
            var_log = (1 + var / self.mean.square()).log().clamp_min(0) # Clamp the log of the variance to zero # Calculating the log of the variance using this math https://en.wikipedia.org/wiki/Log-normal_distribution.
            if not var_log: # if it's zero
                self.sampler = Dirac(mean) # The sampler is a
            else:
                mean_log = self.mean.log() - var_log / 2
                scale_log = var_log.sqrt()
                self.mean_log = mean_log
                self.scale_log = scale_log
                self.sampler = distributions.LogNormal(mean_log, scale_log).sample
        else:
            self.sampler = Dirac(mean)

    def __call__(self, n=tuple()):
        return self.sampler(make_tuple(n or []))
