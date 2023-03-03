# vesselsynth
Synthetized brain vessels for NN pretraining

## `setup_sampler Function
- Checks to see if value is an instance of random.Sampler class
    - If YES, return as is
    - If it's NOT a list or a tuple, return a random.Dirac object
    - If it's not a random.Sampler class, but it is a list or a tuple, return it as a random.Uniform object with value as its arguments


# `SynthSplineBlock` Class

## Inputs
- `shape`: List of integers
- `voxel_size`: float
  - Default is 0.1. Comment says that this equals 100 mu. Not sure what 100 mu is. Could this be a typo for um (micrometers)?
- `tree_density`: Trees/mm^3



# `LogNormal` Class (inherits from `Sampler`)
- Calculates the log of the variance of `scale` and clamps to zero
  - If the log is zero, return Dirac(mean)


# `Dirac` Class (inherits from `Sampler`)
- The Dirac class is a simple fixed-parameter distribution that always **returns a tensor** with the same value, specified by the **mean** parameter passed to the constructor. The `__call__` method allows the object to be used as a function, allowing it to return tensors with **different shapes** if required.
- Returns either mean, or a tensor filled with mean


# random.py

- Defines statistical distrobutions (Dirac, Uniform, Normal, and Log-Normal)
  - All distrobutions inherit from Sampler class

## Dirac(Sampler)
- Input: mean (of distrobution)
- Returns a fixed paramater distrobution centered around the mean

## Uniform(Sampler)
- Input: Takes mean, fwhm, min, or max
- Returns a uniform distrobution using either the mean and full width @ half maximum, or bounded by min and max

## Normal(Sampler)
- Input: mean or mean and scale
- If given mean and scale, returns normal distrobution
- If given just mean, returns Dirac distrobution about the mean

## LogNormal(Sampler)