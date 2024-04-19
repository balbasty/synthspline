# synthspline

Synthetic tubular structures (vessels, axons) for NN pretraining

## Installation

### 1. Install core dependencies with conda (strongly advised)

Currently, only `cppyy` v2.3 is supported. Most versions of pytorch and
cupy should work, although not all were exhaustively tested.
PyTorch 1.13 works for sure.

CPU only
```shell
conda install -c pytorch -c conda-forge pytorch=1.13 cppyy=2.3
```

CUDA
```shell
export CUDA_VERSION=11.1
conda install -c pytorch -c nvidia -c conda-forge \
    pytorch=1.13 pytorch-cuda=${CUDA_VERSION} \
    cupy cuda-version=${CUDA_VERSION} \
    cppyy=2.3
```

It is still unclear to me if torch and cupy need to be installed with the
same cuda version. Refer to their respective installation guides if needed:
- [PyTorch](https://pytorch.org/get-started/previous-versions/)
- [CuPy](https://docs.cupy.dev/en/stable/install.html)
- [cppyy](https://cppyy.readthedocs.io/en/latest/installation.html)

### 2. [Option 1] Install other dependencies with conda

```shell
conda install -c balbasty -c conda-forge \
    jittfields torch-interpol torch-distmap nibabel
```

### 2. [Option 2] Install other dependencies with pip

```shell
pip install jitfields torch-interpol torch-distmap nibabel
```

### 3. Install synthspline

```shell
pip install git+https://github.com/balbasty/synthspline
```

## Getting started

### Synthesizing labels

The class `synthspline.labelsynth.SynthSplineBlock` generates
random trees of cubic splines. Its parameters are defined in its `defaults`
subclass.

```python
from synthspline.labelsynth import SynthSplineBlock

synthesizer = SynthSplineBlock()

for _ in range(nb_samples):
    sample = synthesizer()
```

Each sample is a named tuple with keys `['vessels', 'labels', 'levelmap',
'nblevelmap', 'branchmap', 'skeleton', 'dist']`. Each key contains a
tensor with a batch dimension, a (singleton) channel dimension and three
spatial dimensions.

All parameters defines in `SynthSplineBlock.defaults` can be user-defined.
Most of them take a `synthspline.random.Sampler` object.

```python
from synthspline.labelsynth import SynthSplineBlock
from synthspline.random import Uniform, LogNormal, RandInt

synthesizer = SynthSplineBlock(
    shape=[128]*3,
    nb_levels=RandInt(1, 5),
    tree_density=Uniform(2.),
)

for _ in range(nb_samples):
    sample = synthesizer()
```

Subclasses with defaults tailored to different types of images are
defined in `synthspline.labelzoo`.

### Synthesizing images

A bunch of classes that conditionally generate an image from spline-based
labels are defined in `synthspline.imagezoo`.

### Commandline scripts

We provide scripts that generate labels (saved as nifti files) for each
image type in `scripts/`. Their usage is typically:
```
usage:
    python <path_to_script.py> [[<first>] <last>] [-o <output>] [-d <device>] [-s <shape]

description:
    Generate synthetic spline labels indexed from <first> to <last>,
    with shape <shape>, using device <device> and and write them
    in directory <output>.
```

We also provide scrips that generate images conditioned on labels.
Their usage is:
```
usage:
    python <path_to_script.py> [<input>] [-o <output>] [-n <n>] [-s <subset>] [-d <device>]

description:
    Generate <n> synthetic intensity volumes for each label map
    found in <input> and write them in directory <output>.
    A <subset> of label volumes can be specified as either
    a list of indices, or a range of indices ("<begin>:<end>",
    or "<begin>:" or ":<end>" or ":").
```

### Samplers

We define the following samplers in `synthspline.random`:

```python
Dirac()               # default (0)
Dirac(0)              # positional variant
Dirac(value=0)        # keyword variant
Dirac(loc=0)          # alias
Dirac(mean=0)         # alias


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

Furthermore, sampelrs can be derived by applying determinstic functions
to any existing sampler. The list of supported methods is:
```python
exp()                   # alias: `e ** x`
log()
sqrt()                  # alias: `x ** 0.5`
square()                # alias: `x ** 2`
pow(other)              # alias: `x ** y`
rpow(other)             # alias: `y ** x`
add(other)              # alias: `x + y`
sub(other)              # alias: `x - y`
mul(other)              # alias: `x * y`
div(other)              # alias: `x / y`
floordiv(other)         # alias: `x // y`
matmul(other)           # alias: `x @ y`
equal(other)            # alias: `x == y`
not_equal(other)        # alias: `x != y`
less(other)             # alias: `x < y`
less_equal(other)       # alias: `x <= y`
greater(other)          # alias: `x > y`
greater_equal(other)    # alias: `x >= y`
minimum(other)
maximum(other)
clamp(min, max)
clamp_min(min)
clamp_max(max)
```

### Spline rasterization

The algorithm that encodes and rasterize splines is defined in
`synthspline.curves`. When `jitfields` is installed and activated by
setting `synthspline.backend.jitfields = True`, a fast cuda implementation
is used. Otherwise, it uses a slower pytorch-based implementation.
