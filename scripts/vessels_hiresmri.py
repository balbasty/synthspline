import sys
import interpol
from vesselsynth.synth import SynthVesselHiResMRI
from nitorch import io, spatial
import os
from sys import argv
import torch

# Use faster jitfield backend to compute splines
interpol.backend.jitfields = True

# defaults
home = os.environ.get('HOME')
root = f'{home}/links/data/vessels/synth_new/230123'
device = 'cuda'
shape = 128
start = 0
stop = 1000

help = f"""
python <path_to_script.py> [[<first>] <last>] [-o <output>] [-d <device>] [-s <shape]

>> Generate synthetic vessel volumes indexed from <first> to <last>, 
>> with shape <shape>, using device <device> and and write them 
>> in directory <output>.

>> Defaults:
>> - first  = 0
>> - last   = 1000
>> - shape  = 128
>> - device = 'cuda' if available else 'cpu'
>> - output = {root}
"""

if '-h' in argv or '--help' in argv:
    print(help)
    sys.exit()

# read index range to synthesize
argv = list(argv[1:])
positionals = []
while argv:
    if argv[0].startswith('-'):
        break
    positionals += [argv.pop(0)]
if len(positionals) == 1:
    start = 0
    stop = int(positionals.pop(0))
elif len(positionals) == 2:
    start = int(positionals.pop(0))
    stop = int(positionals.pop(0))

# read other options
while argv:
    arg = argv.pop(0)
    if arg in ('-o', '--output'):
        root = argv.pop(0)
        continue
    if arg in ('-d', '--device'):
        device = argv.pop(0)
        continue
    if arg in ('-s', '--shape'):
        shape = [int(argv.pop(0))]
        while argv and not argv[0].startswith('-'):
            shape += [int(argv.pop(0))]
        continue

# compute output shape
if not isinstance(shape, list):
    shape = [shape]
while len(shape) < 3:
    shape += shape[-1:]

# check backend
device = torch.device(device)
if device.type == 'cuda' and not torch.cuda.is_available():
    print('CUDA not available, using CPU.')
    device = 'cpu'

# setup synthesizer
synth = SynthVesselHiResMRI(shape, device=device)

# synth
os.makedirs(root, exist_ok=True)
for n in range(start+1, stop+1):

    im, lab, lvl, nlvl, brch, skl = synth()
    affine = spatial.affine_default(im.shape[-3:])

    io.savef(im.squeeze(), f'{root}/{n:04d}_vessels_prob.nii.gz', affine=affine)
    io.save(lab.squeeze(), f'{root}/{n:04d}_vessels_label.nii.gz', affine=affine, dtype='int32')
    io.save(lvl.squeeze(), f'{root}/{n:04d}_vessels_level.nii.gz', affine=affine, dtype='uint8')
    io.save(nlvl.squeeze(), f'{root}/{n:04d}_vessels_nblevels.nii.gz', affine=affine, dtype='uint8')
    io.save(brch.squeeze(), f'{root}/{n:04d}_vessels_branch.nii.gz', affine=affine, dtype='uint8')
    io.save(skl.squeeze(), f'{root}/{n:04d}_vessels_skeleton.nii.gz', affine=affine, dtype='uint8')
