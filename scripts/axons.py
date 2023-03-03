from vesselsynth.synth import SynthAxon
from nitorch import io, spatial
import os
from sys import argv

synth = SynthAxon([192]*3, device='cuda')
home = os.environ.get('HOME')
root = f'{home}/links/data/AY/synth/2022-10-03'

if len(argv) == 2:
    start = 0
    stop = int(argv[1])
elif len(argv) == 3:
    start = int(argv[1])
    stop = int(argv[2])
else:
    start = 0
    stop = 1000

for n in range(start+1, stop+1):

    im, lab, lvl, brch = synth()
    affine = spatial.affine_default(im.shape[-3:])

    io.savef(im.squeeze(), f'{root}/{n:04d}_axons_prob.nii.gz', affine=affine)
    io.save(lab.squeeze(), f'{root}/{n:04d}_axons_label.nii.gz', affine=affine, dtype='int32')
    io.save(lvl.squeeze(), f'{root}/{n:04d}_axons_level.nii.gz', affine=affine, dtype='uint8')
    io.save(brch.squeeze(), f'{root}/{n:04d}_axons_branch.nii.gz', affine=affine, dtype='uint8')
