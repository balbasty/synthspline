from vesselsynth.synth import SynthAxon, SynthVesselHiResMRI
import matplotlib.pyplot as plt
from nitorch.plot.colormaps import depth_to_rgb
from nitorch import io, spatial

synth = SynthAxon([128]*3)
# synth = SynthVesselHiResMRI([128]*3)

im, lab, lvl, brch = synth()

affine = spatial.affine_default(im.shape[-3:])
io.savef(im.squeeze(), '/Users/yb947/Dropbox/data/axons_prob.nii.gz', affine=affine)
io.save(lab.squeeze(), '/Users/yb947/Dropbox/data/axons_label.nii.gz', affine=affine, dtype='int32')
io.save(lvl.squeeze(), '/Users/yb947/Dropbox/data/axons_level.nii.gz', affine=affine, dtype='uint8')
io.save(brch.squeeze(), '/Users/yb947/Dropbox/data/axons_branch.nii.gz', affine=affine, dtype='uint8')

# plt.imshow(depth_to_rgb(im.squeeze().float()))
# plt.show()

foo = 0
