import os
import sys
import json
import torch
import interpol
import nibabel as nib

sys.path.append("../vesselsynth/")
from vesselsynth.synth import SynthVesselOCT
from vesselsynth.io import default_affine
from vesselsynth.save_exp import SaveExp

# Use faster jitfield backend to compute splines
interpol.backend.jitfields = True
os.environ['PYTORCH_JIT_USE_NNC_NOT_NVFUSER'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

class VesselsOCT(object):

    def __init__(self):
        self.device = 'cuda'                                            # "cuda" or "cpu"
        self.json_params = json.load(open("vesselsynth_params.json"))   # This is the json file that should be one directory above this one. Defines all variables
        self.shape = self.json_params['shape']                           
        self.n_volumes = self.json_params['n_volumes']
        self.root = self.json_params['output_path']
        self.header = nib.Nifti1Header()

    def main(self):

        os.makedirs(self.root, exist_ok=True)
        self.logParams()
        self.backend(self.device)
        self.outputShape()

        for n in range(self.n_volumes):
            synth_key = ['prob', 'label', "level", "nb_levels", "branch", "skeleton"]
            synth_vols = SynthVesselOCT(shape=self.shape, device=self.device)()

            for i in range(len(synth_key)):
                self.saveVolume(n, synth_key[i], synth_vols[i])            

    def backend(self, device):
        self.device = torch.device(device)
        if self.device.type == 'cuda' and not torch.cuda.is_available():
            print('CUDA not available, using CPU.')
            self.device = 'cpu'

    def outputShape(self):
        if not isinstance(self.shape, list):
            self.shape = [self.shape]
        while len(self.shape) < 3:
            self.shape += self.shape[-1:]

    def saveVolume(self, volume_n, volume_name, volume):
        affine = default_affine(volume.shape[-3:])
        nib.save(nib.Nifti1Image(volume.squeeze().cpu().numpy(), affine, self.header),
                 f'{self.root}/{volume_n:04d}_vessels_{volume_name}.nii.gz')
        
    def logParams(self):
        json_object = json.dumps(self.json_params, indent=4)
        file = open(f'{self.root}/vesselsynth_params.json', 'w')
        file.write(json_object)
        file.close()

VesselsOCT().main()