import os
import sys
import torch
import nibabel as nib
from textwrap import dedent
from tempfile import gettempdir
from vesselsynth import backend
from vesselsynth.synth import SynthSplineBlock
from vesselsynth.utils import default_affine
from vesselsynth.save_exp import SaveExp


class App:
    """
    Utility class to create a command-line app that synthesizes splines.
    """

    def __init__(self, klass=SynthSplineBlock, shape=256, n=1000, root=None):
        """
        Parameters
        ----------
        klass
            A `SynthSplineBlock` subclass
        shape : [list of] int
            The default shape of the synthesized patches
        n : int
            The default number of patches to synthesize
        root : str
            The default output folder
        """
        self.klass = klass
        self.default_shape = shape
        self.default_n = n
        self.default_root = root or gettempdir()
        self.default_device = 'cuda'
        # init
        self.appname = 'python <path_to_script.py>'
        self.start = 0
        self.stop = self.default_n
        self.root = self.default_root
        self.device = 'cuda'

    def run(self):
        """
        Parse the current command line arguments and run the synthesis
        """
        self.parse()
        self.synth_all()

    def help(self):
        return (dedent(
            f"""
            usage:
                {self.appname} [[<first>] <last>] [-o <output>] [-d <device>] [-s <shape]

            description:
                Generate synthetic axons volumes indexed from <first> to <last>,
                with shape <shape>, using device <device> and and write them
                in directory <output>.

            defaults:
                first  = 0
                last   = {self.default_n}
                shape  = {self.default_shape}
                device = 'cuda' if available else 'cpu'
                output = {self.default_root}
            """  # noqa: E501
        ))

    def parse(self, argv=None):
        """Parse the command line arguments"""
        argv = list(argv or sys.argv)
        self.appname = argv.pop(0)

        if '-h' in argv or '--help' in argv:
            print(help)
            sys.exit()

        # read index range to synthesize
        positionals = []
        while argv:
            if argv[0].startswith('-'):
                break
            positionals += [argv.pop(0)]
        if len(positionals) == 1:
            self.start = 0
            self.stop = int(positionals.pop(0))
        elif len(positionals) == 2:
            self.start = int(positionals.pop(0))
            self.stop = int(positionals.pop(0))

        # read other options
        while argv:
            arg = argv.pop(0)
            if arg in ('-o', '--output'):
                self.root = argv.pop(0)
                continue
            if arg in ('-d', '--device'):
                self.device = argv.pop(0)
                continue
            if arg in ('-s', '--shape'):
                self.shape = [int(argv.pop(0))]
                while argv and not argv[0].startswith('-'):
                    self.shape += [int(argv.pop(0))]
                continue

        # compute output shape
        if not isinstance(self.shape, list):
            self.shape = [self.shape]
        while len(self.shape) < 3:
            self.shape += self.shape[-1:]

        # check device is available
        self.device = torch.device(self.device)
        if self.device.type == 'cuda' and not torch.cuda.is_available():
            print('CUDA not available, using CPU.')
            self.device = 'cpu'

    def synth_all(self):
        """Instantiate synthesized and synthesize all patches"""
        backend.jitfields = True
        synth = self.klass(self.shape, device=self.device)

        # TODO: I don't really like everying in SynthExp.
        #       We should make something a bit more programmable
        root = SaveExp(self.root).main()
        os.makedirs(root, exist_ok=True)

        for n in range(self.start+1, self.stop+1):
            self.synth_one(synth, n, root)

    def synth_one(self, synth, n, root):
        """Synthesize the n-th patch"""
        im, lab, lvl, nlvl, brch, skl, dist = synth()
        affine = default_affine(im.shape[-3:])
        h = nib.Nifti1Header()

        nib.save(nib.Nifti1Image(im.squeeze().cpu().numpy(), affine, h),
                 f'{root}/{n:04d}_prob.nii.gz')
        nib.save(nib.Nifti1Image(dist.squeeze().cpu().numpy(), affine, h),
                 f'{root}/{n:04d}_dist.nii.gz')

        h.set_data_dtype('int32')
        nib.save(nib.Nifti1Image(lab.squeeze().cpu().numpy(), affine, h),
                 f'{root}/{n:04d}_label.nii.gz')

        h.set_data_dtype('uint8')
        nib.save(nib.Nifti1Image(lvl.squeeze().cpu().numpy(), affine, h),
                 f'{root}/{n:04d}_level.nii.gz')
        nib.save(nib.Nifti1Image(nlvl.squeeze().cpu().numpy(), affine, h),
                 f'{root}/{n:04d}_nblevels.nii.gz')
        nib.save(nib.Nifti1Image(brch.squeeze().cpu().numpy(), affine, h),
                 f'{root}/{n:04d}_branch.nii.gz')
        nib.save(nib.Nifti1Image(skl.squeeze().cpu().numpy(), affine, h),
                 f'{root}/{n:04d}_skeleton.nii.gz')
