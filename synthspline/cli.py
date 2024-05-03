__all__ = ['LabelApp', 'ImageApp']
import os
import sys
import torch
import yaml
import nibabel as nib
from textwrap import dedent
from tempfile import gettempdir
from synthspline import backend
from synthspline.labelsynth import SynthSplineBlock
from synthspline.datasets import SynthSplineDataset
from synthspline.utils import default_affine
from synthspline.save_exp import SaveExp


class LabelApp:
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
        self.appname = 'python <path_to_script.py>'

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
                Generate synthetic spline labels indexed from <first> to <last>,
                with shape <shape>, using device <device> and and write them
                in directory <output>.

            defaults:
                first  = 0
                last   = {self.default_n}
                shape  = {self.default_shape}
                device = "cuda" if available else "cpu"
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

        self.start = 0
        self.stop = self.default_n
        self.shape = self.default_shape
        self.device = self.default_device
        self.root = self.default_root

        # read index range to synthesize
        positionals = []
        while argv:
            if argv[0].startswith('-'):
                break
            positionals += [argv.pop(0)]
        if len(positionals) == 1:
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
        with open(f'{root}/params.yaml', 'w') as f:
            yaml.safe_dump(synth.params.serialize(), f, sort_keys=False)

        for n in range(self.start+1, self.stop+1):
            self.synth_one(synth, n, root)

    def synth_one(self, synth, n, root):
        """Synthesize the n-th patch"""
        prob, lab, lvl, nlvl, brch, skl, dist = synth()
        affine = default_affine(prob.shape[-3:])
        h = nib.Nifti1Header()

        nib.save(nib.Nifti1Image(prob.squeeze().cpu().numpy(), affine, h),
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


class ImageApp:
    """
    Utility class to create a command-line app that synthesizes images.
    """

    def __init__(self, klass, inp_keys=['label'], out_keys=['image', 'ref'],
                 n=10, inp='.', out=None):
        """
        Parameters
        ----------
        klass
            A `Synth*Image` class or instance.
        keys : [list of] str
            Types of synthetic labels to use for synthesis.
        n : int
            The default number of images per label to synthesize.
        root : str
            The default input folder.
        """
        self.klass = klass
        self.inp_keys = inp_keys
        self.out_keys = out_keys
        self.default_n = n
        self.default_inp = inp
        self.default_out = out
        self.default_device = 'cuda'
        self.default_subset = None
        self.appname = 'python <path_to_script.py>'

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
                {self.appname} [<input>] [-o <output>] [-n <n>] [-s <subset>] [-d <device>]

            description:
                Generate <n> synthetic intensity volumes for each label map
                found in <input> and write them in directory <output>.
                A <subset> of label volumes can be specified as either
                a list of indices, or a range of indices ("<begin>:<end>",
                or "<begin>:" or ":<end>" or ":").

            defaults:
                input  = "."
                output = "<input>/images"
                n      = {self.default_n}
                subset = ":"
                device = "cuda" if available else "cpu"
            """  # noqa: E501
        ))

    def parse(self, argv=None):
        """Parse the command line arguments"""
        argv = list(argv or sys.argv)
        self.appname = argv.pop(0)

        if '-h' in argv or '--help' in argv:
            print(help)
            sys.exit()

        # set defaults
        self.inp = self.default_inp
        self.out = None
        self.n = self.default_n
        self.subset = None
        self.device = self.default_device

        # read index range to synthesize
        positionals = []
        while argv:
            if argv[0].startswith('-'):
                break
            positionals += [argv.pop(0)]
        if len(positionals) == 1:
            self.inp = positionals.pop(0)
        if positionals:
            raise ValueError('Too many positional arguments')

        # read other options
        while argv:
            arg = argv.pop(0)
            if arg in ('-o', '--output'):
                self.out = argv.pop(0)
                continue
            if arg in ('-d', '--device'):
                self.device = argv.pop(0)
                continue
            if arg in ('-n', '--nb-images'):
                self.n = int(argv.pop(0))
                continue
            if arg in ('-s', '--subset'):
                self.subset = []
                while argv and not argv[0].startswith('-'):
                    if ':' in argv[0]:
                        arg = argv.pop(0).split(':')
                        begin = end = step = None
                        if len(arg) > 0:
                            begin = int(arg[0]) if arg[0] else None
                        if len(arg) > 1:
                            end = int(arg[1]) if arg[1] else None
                        if len(arg) > 2:
                            step = int(arg[2]) if arg[2] else None
                        self.subset += [slice(begin, end, step)]
                    else:
                        self.subset += [int(argv.pop(0))]
                continue

        # default out
        if not self.out:
            self.out = os.path.join(self.inp, 'images')

        # check device is available
        self.device = torch.device(self.device)
        if self.device.type == 'cuda' and not torch.cuda.is_available():
            print('CUDA not available, using CPU.')
            self.device = 'cpu'

    def synth_all(self):
        """Instantiate synthesized and synthesize all patches"""
        backend.jitfields = True
        if isinstance(self.klass, type):
            synth = self.klass()
        else:
            synth = self.klass
        dataset = SynthSplineDataset(self.inp, keys=self.inp_keys,
                                     subset=self.subset)
        loader = torch.utils.data.DataLoader(dataset, batch_size=1)

        # TODO: I don't really like everying in SynthExp.
        #       We should make something a bit more programmable
        out = SaveExp(self.out).main()
        os.makedirs(out, exist_ok=True)

        for i, dat in enumerate(loader):
            dat = [x.to(self.device) for x in dat]
            for j in range(self.n):
                print(f'Synthesizing image {j+1}/{self.n} '
                      f'from label {i+1}/{len(loader)}', end='\r')
                self.synth_one(synth, dat, i * self.n + j, out)
        print('')

    def synth_one(self, synth, dat, n, root):
        """Synthesize the n-th patch"""
        out = synth(*dat)
        out = {key: x for key, x in zip(self.out_keys, out)}
        affine = default_affine(list(out.values())[0].shape[-3:])
        h = nib.Nifti1Header()

        for key, val in out.items():
            if val.dtype == torch.bool:
                h.set_data_dtype('uint8')
            else:
                h.set_data_dtype(str(val.dtype).split('.')[-1])
            nib.save(nib.Nifti1Image(val.squeeze().cpu().numpy(), affine, h),
                     f'{root}/{n:04d}_{key}.nii.gz')
