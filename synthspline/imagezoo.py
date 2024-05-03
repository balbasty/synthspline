from torch import nn
import torch
import math as pymath
import random as pyrandom
import cornucopia as cc


class AutoBatchTransform(nn.Module):
    """Base class for `Synth*Image` transforms"""

    class XForm(cc.Transform):
        # A class like this must be implemented in each AutoBatchTransform
        pass

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.xform = cc.batch(self.XForm(*args, **kwargs))

    def forward(self, *args, **kwargs):
        # automatically unpack input arguments if they are passed as a
        # list instead of separate arguments.
        if len(args) == 1 and isinstance(args[0], (list, tuple)):
            args = tuple(args[0])
        return self.xform(*args, **kwargs)


class SynthVesselImage(AutoBatchTransform):
    """Synthesize an image from vessel labels"""

    class XForm(cc.Transform):

        def __init__(self):
            super().__init__()

            self.flip = cc.RandomFlipTransform()
            self.mask = cc.RandomSmoothLabelMap(cc.Fixed(4), 8)
            self.label_map = cc.RandomSmoothLabelMap(16, 8)
            self.erode_label = cc.RandomErodeLabelTransform(
                radius=5,
                new_labels=True,
            )
            self.gmm = cc.RandomGaussianMixtureTransform()
            self.gamma = cc.RandomGammaTransform((0, 5))
            self.bias = cc.RandomMulFieldTransform()
            self.smooth = cc.RandomSmoothTransform()
            self.noise = cc.RandomChiNoiseTransform(0.05)
            self.rescale = cc.QuantileTransform()

        def forward(self, x, c=None):
            """

            Parameters
            ----------
            label : (1, *spatial) tensor
                Volume of unique vessel labels (each vessel has a unique id)
            centerline : (1, *spatial) tensor, optional
                Volume of centerlines

            Returns
            -------
            image : (1, *spatial) tensor
                Synthetic image
            label : (1, *spatial) tensor[bool]
                Ground truth labels (binary mask)
            centerline : (1, *spatial) tensor[bool], optional
                Ground truth centerlines (binary mask)
            """
            # Expects tensors with a channel but with no batch dimension

            # x: volume of unique vessel labels (each vessel has a unique id)
            if c is not None:
                x, c = self.flip(x, c)
            else:
                x = self.flip(x)
            m = self.mask(x) > 0
            x *= m
            if c is not None:
                c *= m

            # generate background classes
            y = self.label_map(x)
            y = self.erode_label(y)

            # create groups of vessels that have the same intensity and give
            # them a unique label
            label_max = y.max()
            vessel_labels = list(sorted(x.unique().tolist()))[1:]
            pyrandom.shuffle(vessel_labels)
            nb_groups = cc.RandInt(1, 5)()
            nb_per_group = int(pymath.ceil(len(vessel_labels) / nb_groups))
            for i in range(nb_groups):
                labels = vessel_labels[i*nb_per_group:(i+1)*nb_per_group]
                for label in labels:
                    y.masked_fill_(x == label, label_max + i + 1)

            # generate image
            y = self.gmm(y)
            y = self.gamma(y)
            y = self.bias(y)
            y = self.smooth(y)
            y = self.noise(y)
            y = self.rescale(y)

            x = x > 0
            if c is not None:
                c = c > 0
                return y, x, c
            else:
                return y, x


class SynthVesselOCTImage(AutoBatchTransform):
    """Synthesize an OCT image from vessel labels"""

    class XForm(cc.Transform):

        def __init__(self, softlabel=False, centerline=False):
            """
            Parameters
            ----------
            softlabel : bool
                Provide and return soft ground truth labels
            centerline : bool
                Provide and return ground truth centerline labels
            """
            super().__init__()
            self.softlabel = softlabel
            self.centerline = centerline

            # ground truth
            self.flip = cc.RandomFlipTransform()
            self.mask = cc.RandomSmoothLabelMap(cc.Fixed(4), 8)

            # background
            self.label_map = cc.RandomSmoothLabelMap(16, shape=8)
            self.erode_label = cc.RandomErodeLabelTransform(
                radius=5, new_labels=True
            )

            # noisy vessels
            self.erodespline = cc.RandomSmoothMorphoLabelTransform(
                shape=128,
                max_radius=4,
                min_radius=-4,
            )
            self.shallow = cc.RandomSmoothShallowLabelTransform(
                shape=128,
                max_width=5,
            ) * 0.3  # only apply 30% of the time
            self.noisylabel = cc.randomize(cc.SmoothBernoulliTransform)(
                shape=cc.RandInt(2, 128),
                prob=cc.Uniform(0, 0.2),
            )

            # crappy stuff
            self.balls = cc.randomize(cc.SmoothBernoulliDiskTransform)(
                prob=cc.Uniform(0.01),
                radius=cc.Uniform(5),
                value='random',
            )

            # intensity randomization
            self.gmm = cc.RandomGaussianMixtureTransform()
            self.gamma = cc.RandomGammaTransform((0, 5))
            self.bias = cc.RandomMulFieldTransform()
            self.smooth = cc.RandomSmoothTransform()
            self.noise = cc.RandomGammaNoiseTransform(0.1)
            self.rescale = cc.QuantileTransform()

        def forward(self, label, softlabel=None, centerline=None):
            """
            Parameters
            ----------
            label : (1, *spatial) tensor
                Volume of unique vessel labels (each vessel has a unique id)
            softlabel : (B, 1, *spatial) tensor, optional
                Volume of vessel probabilities
            centerline : (B, 1, *spatial) tensor, optional
                Volume of centerlines

            Returns
            -------
            image : (1, *spatial) tensor
                Synthetic image
            label : (1|2, *spatial) tensor
                Ground truth labels
                If `softlabel`, dtype is float, else it is int
                If `centerline`, the second channel contains the
                centerline label
            """
            x, p, c = label, softlabel, centerline

            flip = self.flip.make_final()
            x = flip(x)
            if p is not None:
                p = flip(p)
            if c is not None:
                c = flip(c)

            # mask out parts of the ground truth
            m = self.mask(x) > 0
            x = x * m
            if p is not None:
                p = p * m
            else:
                p = x > 0
            if c is not None:
                c = c * m

            # noisify vessel labels
            v = x.clone()
            v0, v = v, self.erode_axon(v)
            while not v.any():
                v = self.erode_axon(v0)
            v0, v = v, self.shallow(v)
            while not v.any():
                v = self.shallow(v0)
            del v0
            v = self.noisylabel(v)

            # create groups of vessels that have the same intensity and give
            # them a unique label
            y = torch.zeros_like(x, dtype=torch.int)
            vessel_labels = list(sorted(v.unique().tolist()))[1:]
            pyrandom.shuffle(vessel_labels)
            nb_groups = cc.RandInt(1, 5)()
            nb_per_group = int(pymath.ceil(len(vessel_labels) / nb_groups))
            for i in range(nb_groups):
                labels = vessel_labels[i*nb_per_group:(i+1)*nb_per_group]
                for label in labels:
                    y.masked_fill_(v == label, i + 1)
            del v

            y = self.gmm(y)
            if p is not None:
                y *= p

            # generate background image
            if self.background:
                z = self.label_map(y)
                z = self.erode_label(z)
                z = self.gmm(z)
                y += (1 - p) * z
                del z

            # sample balls
            y = self.balls(y)

            # generate image
            y = self.gmm(y)
            y = self.gamma(y)
            y = self.bias(y)
            y = self.smooth(y)
            y = self.noise(y)
            y = self.rescale(y)

            if c is not None:
                return y, torch.cat([p, c > 0])
            else:
                return y, p


class SynthVesselPhotoImage(AutoBatchTransform):
    """Synthesize a blockface photograph image from vessel labels"""

    class XForm(cc.Transform):

        def __init__(self):
            super().__init__()

            # ground truth
            self.flip = cc.RandomFlipTransform()
            self.mask = cc.RandomSmoothLabelMap(cc.Fixed(4), shape=8)

            # background
            self.label_map = cc.RandomSmoothLabelMap(16, shape=8)
            self.erode_label = cc.RandomErodeLabelTransform(
                radius=5, new_labels=True
            )

            # noisy vessels
            self.erodespline = cc.RandomSmoothMorphoLabelTransform(
                shape=128,
                max_radius=2,
                min_radius=-2,
            ) * 0.1
            self.shallow = cc.RandomSmoothShallowLabelTransform(
                shape=128,
                max_width=2,
            ) * 0.1
            # self.noisylabel = cc.randomize(cc.SmoothBernoulliTransform)(
            #    shape=cc.RandInt(2, 128),
            #    prob=cc.Uniform(0, 0.2),
            # )

            # crappy stuff
            self.balls = cc.randomize(cc.SmoothBernoulliDiskTransform)(
                prob=cc.Uniform(0.01),
                radius=cc.Uniform(5),
                value='random',
            )

            # intensity randomization
            self.gmm = cc.RandomGaussianMixtureTransform(fwhm=5)
            self.gamma = cc.RandomGammaTransform((0, 5))
            self.smooth = cc.RandomSmoothTransform(fwhm=1) * 0.5
            self.bias = cc.RandomSlicewiseMulFieldTransform(
                thickness=cc.Fixed(1),
                slice=-1,
            )
            self.noise = (
                cc.RandomGammaNoiseTransform(0.02) |
                cc.RandomChiNoiseTransform(0.02, 2)
            )
            self.thick = cc.RandomLowResSliceTransform(
                resolution=cc.Uniform(8, 12),
                thickness=cc.Uniform(0, 0.5),
                noise=self.bias + self.noise,
                axis=-1,
            )
            self.rescale = cc.QuantileTransform()

        def forward(self, label):
            """
            Parameters
            ----------
            label : (1, *spatial) tensor
                Volume of unique vessel labels (each vessel has a unique id)

            Returns
            -------
            image : (1, *spatial) tensor
                Synthetic image
            label : (1, *spatial) tensor
                Ground truth labels
            """
            x = label
            x = self.flip(x)

            # mask out parts of the ground truth
            m = self.mask(x) > 0
            x = x * m
            p = x > 0

            # noisify vessel labels
            v = x.clone()
            if v.any():
                v0, v = v, self.erodespline(v)
                tried = 1
                while not v.any():
                    tried += 1
                    if tried > 5:
                        break
                    v = self.erodespline(v0)
                v0, v = v, self.shallow(v)
                tried = 1
                while not v.any():
                    tried += 1
                    if tried > 5:
                        break
                    v = self.shallow(v0)
                del v0
            # v = self.noisylabel(v)

            # create groups of vessels that have the same intensity and give
            # them a unique label
            y = torch.zeros_like(x, dtype=torch.int)
            vessel_labels = list(sorted(v.unique().tolist()))[1:]
            pyrandom.shuffle(vessel_labels)
            nb_groups = cc.RandInt(1, 5)()
            nb_per_group = int(pymath.ceil(len(vessel_labels) / nb_groups))
            for i in range(nb_groups):
                labels = vessel_labels[i*nb_per_group:(i+1)*nb_per_group]
                for label in labels:
                    y.masked_fill_(v == label, i + 1)
            del v

            y = self.gmm(y)
            if p is not None:
                y *= p

            # generate background image
            z = self.label_map(y)
            z = self.erode_label(z)
            z = self.gmm(z)
            y += (~p) * z
            del z

            # sample balls
            # y = self.balls(y)

            # generate image
            y = self.gamma(y)
            y = self.smooth(y)
            y = self.thick(y)
            y = self.rescale(y)

            return y, p


class SynthAxonImage(AutoBatchTransform):
    """Synthesize a LSM image from axon labels"""

    class XForm(cc.Transform):

        def __init__(self, background=0.5):
            """
            Parameters
            ----------
            background : float
                Probability of generate random shapes in the background.
                If `0`, the background is plain dark.
            """
            super().__init__()
            self.background = background

            self.flip = cc.RandomFlipTransform()
            self.erode_axon = cc.RandomSmoothMorphoLabelTransform(
                shape=128,
                max_radius=4,
                min_radius=-4,
            )
            self.shallow = cc.RandomSmoothShallowLabelTransform(
                shape=128,
                max_width=3,
            ) * 0.3
            self.noisylabel = cc.randomize(cc.SmoothBernoulliTransform)(
                shape=cc.RandInt(2, 128),
                prob=cc.Uniform(0, 0.2),
            )
            self.soma = cc.randomize(cc.SmoothBernoulliDiskTransform)(
                shape=cc.RandInt(2, 16),
                prob=cc.Uniform(0, 0.02),
                radius=10,
                returns='disks',
            )
            self.label_map = cc.RandomSmoothLabelMap(16, 8)
            self.erode_label = cc.RandomErodeLabelTransform(
                radius=5, new_labels=True)
            self.gmm = cc.RandomGaussianMixtureTransform(
                background=None if self.background else 0)
            self.gamma = cc.RandomGammaTransform((0, 5))
            self.addbias = cc.RandomAddFieldTransform(vmin=0, vmax=0.25)
            self.mulbias = cc.RandomMulFieldTransform(symmetric=1)
            self.smooth = cc.RandomSmoothTransform(2)
            self.noise = (
                cc.RandomChiNoiseTransform() | cc.RandomGammaNoiseTransform()
            )
            self.rescale = cc.QuantileTransform()

        def forward(self, lab, prob=None):
            """
            Parameters
            ----------
            lab : (1, *spatial) tensor
                Volume of unique labels (each vessel has a unique id)
            prob : (1, *spatial) tensor
                Volume of probability

            Returns
            -------
            image : (1, *spatial) tensor
                Synthetic image
            prob : (1, *spatial) tensor
                Ground truth probability
            """
            # lab: volume of unique axon labels (each axon has a unique id)
            # prob: volume of axon probability

            if isinstance(lab, (list, tuple)):
                lab, prob = lab

            lab, prob = self.flip(lab, prob)

            # generate axon class from which to sample the gmm
            # (imperfect compared to ground truth label)
            v = lab.clone()
            v0, v = v, self.erode_axon(v)
            while not v.any():
                v = self.erode_axon(v0)
            v0, v = v, self.shallow(v)
            while not v.any():
                v = self.shallow(v0)
            del v0
            v = self.noisylabel(v)

            # create groups of axons that have the same intensity and
            # give them a unique label
            y = torch.zeros_like(lab, dtype=torch.int)
            vessel_labels = list(sorted(v.unique().tolist()))[1:]
            pyrandom.shuffle(vessel_labels)
            nb_groups = cc.RandInt(1, 5)()
            nb_per_group = int(pymath.ceil(len(vessel_labels) / nb_groups))
            for i in range(nb_groups):
                labels = vessel_labels[i*nb_per_group:(i+1)*nb_per_group]
                for label in labels:
                    y.masked_fill_(v == label, i + 1)
                soma = self.soma(y)
                y.masked_fill(soma > 0, i + 1)
            del v

            # generate foreground image
            y = self.gmm(y)
            y *= prob

            # generate background image
            if cc.Uniform(1)() < self.background:
                z = self.label_map(y)
                z = self.erode_label(z)
                z = self.gmm(z)
                y += (1 - prob) * z
                del z

            # add artifacts
            y = self.addbias(y)
            y = self.mulbias(y)
            y = self.gamma(y)
            y = self.smooth(y)
            y = self.noise(y)
            y = self.rescale(y)

            return y, prob
