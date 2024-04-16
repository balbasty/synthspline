from torch import nn
import torch
import math as pymath
import random as pyrandom
import cornucopia as cc


class SynthVesselImage(nn.Module):
    """Synthesize an image from vessel labels"""

    def __init__(self):
        super().__init__()
        self.xform = cc.ctx.batch(self.XForm())

    def forward(self, label, centerline=None):
        """

        Parameters
        ----------
        label : (B, 1, *spatial) tensor
            Volume of unique vessel labels (each vessel has a unique id)
        centerline : (B, 1, *spatial) tensor, optional
            Volume of centerlines

        Returns
        -------
        image : (B, 1, *spatial) tensor
            Synthetic image
        label : (B, 1, *spatial) tensor[bool]
            Ground truth labels (binary mask)
        centerline : (B, 1, *spatial) tensor[bool], optional
            Ground truth centerlines (binary mask)

        """
        return self.xform(label, centerline)

    class XForm(cc.Transform):

        def __init__(self):
            super().__init__()

            self.flip = cc.RandomFlipTransform()
            self.mask = cc.RandomSmoothLabelMap(cc.random.Fixed(4), shape=8)
            self.label_map = cc.RandomSmoothLabelMap(16, shape=8)
            self.erode_label = cc.RandomErodeLabelTransform(
                radius=5, new_labels=True
            )
            self.gmm = cc.RandomGaussianMixtureTransform()
            self.gamma = cc.RandomGammaTransform((0, 5))
            self.bias = cc.RandomMulFieldTransform()
            self.smooth = cc.RandomSmoothTransform()
            self.noise = cc.RandomChiNoiseTransform(10)
            self.rescale = cc.QuantileTransform()

        def forward(self, x, c=None):

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
            nb_groups = cc.random.RandInt(1, 5)()
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


class SynthVesselOCTImage(nn.Module):
    """Synthesize an OCT image from vessel labels"""

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
        self.xform = cc.ctx.batch(self.XForm(softlabel, centerline))

    def forward(self, input):
        """

        Parameters
        ----------
        label : (B, 1, *spatial) tensor
            Volume of unique vessel labels (each vessel has a unique id)
        softlabel : (B, 1, *spatial) tensor, optional
            Volume of vessel probabilities
        centerline : (B, 1, *spatial) tensor, optional
            Volume of centerlines

        Returns
        -------
        image : (B, 1, *spatial) tensor
            Synthetic image
        label : (B, 1|2, *spatial) tensor
            Ground truth labels
            If `softlabel`, dtype is float, else it is int
            If `centerline`, the second channel contains the centerline label

        """
        return self.xform(input)

    class XForm(cc.Transform):

        def __init__(self, softlabel=False, centerline=False):
            super().__init__()
            self.softlabel = softlabel
            self.centerline = centerline

            # ground truth
            self.flip = cc.RandomFlipTransform()
            self.mask = cc.RandomSmoothLabelMap(cc.random.Fixed(4), shape=8)

            # background
            self.label_map = cc.RandomSmoothLabelMap(16, shape=8)
            self.erode_label = cc.RandomErodeLabelTransform(
                radius=5, new_labels=True
            )

            # noisy vessels
            self.erodespline = cc.RandomSmoothMorphoLabelTransform(
                shape=128, max_radius=4, min_radius=-4)
            self.shallow = cc.RandomSmoothShallowLabelTransform(
                shape=128, max_width=5) * 0.3
            self.noisylabel = cc.ctx.randomize(
                cc.SmoothBernoulliTransform,
                dict(shape=cc.random.RandInt(2, 128),
                     prob=cc.random.Uniform(0, 0.2)),
            )

            # crappy stuff
            self.balls = cc.ctx.randomize(
                cc.SmoothBernoulliDiskTransform,
                dict(prob=cc.random.Uniform(0.01),
                     radius=cc.random.Uniform(5),
                     value='random')
            )

            # intensity randomization
            self.gmm = cc.RandomGaussianMixtureTransform()
            self.gamma = cc.RandomGammaTransform((0, 5))
            self.bias = cc.RandomFieldTransform()
            self.smooth = cc.RandomSmoothTransform()
            self.noise = cc.RandomGammaNoiseTransform(10)
            self.rescale = cc.QuantileTransform()

        def forward(self, x, p=None, c=None):

            # x: volume of unique vessel labels (each vessel has a unique id)
            # p: volume of vessel probabilities (optional)
            # c: centerline (optional)

            if isinstance(x, (list, tuple)):
                p = c = None
                if self.softlabel and self.centerline:
                    x, p, c = x
                elif self.softlabel:
                    x, p = x
                elif self.centerline:
                    x, c = x

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
            nb_groups = cc.random.RandInt(1, 5)()
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


class SynthVesselPhotoImage(nn.Module):
    """Synthesize a blockface photograph image from vessel labels"""

    def __init__(self):
        super().__init__()
        self.xform = cc.ctx.batch(self.XForm())

    def forward(self, label):
        """

        Parameters
        ----------
        label : (B, 1, *spatial) tensor
            Volume of unique vessel labels (each vessel has a unique id)

        Returns
        -------
        image : (B, 1, *spatial) tensor
            Synthetic image
        label : (B, 1, *spatial) tensor
            Ground truth labels

        """
        return self.xform(label)

    class XForm(cc.Transform):

        def __init__(self):
            super().__init__()

            # ground truth
            self.flip = cc.RandomFlipTransform()
            self.mask = cc.RandomSmoothLabelMap(cc.random.Fixed(4), shape=8)

            # background
            self.label_map = cc.RandomSmoothLabelMap(16, shape=8)
            self.erode_label = cc.RandomErodeLabelTransform(
                radius=5, new_labels=True
            )

            # noisy vessels
            self.erodespline = cc.RandomSmoothMorphoLabelTransform(
                shape=128, max_radius=2, min_radius=-2) * 0.1
            self.shallow = cc.RandomSmoothShallowLabelTransform(
                shape=128, max_width=2) * 0.1
            # self.noisylabel = cc.randomize(cc.SmoothBernoulliTransform)(
            #    shape=cc.random.RandInt(2, 128),
            #    prob=cc.random.Uniform(0, 0.2),
            # )

            # crappy stuff
            self.balls = cc.ctx.randomize(
                cc.SmoothBernoulliDiskTransform,
                dict(prob=cc.random.Uniform(0.01),
                     radius=cc.random.Uniform(5),
                     value='random')
            )

            # intensity randomization
            self.gmm = cc.RandomGaussianMixtureTransform(fwhm=5)
            self.gamma = cc.RandomGammaTransform((0, 5))
            self.smooth = cc.RandomSmoothTransform(fwhm=1) * 0.5
            self.bias = cc.RandomSlicewiseMulFieldTransform(
                thickness=cc.random.Fixed(1),
                slice=-1,
            )
            self.noise = cc.ctx.switch({
                    cc.RandomGammaNoiseTransform(0.02): 0.5,
                    cc.RandomChiNoiseTransform(0.02, nb_channels=2): 0.5,
            })
            self.thick = cc.RandomLowResSliceTransform(
                resolution=cc.random.Uniform(8, 12),
                thickness=cc.random.Uniform(0, 0.5),
                noise=self.bias + self.noise,
                axis=-1,
            )
            self.rescale = cc.QuantileTransform()

        def forward(self, x):

            # x: volume of unique vessel labels (each vessel has a unique id)
            # p: volume of vessel probabilities (optional)
            # c: centerline (optional)

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
            nb_groups = cc.random.RandInt(1, 5)()
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


class SynthAxonImage(nn.Module):
    """Synthesize a LSM image from axon labels"""

    def __init__(self, background=True):
        super().__init__()
        self.xform = cc.ctx.batch(self.XForm(background))

    def forward(self, *args):
        """

        Parameters
        ----------
        hardlabel : (B, 1, *spatial) tensor
            Volume of unique labels (each vessel has a unique id)
        softlabel : (B, 1, *spatial) tensor
            Volume of probability

        Returns
        -------
        image : (B, 1, *spatial) tensor
            Synthetic image
        prob : (B, 1, *spatial) tensor
            Ground truth probability

        """
        out = self.xform(*args)
        return out

    class XForm(cc.Transform):

        def __init__(self, background=True):
            super().__init__()
            self.background = background

            self.flip = cc.RandomFlipTransform()
            self.erode_axon = cc.RandomSmoothMorphoLabelTransform(
                shape=128, max_radius=4, min_radius=-4)
            self.shallow = cc.RandomSmoothShallowLabelTransform(
                shape=128, max_width=3) * 0.3
            self.noisylabel = cc.ctx.randomize(
                cc.SmoothBernoulliTransform,
                dict(shape=cc.random.RandInt(2, 128),
                     prob=cc.random.Uniform(0, 0.2)),
            )
            self.soma = cc.randomize(
                cc.SmoothBernoulliDiskTransform,
                dict(shape=cc.random.RandInt(2, 16),
                     prob=cc.random.Uniform(0, 0.02),
                     radius=cc.random.Fixed(cc.random.Uniform(1, 10)))
            )
            self.label_map = cc.RandomSmoothLabelMap(
                nb_classes=cc.random.RandInt(1, 16),
                shape=cc.random.RandInt(2, 8))
            self.erode_label = cc.RandomErodeLabelTransform(
                radius=5, output_labels='unique')
            self.gmm = cc.RandomGaussianMixtureTransform(
                background=None if self.background else 0)
            self.gamma = cc.RandomGammaTransform(cc.random.Uniform(0, 5))
            self.addbias = cc.RandomAddFieldTransform(
                shape=cc.random.RandInt(2, 8),
                vmax=cc.random.Uniform(0, 64))
            self.mulbias = cc.RandomMulFieldTransform(cc.random.RandInt(2, 8))
            self.smooth = cc.RandomSmoothTransform(cc.random.Uniform(0, 2))
            self.noise = cc.RandomChiNoiseTransform(cc.random.Uniform(0, 10))
            self.rescale = cc.QuantileTransform()

        def forward(self, lab, prob=None):
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

            # create groups of vessels that have the same intensity and give
            # them a unique label
            y = torch.zeros_like(lab, dtype=torch.int)
            vessel_labels = list(sorted(v.unique().tolist()))[1:]
            pyrandom.shuffle(vessel_labels)
            nb_groups = cc.random.RandInt(1, 5)()
            nb_per_group = int(pymath.ceil(len(vessel_labels) / nb_groups))
            for i in range(nb_groups):
                labels = vessel_labels[i*nb_per_group:(i+1)*nb_per_group]
                for label in labels:
                    y.masked_fill_(v == label, i + 1)
                soma = self.soma.get_parameters(y).get_parameters(y)
                y.masked_fill(soma, i + 1)
            # del v

            # generate foreground image
            y = self.gmm(y)
            y *= prob

            # generate background image
            if self.background:
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
