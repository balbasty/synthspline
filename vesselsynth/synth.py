import torch
from torch import nn as tnn
from nitorch import nn
from nitorch.core import py, optionals, linalg, utils
from nitorch.spatial._curves import BSplineCurve, draw_curves


def _get_colormap_depth(colormap, n=256, dtype=None, device=None):
    plt = optionals.try_import_as('matplotlib.pyplot')
    mcolors = optionals.try_import_as('matplotlib.colors')
    if colormap is None:
        if not plt:
            raise ImportError('Matplotlib not available')
        colormap = plt.get_cmap('rainbow')
    elif plt and isinstance(colormap, str):
        colormap = plt.get_cmap(colormap)
    if mcolors and isinstance(colormap, mcolors.Colormap):
        colormap = [colormap(i/(n-1))[:3] for i in range(n)]
    else:
        raise ImportError('Matplotlib not available')
    colormap = torch.as_tensor(colormap, dtype=dtype, device=device)
    return colormap


def mip_depth(x, dim=-1, colormap='rainbow'):
    cmap = _get_colormap_depth(colormap, x.shape[dim],
                               dtype=x.dtype, device=x.device)
    x = utils.movedim(x, dim, -1)
    d = linalg.dot(x.unsqueeze(-2), cmap.T)
    d /= x.sum(-1, keepdim=True)
    d *= x.max(-1, keepdim=True).values
    return d


class SynthVesselBlock(tnn.Module):

    def __init__(
            self,
            shape,
            nb_classes=4,
            voxel_size=0.1,  # 100 mu
            tree_density_exp=0.5,  # trees/mm3 (should be 8 according to known_stats)
            tree_density_scale=0.5,
            device=None):
        super().__init__()
        self.simplex = nn.HyperRandomSmoothSimplexMap(shape, nb_classes,
                                                      device=device, fwhm_exp=12)
        self.mixture = nn.HyperRandomGaussianMixture(nb_classes, fwhm='uniform',
                                                     fwhm_exp=12, fwhm_scale=4)
        self.bias = nn.HyperRandomBiasFieldTransform(fwhm_exp=16, sigmoid=True)
        self.vbias = nn.HyperRandomBiasFieldTransform(fwhm_exp=16, sigmoid=True)
        self.noise = nn.HyperRandomChiNoise(sigma_exp=0.01, sigma_scale=0.02)
        self.smooth = nn.RandomSmooth(fwhm_exp=1, fwhm_scale=1)
        self.norm = nn.AffineQuantiles(qmin=0, qmax=1)
        self.norm5 = nn.AffineQuantiles(qmin=0.05, qmax=0.95)
        self.vx = voxel_size
        self.tree_density_exp = tree_density_exp
        self.tree_density_scale = tree_density_scale
        self.tree_density = nn.generators.distribution._LogNormal
        self.tree_density = self.tree_density(tree_density_exp, tree_density_scale)

    def forward(self, batch=1):

        # sample background
        import time
        start = time.time()
        bg = self.simplex(batch).argmax(dim=1, keepdim=True)
        bg = self.mixture(bg)
        bg = self.smooth(bg)
        bg = self.norm5(bg).clamp_(0, 1)
        print('background: ', time.time() - start)
        shape = bg.shape[2:]
        dim = bg.dim() - 2

        # sample vessels
        volume = py.prod(bg.shape[2:]) * (self.vx ** (bg.dim() - 2))
        density = self.tree_density.sample()
        nb_trees = int(volume * density // 1)

        def clamp(x):
            # ensure point is inside FOV
            x = x.clamp_min(0)
            x = torch.min(x, torch.as_tensor(shape, dtype=x.dtype))
            return x

        def length(a, b):
            return (a-b).square().sum().sqrt()

        def linspace(a, b, n):
            vector = (b-a) / (n-1)
            return a + vector * torch.arange(n).unsqueeze(-1)

        start = time.time()
        l0 = (py.prod(shape) ** (1 / dim))  # typical length
        curves = []
        print(nb_trees)
        for n_tree in range(nb_trees):

            # sample initial point and length
            n = 0
            while n < 3:
                a = clamp(torch.randn([dim]) * l0)    # initial point
                l = torch.rand([dim]) * l0            # length
                b = clamp(a + l)                      # end point
                l = length(a, b)                      # true length
                n = (l / 5).ceil().int().item()       # number of discrete points
            t = torch.rand([1]) * 7               # tortuosity
            d = 1.2 + 0.1 * torch.randn([1])    # 1/sqrt(diameter)
            r = 0.5 / (d*d)                       # radius

            waypoints = linspace(a, b, n) + t * torch.randn([n, dim])
            radii = r * torch.randn([n]).mul(0.1).exp()
            curve = BSplineCurve(waypoints.to(bg.device),
                                 radius=radii.to(bg.device))
            curves.append(curve)
        print('sample curves: ', time.time() - start)

        # draw vessels
        start = time.time()
        true_vessels = draw_curves(shape, curves, max_iter=10)[None, None]
        print('draw curves: ', time.time() - start)

        import matplotlib.pyplot as plt
        plt.imshow(mip_depth(true_vessels[0, 0]))
        plt.show()

        start = time.time()
        vessels = self.vbias(true_vessels)
        print(vessels.max(), true_vessels.max())
        # vessels = self.norm(vessels).clamp_(0, 1)
        vessels *= torch.rand([1], device=vessels.device).mul_(2)

        # add to main image
        print(bg.min(), bg.max(), vessels.max())
        bg *= (1 - true_vessels)
        bg.addcmul_(vessels, true_vessels)
        bg = self.bias(bg)
        bg = self.noise(bg)
        bg = self.norm(bg).clamp_(0, 1)
        print('make image: ', time.time() - start)

        import matplotlib.pyplot as plt
        plt.imshow(bg[0, 0, ..., 0])
        plt.show()

        return bg, true_vessels








