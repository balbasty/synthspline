__all__ = ['BSplineCurve', 'BSplineCurves']
import torch
from interpol import (
    grid_pull, grid_push, grid_grad, spline_coeff, identity_grid)
import math as pymath
from vesselsynth import backend
from vesselsynth.brent import Brent
from vesselsynth.utils import to_tensor


# ======================================================================
# Object representing a single cubic spline, with lots of useful methods
# ======================================================================


class BSplineCurve(torch.nn.Module):
    """
    A smooth N-D curve parameterized by a B-spline

    Attributes
    ----------
    waypoints : (N, D) tensor
        List of waypoints, that the curve interpolates.
    coeff : (N, D) tensor
        Spline coefficients that encode the curve.
    radius : number or (N,) tensor
        Radius of the spline, eventually per node in the curve
    coeff_radius : (N,) tensor, optional
        Spline coefficients that encode the radius.
        Only present if radius is a tensor
    """

    def __init__(self, waypoints, order=3, radius=1):
        """

        Parameters
        ----------
        waypoints : (N, D) tensor
            List of waypoints, that the curve will interpolate.
        order : int, default=3
            Order of the encoding B-splines
        radius : float or (N,) tensor
            Radius of the curve at each waypoint.
        """
        super().__init__()
        self.order = order
        self.bound = 'dct2'
        waypoints = to_tensor(waypoints, dtype=float)
        coeff = spline_coeff(waypoints, interpolation=self.order,
                             bound=self.bound, dim=0)
        self.register_buffer('waypoints', waypoints)
        self.register_buffer('coeff', coeff)
        if not isinstance(radius, (int, float)):
            radius = to_tensor(radius, **tensor_backend(waypoints))
            coeff_radius = spline_coeff(radius, interpolation=self.order,
                                        bound=self.bound, dim=0)
            self.register_buffer('radius', radius)
            self.register_buffer('coeff_radius', coeff_radius)
        else:
            self.radius = radius
            self.coeff_radius = None

    @property
    def _gridopt(self):
        return dict(bound=self.bound, interpolation=self.order)

    @property
    def _backend(self):
        return dict(dtype=self.waypoints.dtype, device=self.waypoints.device)

    def update_waypoints(self):
        """Convert coefficients into waypoints"""
        t = torch.linspace(0, 1, len(self.coeff), **self._backend)
        p = self.eval_position(t)
        if p.shape == self.waypoints.shape:
            self.waypoints.copy_(p)
        else:
            self.waypoints = p

    def update_radius(self):
        """Convert coefficients into radii"""
        if self.coeff_radius is None:
            return
        t = torch.linspace(0, 1, len(self.coeff_radius), **self._backend)
        r = self.eval_radius(t)
        if torch.is_tensor(self.radius) and r.shape == self.radius.shape:
            self.radius.copy_(r)
        else:
            self.radius = r

    def restrict(self, from_shape, to_shape=None):
        """
        Transform the node coordinates and radius to reflect a change
        in resolution of the embedding space (to lower resolution).

        !!! note
            This function can be useful when fitting a curve to an
            observed image. It is not used in synthesis.
        """
        to_shape = to_shape or [pymath.ceil(s/2) for s in from_shape]
        shifts = [0.5 * (frm / to - 1)
                  for frm, to in zip(from_shape, to_shape)]
        scales = [frm / to for frm, to in zip(from_shape, to_shape)]
        shifts = torch.as_tensor(shifts, **self._backend)
        scales = torch.as_tensor(scales, **self._backend)
        self.waypoints.sub_(shifts).div_(scales)
        self.coeff.sub_(shifts).div_(scales)
        self.radius.div_(scales.prod().pow_(1/len(scales)))
        self.coeff_radius.div_(scales.prod().pow_(1/len(scales)))

    def prolong(self, from_shape, to_shape=None):
        """
        Transform the node coordinates and radius to reflect a change
        in resolution of the embedding space (to higher resolution).

        !!! note
            This function can be useful when fitting a curve to an
            observed image. It is not used in synthesis.
        """
        to_shape = to_shape or [2*s for s in from_shape]
        from_shape, to_shape = to_shape, from_shape
        shifts = [0.5 * (frm / to - 1)
                  for frm, to in zip(from_shape, to_shape)]
        scales = [frm / to for frm, to in zip(from_shape, to_shape)]
        shifts = torch.as_tensor(shifts, **self._backend)
        scales = torch.as_tensor(scales, **self._backend)
        self.waypoints.mul_(scales).add_(shifts)
        self.coeff.mul_(scales).add_(shifts)
        self.radius.mul_(scales.prod().pow_(1/len(scales)))
        self.coeff_radius.mul_(scales.prod().pow_(1/len(scales)))

    def eval_position(self, t):
        """
        Evaluate the position at a given (batched) time

        Parameters
        ----------
        t : (*batch) tensor
            Time at which to evaluate the curve

        Returns
        -------
        x : (*batch, D) tensor
            Curve coordinate at that time
        """
        # convert (0, 1) to (0, N-1)
        N, D = self.coeff.shape
        shape = t.shape
        t = t.flatten()
        t = t.clamp(0, 1) * (N - 1)

        # interpolate
        y = self.coeff.T                      # [D, N]
        t = t.unsqueeze(-1)                   # [B, 1]
        x = grid_pull(y, t, **self._gridopt)  # [D, B]
        x = x.T                               # [B, D]
        x = x.reshape([*shape, D])            # [*B, D]
        return x

    def eval_radius(self, t):
        """
        Evaluate the radius at a given (batched) time

        Parameters
        ----------
        t : (*batch) tensor
            Time at which to evaluate the curve

        Returns
        -------
        r : number or (*batch) tensor
            Curve radius at that time
        """
        if not torch.is_tensor(self.radius):
            return self.radius

        # convert (0, 1) to (0, N-1)
        N = len(self.coeff_radius)
        shape = t.shape
        t = t.flatten()
        t = t.clamp(0, 1) * (N - 1)

        # interpolate
        y = self.coeff_radius                 # [D]
        t = t.unsqueeze(-1)                   # [B, 1]
        x = grid_pull(y, t, **self._gridopt)  # [B]
        x = x.reshape(shape)                  # [*B]
        return x

    def grad_position(self, t):
        """
        Gradient of the evaluated position wrt time

        !!! note
            This function can be useful when fitting a curve to an
            observed image. It is not used in synthesis.

        Parameters
        ----------
        t : (*batch) tensor
            Time wrt which to evaluate the gradient

        Returns
        -------
        g : (*batch, D) tensor
            Gradient of the curve with respect to time
        """
        # convert (0, 1) to (0, N-1)
        N, D = self.coeff.shape
        shape = t.shape
        t = t.flatten()
        t = t.clamp(0, 1) * (N - 1)

        # interpolate
        y = self.coeff.T                          # [D, N]
        t = t.unsqueeze(-1)                       # [B, 1]
        g = grid_grad(y, t, **self._gridopt)      # [D, B, 1]
        g = g.squeeze(-1).T                       # [B, D]
        g = g.reshape([*shape, D])                # [*B, D]
        g *= (N - 1)
        return g

    def eval_grad_position(self, t):
        """
        Evaluate position and its gradient wrt time

        !!! note
            This function can be useful when fitting a curve to an
            observed image. It is not used in synthesis.

        Parameters
        ----------
        t : (*batch) tensor
            Time wrt which to evaluate the gradient

        Returns
        -------
        x : (*batch, D) tensor
            Curve coordinate at that time
        g : (*batch, D) tensor
            Gradient of the curve with respect to time
        """
        # convert (0, 1) to (0, N-1)
        N, D = self.coeff.shape
        shape = t.shape
        t = t.flatten()
        t = t.clamp(0, 1) * (len(self.waypoints) - 1)

        # interpolate
        y = self.coeff.T                          # [D, N]
        t = t.unsqueeze(-1)                       # [B, 1]
        x = grid_pull(y, t, **self._gridopt)      # [D, B]
        x = x.T                                   # [B, D]
        g = grid_grad(y, t, **self._gridopt)      # [D, B, 1]
        g = g.squeeze(-1).T                       # [B, D]

        x = x.reshape([*shape, D])                # [*B, D]
        g = g.reshape([*shape, D])                # [*B, D]
        g *= (N - 1)
        return x, g

    def push_position(self, g, t):
        """
        Push gradient into the control points
        (= chain rule when differentiating wrt control points)

        !!! note
            This function can be useful when fitting a curve to an
            observed image. It is not used in synthesis.

        Parameters
        ----------
        g : (*batch, D) tensor
            Gradient of a scalar loss with respect to the curve's
            coordinates evaluated at time `t`.
        t : (*batch) tensor
            Time at which the curve was evaluated.

        Returns
        -------
        g : (N, D) tensor
            Gradient of the scalar loss with respect to the curve's
            control points.
        """
        # convert (0, 1) to (0, N-1)
        N, D = self.coeff.shape
        t = t.flatten()
        t = t.clamp(0, 1) * (N - 1)

        g = g.reshape(-1, D).T                      # [D, B]
        t = t.unsqueeze(-1)                         # [B, 1]
        y = grid_push(g, t, [N], **self._gridopt)   # [D, N]
        y = y.T                                     # [N, D]
        return y

    def push_radius(self, x, t):
        """
        Push gradient into the radius control points
        (= chain rule when differentiating wrt radius control points)

        !!! note
            This function can be useful when fitting a curve to an
            observed image. It is not used in synthesis.

        Parameters
        ----------
        g : (*batch) tensor
            Gradient of a scalar loss with respect to the curve's
            radius evaluated at time `t`.
        t : (*batch) tensor
            Time at which the curve was evaluated.

        Returns
        -------
        g : (N,) tensor
            Gradient of the scalar loss with respect to the radius'
            control points.
        """
        # convert (0, 1) to (0, N-1)
        N = len(self.coeff_radius)
        t = t.flatten()
        t = t.clamp(0, 1) * (N - 1)

        x = x.flatten()                             # [B]
        t = t.unsqueeze(-1)                         # [B, 1]
        y = grid_push(x, t, [N], **self._gridopt)   # [N]
        return y

    def min_dist(self, x, method='gn', **kwargs):
        """Compute the minimum distance from a (set of) point(s) to the curve.

        Parameters
        ----------
        x : (*shape, dim) tensor
            Coordinates
        method : {'table', 'quad', 'gn}, default='gn'
            - If 'table', use a table search.
            - If 'quad', use a quadratic optimizer (Brent).
            - If 'gn', use a Gauss-Newton optimizer

        Returns
        -------
        t : (*shape) tensor
            Coordinate of the closest point
        d : (*shape) tensor
            Minimum distance between each point and the curve

        References
        ----------
            "Robust and efficient computation of the closest point on a
            spline curve"
            Hongling Wang, Joseph Kearney, Kendall Atkinson
            Proc. 5th International Conference on Curves and Surfaces (2002)
        """

        method = method[0].lower()

        if backend.jitfields and not kwargs:
            # use CUDA/C++ implementation
            if method == 'g':
                from jitfields.distance import \
                    spline_distance_gaussnewton as fn
            elif method == 'q':
                from jitfields.distance import \
                    spline_distance_brent as fn
            else:
                from jitfields.distance import \
                    spline_distance_table as fn
            d, t = fn(x, self.coeff, bound=self.bound, order=self.order)
            t /= (self.coeff.shape[-2] - 1)
            return t, d

        # fallback to python implementation
        # (or jitfields with used-defined initialization strategy)
        if method == 'g':
            fn = self.min_dist_gn
        elif method == 'q':
            fn = self.min_dist_quad
        else:
            fn = self.min_dist_table

        return fn(x, **kwargs)

    def min_dist_table(self, x, steps=None):
        """Compute the minimum distance from a (set of) point(s) to a curve.

        This function performs a global search across a list of values

        Parameters
        ----------
        x : (*shape, dim) tensor
            Coordinates
        steps : int or sequence, default=max(x.shape)
            Number of values evaluated, or explicit list of values to evaluate.

        Returns
        -------
        t : (*shape) tensor
            Coordinate of the closest point
        d : (*shape) tensor
            Minimum distance between each point and the curve

        """
        if backend.jitfields:
            # use CUDA/C++ implementation
            from jitfields.distance import spline_distance_table as fn
            opt = dict(steps=steps, order=self.order, bound=self.bound)
            d, t = fn(x, self.coeff, **opt)
            t /= (self.coeff.shape[-2] - 1)
            return t, d

        def mind_fast(x, xref, tref, veryfast=False):
            x = x.unsqueeze(-2)
            d = dot(xref, x).mul_(-2).add_(dot(xref, xref))
            d, t = d.min(-1)
            t = tref[t]
            if not veryfast:
                d += dot(x, x)
            return t, d

        def mind(x, all_x, all_t, chunk=16):
            t = torch.zeros_like(x[..., 0])
            d = torch.full_like(x[..., 0], float('inf'))
            nchunks = int(pymath.ceil(len(all_x) / chunk))
            for c in range(nchunks):
                t1 = all_t[c*chunk:(c+1)*chunk]
                x1 = all_x[c*chunk:(c+1)*chunk]
                t1, d1 = mind_fast(x, x1, t1, veryfast=True)
                mask = d1 < d
                t[mask] = t1[mask]
                d[mask] = d1[mask]
                # t = torch.where(d1 < d, t1, t)
                # d = torch.minimum(d, d1, out=d)
            d += dot(x, x)
            d = d.sqrt_()
            return t, d

        # initialize using a discrete search
        if steps is None:
            length = self.waypoints[1:] - self.waypoints[:-1]
            length = dot(length, length).sqrt_().sum()
            steps = max(3, (length / 2).ceil().int().item())
        if isinstance(steps, int):
            all_t = torch.linspace(0, 1, steps, **tensor_backend(x))
        else:
            all_t = torch.as_tensor(steps, **tensor_backend(x)).flatten()
        all_x = torch.stack([self.eval_position(t1) for t1 in all_t])
        t, d = mind(x, all_x, all_t)

        return t, d.sqrt_()

    def min_dist_quad(self, x, max_iter=16, tol=1e-3, step=0.01,
                      init=None, init_kwargs=None):
        """Compute the minimum distance from a (set of) point(s) to a curve.

        This function uses quadratic optimization to estimate the distance map.

        Parameters
        ----------
        x : (*shape, dim) tensor
            Coordinates
        max_iter : int, default=2**16
            Maximum number of iterations
        tol : float, default=1e-6
            Tolerance for early stopping
        step : float, default=0.1
            Step around the initial value, used to fit the first
            quadratic curve.
        init : (*shape) tensor or 'table', default='table'
            Initial guess for `t`.
            If 'table', initialize using a table search.

        Returns
        -------
        t : (*shape) tensor
            Coordinate of the closest point
        d : (*shape) tensor
            Minimum distance between each point and the curve

        """

        def dist(t):
            d = self.eval_position(t).sub_(x)
            d = dot(d, d)
            return d

        if init is None:
            # initialize using a discrete search
            t, _ = self.min_dist_table(x, **(init_kwargs or {}))
        else:
            t = init.clone()

        if backend.jitfields:
            # use CUDA/C++ implementation
            from jitfields.distance import spline_distance_brent_ as fn
            opt = dict(max_iter=max_iter, tol=tol,
                       order=self.order, bound=self.bound)
            d, t = fn(dist(t), t, x, self.coeff, **opt)
            t /= (self.coeff.shape[-2] - 1)
            return t, d

        brent = Brent(tol=tol, max_iter=max_iter, step=step)
        t, d = brent.fit(t, dist)

        return t, d.sqrt_()

    def min_dist_gn(self, x, max_iter=16, tol=1e-3,
                    init=None, init_kwargs=None):
        """Compute the minimum distance from a (set of) point(s) to a curve.

        This function uses Gauss-Newton optimization to estimate the
        distance map.

        Parameters
        ----------
        x : (*shape, dim) tensor
            Coordinates
        max_iter : int, default=2**16
            Maximum number of iterations
        tol : float, default=1e-6
            Tolerance for early stopping
        init : (*shape) tensor or {'table', 'quad'}, default='table'
            Initial guess for `t`.
            If 'table', initialize using a table search.
            If 'quad', initialize using a quadratic optimizer.

        Returns
        -------
        t : (*shape) tensor
            Coordinate of the closest point
        d : (*shape) tensor
            Minimum distance between each point and the curve

        """
        if init is None or init == 'quad':
            # initialize using a quadratic search
            t, d = self.min_dist_quad(x, **(init_kwargs or {}))
            d = d.square_()
        elif init == 'table':
            # initialize using a discrete search
            t, d = self.min_dist_table(x,  **(init_kwargs or {}))
            d = d.square_()
        else:
            t = init.clone()
            d = self.eval_position(t) - x
            d = d.square_().sum(-1)

        if backend.jitfields:
            # use CUDA/C++ implementation
            from jitfields.distance import spline_distance_gaussnewton_ as fn
            opt = dict(max_iter=max_iter, tol=tol,
                       order=self.order, bound=self.bound)
            d, t = fn(d, t, x, self.coeff, **opt)
            t /= (self.coeff.shape[-2] - 1)
            return t, d

        # Fine tune using Gauss-Newton optimization
        nll = d.sum(-1)
        # print(f'{0:03d} {nll.sum().item():12.6g}')
        for n_iter in range(1, max_iter+1):
            # compute the distance between x and s(t) + gradients
            d, g = self.eval_grad_position(t)
            d.sub_(x)
            h = dot(g, g)
            g = dot(g, d)
            h.add_(1e-3)
            g.div_(h)

            # Perform GN step (with line search)
            t0 = t.clone()
            nll0 = nll
            armijo = 1
            success = torch.zeros_like(t, dtype=torch.bool)
            for n_ls in range(12):
                t = torch.where(success, t, t0 - armijo * g).clamp_(0, 1)
                d = self.eval_position(t).sub_(x)
                nll = d.square().sum(-1)
                success = success.logical_or_(nll < nll0)
                if success.all():
                    break
                armijo /= 2
            t = torch.where(success, t, t0)
            if not success.any():
                break

            # print(f'{n_iter:03d} '
            #       f'{nll.sum().item()/nll.numel():12.6g} '
            #       f'{(nll0 - nll).sum().item()/t.numel():6.3g} ')
            if (nll0 - nll).sum() < tol * t.numel():
                break

        d = self.eval_position(t).sub_(x)
        d = d.square_().sum(-1).sqrt_()

        return t, d

    def rasterize(self, shape, mode='cosine', tiny=0, **kwargs):
        """Draw a BSpline curve

        Parameters
        ----------
        shape : list[int]
            Shape of the grid on which to rasterize
        mode : {'binary', 'gaussian', 'cosine'}
            - If 'binary', return a binary map where voxels that are
              within a curve are `True`
            - If 'gaussian', assume that a Gaussian distribution is moved
              along the centerline, with its full-width at half-maximum
              equal to the spline diameter. The resulting soft map has
              value one one the centerline and 0.5 on the tube boundary,
              with a quadratic decay.
            - If 'cosine', return a pseudo "partial volume" map, with
              value 0.5 on the tube boundary, 1 in all voxels that are
              at least one voxel inside the tube and zero in all voxels
              that are at least one voxel outside of the tube, with a
              cosine decay.

        Returns
        -------
        img : (*shape) tensor
            Rasterized curve
        """
        x = identity_grid(shape, **tensor_backend(self.waypoints))
        t, d = self.min_dist(x, **kwargs)
        r = self.eval_radius(t)
        if mode[0].lower() == 'b':
            return d <= r
        else:
            return dist_to_prob(d, r, mode, tiny)

    def length(self):
        """Compute the analytical length of a cubic B-spline"""
        if self.order != 3:
            raise ValueError("Only order 3 supported")
        if self.bound != 'dct2':
            raise ValueError("Only bound dct2 supported")
        first = [self.coeff[0], *self.coeff[:3]]
        length = cubic_hermite_length(*to_hermite(*first))
        for i in range(len(self.coeff)-3):
            length += cubic_hermite_length(*to_hermite(*self.coeff[i:i+4]))
        last = [*self.coeff[-3:], self.coeff[-1]]
        length += cubic_hermite_length(*to_hermite(*last))
        return length

    def evaluate_equidistant(self, delta=1):
        """
        Evaluate the curve at equidistant locations in the Euclidean sense

        Note that this is different from evaluating the curve at
        equidistant time points!

        Parameters
        ----------
        delta : float
            Distance between evaluated points

        Returns
        -------
        x : (K,) tensor
            Evaluated coordinates
        """
        n = int(max(self.length() / delta, 1) * 128)
        t = torch.linspace(0, 1, n, **self._backend)
        pos = self.eval_position(t)
        dist = (pos[1:] - pos[:-1]).square().sum(-1).sqrt()
        dist = torch.cat([torch.zeros_like(dist[:1]), dist])
        dist = torch.cumsum(dist, dim=0)

        d = 0
        pos_final = []
        while d <= dist[-1]:
            i = (dist - d).square().argmin()
            pos_final += [pos[i]]
            d += delta
        return torch.stack(pos_final)


# ======================================================================
# Helpers to estimate the length of a cubic spline
# ======================================================================


GAUSS_LEGENDRE_COEFF = (
    (0, 0.5688889),
    (-0.5384693, 0.4782867),
    (0.5384693, 0.47862867),
    (-0.90617985, 0.23692688),
    (0.90617985, 0.23692688)
)


def cubic_hermite_length(f0, g0, f1, g1):
    """
    Compute length of a cubic Hermite spline by 5-point
    Gauss-Legendre quadrature
    """
    # https://medium.com/@all2one/how-to-compute-the-length-of-a-spline-e44f5f04c40
    # https://en.wikipedia.org/wiki/Gaussian_quadrature

    # evaluate spline derivative
    c0 = g0
    c1 = 6 * (f1 - f0) - 4 * g0 - 2 * g1
    c2 = 6 * (f0 - f1) + 3 * (g0 + g1)
    def g(t): return c0 + t * (c1 + t * c2)

    # evaluate vector norm
    def norm(v): return v.square().sum().sqrt()

    # integrate (note the change of coordinates (-1, 1) -> (0, 1))
    length = 0
    for x, w in GAUSS_LEGENDRE_COEFF:
        t = 0.5 * (1 + x)
        length += norm(g(t))
    length *= 0.5
    return length


def to_hermite(c00, c0, c1, c11):
    """Transform a "control point" cubic spline into a cubic Hermite spline"""
    # cubic spline coefficients evaluated at integer coordinates 0 and 1
    # s30, s31, d30, d31 = 4/6, 1/6, 0, +/- 0.5

    f0 = (4 * c0 + c00 + c1) / 6
    f1 = (4 * c1 + c0 + c11) / 6
    g0 = (c00 - c1) / 2
    g1 = (c0 - c11) / 2
    return f0, g0, f1, g1


# ======================================================================
# Rasterization helpers
# ======================================================================


def dist_to_prob(d, r, mode='gaussian', tiny=0):
    """Transform distances to probabilities

    Parameters
    ----------
    d : tensor
        Distance map
    r : tensor
        Radius map
    mode : {'gaussian', 'cosine'}
        Function that maps distances to (0, 1) probabilities
    tiny : float, default=0
        Ensure that probabilities are in (tiny, 1 - tiny)

    Returns
    -------
    p : tensor
        Probability map

    """
    if torch.is_tensor(d):
        d = d.clone()
    if torch.is_tensor(r):
        r = r.clone()
    return dist_to_prob_(d, r, mode, tiny)


def dist_to_prob_(d, r, mode='gaussian', tiny=0):
    """Transform distances to probabilities (in-place). See `dist_to_prob`."""
    mode = mode.lower()
    d = torch.as_tensor(d)
    if mode[0] == 'g':
        r = radius_to_prec_(r)
        d.square_().mul_(r).mul_(-0.5).exp_()
    elif mode[0] == 'c':
        mask = (d - r).abs() < 0.5
        blend = (d - r).add_(0.5).mul_(pymath.pi/2).cos_()
        d = torch.where(mask, blend, (d < r).to(blend))
    else:
        raise ValueError('Unknown model', mode)
    if tiny:
        d.mul_(1-2*tiny).add_(tiny)
    return d


def radius_to_prec(r):
    """Transform radius into Gaussian precision"""
    if torch.is_tensor(r):
        r = r.clone()
    return radius_to_prec_(r)


def radius_to_prec_(r):
    """Transform radius into Gaussian precision (in-place)"""
    r *= 2            # diameter
    r /= 2.355        # standard deviation
    if torch.is_tensor(r):
        r.square_()       # variance
        r.reciprocal_()   # precision
    else:
        r = 1 / (r*r)
    return r


# ======================================================================
# Basic helpers
# ======================================================================

def tensor_backend(x):
    return dict(dtype=x.dtype, device=x.device)


def dot(x, y):
    """Dot product along the last dimension"""
    return x.unsqueeze(-2).matmul(y.unsqueeze(-1)).squeeze(-1).squeeze(-1)


# ======================================================================
# Object representing a collection of cubic spline
# The only useful method is `rasterize`.
# ======================================================================


class BSplineCurves(torch.nn.ModuleList):
    """A collection of curves encoded by B-splines"""

    def rasterize(self, shape, mode='cosine', fast=0, **opt):
        """Rasterize curves on a grid

        Parameters
        ----------
        shape : list[int]
            Shape of the grid on which to rasterize
        mode : {'binary', 'gaussian', 'cosine'}
            - If 'binary', return a binary map where voxels that are
              within a curve are `True`
            - If 'gaussian', assume that a Gaussian distribution is moved
              along the centerline, with its full-width at half-maximum
              equal to the spline diameter. The resulting soft map has
              value one one the centerline and 0.5 on the tube boundary,
              with a quadratic decay.
            - If 'cosine', return a pseudo "partial volume" map, with
              value 0.5 on the tube boundary, 1 in all voxels that are
              at least one voxel inside the tube and zero in all voxels
              that are at least one voxel outside of the tube, with a
              cosine decay.
        fast : float, default=0
            Only fine-tune the distance map on voxels that are expected to
            be at least this close to the curve according to the rough
            dictionary-based distance. This may or may not save time.

        Returns
        -------
        image : (*shape) tensor
            Rasterized curve
        lab : (*shape) tensor[int]
            Label of closest curve to each voxel
        dist : (*shape) tensor
            Distance map to closest centerline

        """
        mode = mode[0].lower()
        if mode == 'b':
            return self._rasterize_binary(shape, fast, **opt)
        else:
            return self._rasterize_prob(shape, fast, mode, **opt)

    def _rasterize_prob(self, shape, fast=0, mode='cosine', **opt):
        if fast:
            return self._rasterize_prob_fast(shape, fast, mode, **opt)

        curves = list(self)
        locations = identity_grid(shape, **tensor_backend(curves[0].waypoints))
        label = locations.new_zeros(shape, dtype=torch.long)
        sum_prob = locations.new_ones(shape)
        max_prob = locations.new_zeros(shape)
        dist = locations.new_full(shape, float('inf'))

        count = 0
        ncurves = len(curves)
        while curves:
            curve = curves.pop(0)
            count += 1
            print(f"Rasterizing curve {count:03d}/{ncurves:03d}", end='\r')
            time, dist1 = curve.min_dist(locations, **opt)
            dist = torch.minimum(dist, dist1)
            radius = curve.eval_radius(time)
            prob = dist_to_prob(dist1, radius, mode)
            label.masked_fill_(prob > max_prob, count)
            max_prob = torch.maximum(max_prob, prob)
            sum_prob *= prob.neg_().add_(1)  # probability no curve
        sum_prob = sum_prob.neg_().add_(1)   # probability at least one curve

        return sum_prob, label, dist

    def _rasterize_prob_fast(self, shape, threshold, mode='cosine', **opt):
        curves = list(self)
        locations = identity_grid(shape, **tensor_backend(curves[0].waypoints))
        label = locations.new_zeros(shape, dtype=torch.long)
        sum_prob = locations.new_ones(shape)
        max_prob = locations.new_zeros(shape)
        prob = locations.new_zeros(shape)
        dist = locations.new_full(shape, float('inf'))

        count = 0
        ncurves = len(curves)
        while curves:
            curve = curves.pop(0)
            count += 1
            print(f"Rasterizing curve {count:03d}/{ncurves:03d}", end='\r')

            if threshold is True:
                threshold1 = 10 * max(curve.radius)
            else:
                threshold1 = threshold

            # initialize distance from table and only process
            # points that are close enough from the curve
            time, dist1 = curve.min_dist_table(locations)
            mask = dist1 < threshold1
            if mask.any():
                sublocations = locations[mask, :]

                opt.setdefault('init_kwargs', {})
                opt['init_kwargs']['init'] = time[mask]
                time, dist1 = curve.min_dist(sublocations, **opt)
                radius = curve.eval_radius(time)
                subprob = dist_to_prob(dist1, radius, mode)
                prob.zero_()
                prob[mask] = subprob
                dist[mask] = torch.minimum(dist[mask], dist1)

                label.masked_fill_(prob > max_prob, count)
                max_prob = torch.maximum(max_prob, prob)
                sum_prob *= prob.neg_().add_(1)  # probability no curve
        sum_prob = sum_prob.neg_().add_(1)   # probability at least one curve

        print('')
        return sum_prob, label, dist

    def _rasterize_binary(self, shape, fast=0, **kwargs):
        if fast:
            return self._rasterize_binary_fast(shape, fast, **kwargs)

        curves = list(self)
        locations = identity_grid(shape, **tensor_backend(curves[0].waypoints))
        label = locations.new_zeros(shape, dtype=torch.long)
        dist = locations.new_full(shape, float('inf'))

        count = 0
        while curves:
            curve = curves.pop(0)
            count += 1
            time, dist1 = curve.min_dist(locations, **kwargs)
            dist = torch.minimum(dist, dist1)
            radius = curve.eval_radius(time)
            is_vessel = dist1 <= radius
            label.masked_fill_(is_vessel, count)

        return label > 0, label, dist

    def _rasterize_binary_fast(self, shape, threshold, **kwargs):
        curves = list(self)
        locations = identity_grid(shape, **tensor_backend(curves[0].waypoints))
        label = locations.new_zeros(shape, dtype=torch.long)
        dist = locations.new_full(shape, float('inf'))

        count = label.new_zeros([])
        while curves:
            curve = curves.pop(0)
            count += 1

            if threshold is True:
                threshold1 = 10 * max(curve.radius)
            else:
                threshold1 = threshold

            # initialize distance from table and only process
            # points that are close enough from the curve
            time, dist1 = curve.min_dist_table(locations)
            mask = dist1 < threshold1
            if mask.any():
                sublocations = locations[mask, :]

                kwargs.setdefault('init_kwargs', {})
                kwargs['init_kwargs']['init'] = time[mask]
                time, dist1 = curve.min_dist(sublocations, **kwargs)
                radius = curve.eval_radius(time)
                is_vessel = dist1 <= radius
                label[mask] = torch.where(is_vessel, count, label[mask])
                dist[mask] = torch.minimum(dist[mask], dist1)

        return label > 0, label, dist

    def _rasterize_inv(self, shape, mode='cosine', tiny=0):
        # prod_k (1 - p_k)
        s = list(self)
        x = identity_grid(shape, **tensor_backend(s[0].waypoints))
        s1 = s.pop(0)
        t, d = s1.min_dist(x)
        r = s1.eval_radius(t)
        c = dist_to_prob(d, r, mode, tiny=tiny).neg_().add_(1)
        while s:
            s1 = s.pop(0)
            t, d = s1.min_dist(x)
            r = s1.eval_radius(t)
            c.mul_(dist_to_prob(d, r, mode, tiny=tiny).neg_().add_(1))
        return c
