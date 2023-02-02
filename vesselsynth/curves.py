import torch
from interpol import grid_pull, grid_push, grid_grad, spline_coeff, identity_grid
import math as pymath


def backend(x):
    return dict(dtype=x.dtype, device=x.device)


def dot(x, y):
    return x.unsqueeze(-2).matmul(y.unsqueeze(-1)).squeeze(-1).squeeze(-1)


class BSplineCurve:
    """A smooth N-D curve parameterized by B-splines"""

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
        waypoints = torch.as_tensor(waypoints)
        if not waypoints.dtype.is_floating_point:
            waypoints = waypoints.to(torch.get_default_dtype())
        self.waypoints = waypoints
        self.order = order
        self.bound = 'dct2'
        self.coeff = spline_coeff(waypoints, interpolation=self.order,
                                  bound=self.bound, dim=0)
        if not isinstance(radius, (int, float)):
            radius = torch.as_tensor(radius, **backend(waypoints))
        self.radius = radius
        if torch.is_tensor(radius):
            self.coeff_radius = spline_coeff(radius, interpolation=self.order,
                                             bound=self.bound, dim=0)

    def to(self, *args, **kwargs):
        self.waypoints = self.waypoints.to(*args, **kwargs)
        self.coeff = self.coeff.to(*args, **kwargs)
        self.radius = self.radius.to(*args, **kwargs)
        self.coeff_radius = self.coeff_radius.to(*args, **kwargs)
        return self

    def cpu(self):
        return self.to('cpu')

    def cuda(self):
        return self.to('cuda')

    def update_waypoints(self):
        """Convert coefficients into waypoints"""
        t = torch.linspace(0, 1, len(self.coeff), **backend(self.coeff))
        p = self.eval_position(t)
        if p.shape == self.waypoints.shape:
            self.waypoints.copy_(p)
        else:
            self.waypoints = p

    def update_radius(self):
        """Convert coefficients into radii"""
        if not hasattr(self, 'coeff_radius'):
            return
        t = torch.linspace(0, 1, len(self.coeff_radius),
                           **backend(self.coeff_radius))
        r = self.eval_radius(t)
        if torch.is_tensor(self.radius) and r.shape == self.radius.shape:
            self.radius.copy_(r)
        else:
            self.radius = r

    def restrict(self, from_shape, to_shape=None):
        """Apply transform == to a restriction of the underlying grid"""
        to_shape = to_shape or [pymath.ceil(s/2) for s in from_shape]
        shifts = [0.5 * (frm / to - 1)
                  for frm, to in zip(from_shape, to_shape)]
        scales = [frm / to for frm, to in zip(from_shape, to_shape)]
        shifts = torch.as_tensor(shifts, **backend(self.waypoints))
        scales = torch.as_tensor(scales, **backend(self.waypoints))
        self.waypoints.sub_(shifts).div_(scales)
        self.coeff.sub_(shifts).div_(scales)
        self.radius.div_(scales.prod().pow_(1/len(scales)))
        self.coeff_radius.div_(scales.prod().pow_(1/len(scales)))

    def prolong(self, from_shape, to_shape=None):
        """Apply transform == to a prolongation of the underlying grid"""
        to_shape = to_shape or [2*s for s in from_shape]
        from_shape, to_shape = to_shape, from_shape
        shifts = [0.5 * (frm / to - 1)
                  for frm, to in zip(from_shape, to_shape)]
        scales = [frm / to for frm, to in zip(from_shape, to_shape)]
        shifts = torch.as_tensor(shifts, **backend(self.waypoints))
        scales = torch.as_tensor(scales, **backend(self.waypoints))
        self.waypoints.mul_(scales).add_(shifts)
        self.coeff.mul_(scales).add_(shifts)
        self.radius.mul_(scales.prod().pow_(1/len(scales)))
        self.coeff_radius.mul_(scales.prod().pow_(1/len(scales)))

    def eval_position(self, t):
        """Evaluate the position at a given (batched) time"""
        # convert (0, 1) to (0, n)
        shape = t.shape
        t = t.flatten()
        t = t.clamp(0, 1) * (len(self.waypoints) - 1)

        # interpolate
        y = self.coeff.T                      # [D, K]
        t = t.unsqueeze(-1)                   # [N, 1]
        x = grid_pull(y, t, interpolation=self.order, bound=self.bound)
        x = x.T                               # [N, D]
        x = x.reshape([*shape, x.shape[-1]])
        return x

    def eval_radius(self, t):
        """Evaluate the radius at a given (batched) time"""
        if not torch.is_tensor(self.radius):
            return self.radius

        # convert (0, 1) to (0, n)
        shape = t.shape
        t = t.flatten()
        t = t.clamp(0, 1) * (len(self.coeff_radius) - 1)

        # interpolate
        y = self.coeff_radius                 # [K]
        t = t.unsqueeze(-1)                   # [N, 1]
        x = grid_pull(y, t, interpolation=self.order, bound=self.bound)
        x = x.reshape(shape)
        return x

    def grad_position(self, t):
        """Gradient of the evaluated position wrt time"""
        # convert (0, 1) to (0, n)
        shape = t.shape
        t = t.flatten()
        t = t.clamp(0, 1) * (len(self.waypoints) - 1)

        # interpolate
        y = self.coeff.T                          # [D, K]
        t = t.unsqueeze(-1)                       # [N, 1]
        g = grid_grad(y, t, interpolation=self.order, bound=self.bound)
        g = g.squeeze(-1).T                       # [N, D]
        g = g.reshape([*shape, g.shape[-1]])
        g *= (len(self.waypoints) - 1)
        return g

    def eval_grad_position(self, t):
        """Evaluate position and its gradient wrt time"""

        # convert (0, 1) to (0, n)
        shape = t.shape
        t = t.flatten()
        t = t.clamp(0, 1) * (len(self.waypoints) - 1)

        # interpolate
        y = self.coeff.T                          # [D, K]
        t = t.unsqueeze(-1)                       # [N, 1]
        x = grid_pull(y, t, interpolation=self.order, bound=self.bound)
        x = x.T                                   # [N, D]
        g = grid_grad(y, t, interpolation=self.order, bound=self.bound)
        g = g.squeeze(-1).T                       # [N, D]

        x = x.reshape([*shape, x.shape[-1]])
        g = g.reshape([*shape, g.shape[-1]])
        g *= (len(self.waypoints) - 1)
        return x, g

    def push_position(self, x, t):
        """Push gradient into the control points
        (= differentiate wrt control points)"""
        t = t.flatten()
        t = t.clamp(0, 1) * (len(self.coeff) - 1)

        x = x.reshape(-1, x.shape[-1]).T  # [D, N]
        t = t.unsqueeze(-1)               # [N, 1]
        y = grid_push(x, t, [len(self.coeff)],
                      bound=self.bound, interpolation=self.order)
        y = y.T                           # [K, D]
        return y

    def push_radius(self, x, t):
        """Push gradient into the radius control points
        (= differentiate wrt radius control points)"""
        t = t.flatten()
        t = t.clamp(0, 1) * (len(self.coeff_radius) - 1)

        x = x.flatten()                   # [N]
        t = t.unsqueeze(-1)               # [N, 1]
        y = grid_push(x, t, [len(self.coeff_radius)],
                      bound=self.bound, interpolation=self.order)
        return y


def min_dist(x, s, method='gn', **kwargs):
    """Compute the minimum distance from a (set of) point(s) to a curve.

    Parameters
    ----------
    x : (*shape, dim) tensor
        Coordinates
    s : BSplineCurve
        Parameterized curve
    method : {'table', 'quad', 'gn}, default='gn'
        If 'table', use a table search.
        If 'quad', use a quadratic optimizer (Brent).
        If 'gn', use a Gauss-Newton optimizer

    Returns
    -------
    t : (*shape) tensor
        Coordinate of the closest point
    d : (*shape) tensor
        Minimum distance between each point and the curve

    References
    ----------
    ..[1]   "Robust and efficient computation of the closest point on a
            spline curve"
            Hongling Wang, Joseph Kearney, Kendall Atkinson
            Proc. 5th International Conference on Curves and Surfaces (2002)
    """
    method = method[0].lower()
    if method == 'g':
        return min_dist_gn(x, s, **kwargs)
    elif method == 'q':
        return min_dist_quad(x, s, **kwargs)
    else:
        return min_dist_table(x, s, **kwargs)


def min_dist_table(x, s, steps=None):
    """Compute the minimum distance from a (set of) point(s) to a curve.

    This function performs a global search across a list of values

    Parameters
    ----------
    x : (*shape, dim) tensor
        Coordinates
    s : BSplineCurve
        Parameterized curve
    steps : int or sequence, default=max(x.shape)
        Number of values evaluated, or explicit list of values to evaluate.

    Returns
    -------
    t : (...) tensor
        Coordinate of the closest point
    d : (...) tensor
        Minimum distance between each point and the curve

    """
    @torch.jit.script
    def dot(x, y):
        return x[..., None, :].matmul(y[..., :,  None])[..., 0, 0]

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
        length = s.waypoints[1:] - s.waypoints[:-1]
        length = dot(length, length).sqrt_().sum()
        steps = max(3, (length / 2).ceil().int().item())
    if isinstance(steps, int):
        all_t = torch.linspace(0, 1, steps, **backend(x))
    else:
        all_t = torch.as_tensor(steps, **backend(x)).flatten()
    all_x = torch.stack([s.eval_position(t1) for t1 in all_t])
    t, d = mind(x, all_x, all_t)

    return t, d.sqrt_()


def min_dist_quad(x, s, max_iter=16, tol=1e-3, step=0.01,
                  init=None, init_kwargs=None):
    """Compute the minimum distance from a (set of) point(s) to a curve.

    This function uses quadratic optimization to estimate the distance map.

    Parameters
    ----------
    x : (*shape, dim) tensor
        Coordinates
    s : BSplineCurve
        Parameterized curve
    max_iter : int, default=2**16
        Maximum number of iterations
    tol : float, default=1e-6
        Tolerance for early stopping
    step : float, default=0.1
        Step around the initial value, used to fit the first quadratic curve.
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
    @torch.jit.script
    def dot(x, y):
        return x[..., None, :].matmul(y[..., :,  None])[..., 0, 0]

    def dist(t):
        d = s.eval_position(t).sub_(x)
        d = dot(d, d)
        return d

    if init is None:
        # initialize using a discrete search
        t, _ = min_dist_table(x, s, **(init_kwargs or {}))
    else:
        t = init.clone()

    brent = Brent(tol=tol, max_iter=max_iter, step=step)
    t, d = brent.fit(t, dist)

    return t, d.sqrt_()


def min_dist_gn(x, s, max_iter=16, tol=1e-3, init=None, init_kwargs=None):
    """Compute the minimum distance from a (set of) point(s) to a curve.

    This function uses Gauss-Newton optimization to estimate the distance map.

    Parameters
    ----------
    x : (*shape, dim) tensor
        Coordinates
    s : BSplineCurve
        Parameterized curve
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
        t, d = min_dist_quad(x, s, **(init_kwargs or {}))
        d = d.square_()
    elif init == 'table':
        # initialize using a discrete search
        t, d = min_dist_table(x, s, **(init_kwargs or {}))
        d = d.square_()
    else:
        t = init.clone()
        d = s.eval_position(t) - x
        d = d.square_().sum(-1)

    # Fine tune using Gauss-Newton optimization
    nll = d.sum(-1)
    # print(f'{0:03d} {nll.sum().item():12.6g}')
    for n_iter in range(1, max_iter+1):
        # compute the distance between x and s(t) + gradients
        d, g = s.eval_grad_position(t)
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
            d = s.eval_position(t).sub_(x)
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

    d = s.eval_position(t).sub_(x)
    d = d.square_().sum(-1).sqrt_()

    return t, d


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


def draw_curve(shape, s, mode='cosine', tiny=0, **kwargs):
    """Draw a BSpline curve

    Parameters
    ----------
    shape : list[int]
    s : BSplineCurve
    mode : {'binary', 'gaussian'}

    Returns
    -------
    x : (*shape) tensor
        Drawn curve

    """
    x = identity_grid(shape, **backend(s.waypoints))
    t, d = min_dist(x, s, **kwargs)
    r = s.eval_radius(t)
    if mode[0].lower() == 'b':
        return d <= r
    else:
        return dist_to_prob(d, r, mode, tiny)


def draw_curves(shape, curves, mode='cosine', fast=0, **kwargs):
    """Draw multiple BSpline curves

    Parameters
    ----------
    shape : list[int]
    s : list[BSplineCurve]
    mode : {'binary', 'gaussian'}
    fast : float, default=0

    Returns
    -------
    x : (*shape) tensor
        Drawn curve
    lab : (*shape) tensor[int]
        Label of closest curve

    """
    mode = mode[0].lower()
    if mode == 'b':
        return draw_curves_binary(shape, curves, fast, **kwargs)
    else:
        return draw_curves_prob(shape, curves, fast, mode, **kwargs)


def draw_curves_prob(shape, curves, fast=0, mode='cosine', **kwargs):
    if fast:
        return draw_curves_prob_fast(shape, curves, fast, mode, **kwargs)

    curves = list(curves)
    locations = identity_grid(shape, **backend(curves[0].waypoints))
    label = locations.new_zeros(shape, dtype=torch.long)
    sum_prob = locations.new_ones(shape)
    max_prob = locations.new_zeros(shape)

    count = 0
    while curves:
        curve = curves.pop(0)
        count += 1
        time, dist = min_dist(locations, curve, **kwargs)
        radius = curve.eval_radius(time)
        prob = dist_to_prob(dist, radius, mode)
        label.masked_fill_(prob > max_prob, count)
        max_prob = torch.maximum(max_prob, prob)
        sum_prob *= prob.neg_().add_(1)  # probability of no vessel
    sum_prob = sum_prob.neg_().add_(1)   # probability of at least one vessel

    return sum_prob, label


def draw_curves_prob_fast(shape, curves, threshold, mode='cosine', **kwargs):
    curves = list(curves)
    locations = identity_grid(shape, **backend(curves[0].waypoints))
    label = locations.new_zeros(shape, dtype=torch.long)
    sum_prob = locations.new_ones(shape)
    max_prob = locations.new_zeros(shape)
    prob = locations.new_zeros(shape)

    count = 0
    ncurves = len(curves)
    while curves:
        curve = curves.pop(0)
        count += 1
        print(f"{count:03d}/{ncurves:03d}", end='\r')

        if threshold is True:
            threshold1 = 10 * max(curve.radius)
        else:
            threshold1 = threshold

        # initialize distance from table and only process
        # points that are close enough from the curve
        time, dist = min_dist_table(locations, curve)
        mask = dist < threshold1
        if mask.any():
            sublocations = locations[mask, :]

            kwargs.setdefault('init_kwargs', {})
            kwargs['init_kwargs']['init'] = time[mask]
            time, dist = min_dist(sublocations, curve, **kwargs)
            radius = curve.eval_radius(time)
            subprob = dist_to_prob(dist, radius, mode)
            prob.zero_()
            prob[mask] = subprob

            label.masked_fill_(prob > max_prob, count)
            max_prob = torch.maximum(max_prob, prob)
            sum_prob *= prob.neg_().add_(1)  # probability of no curve
    sum_prob = sum_prob.neg_().add_(1)   # probability of at least one curve

    print('')
    return sum_prob, label


def draw_curves_binary(shape, curves, fast=0, **kwargs):
    if fast:
        return draw_curves_binary_fast(shape, curves, fast, **kwargs)

    curves = list(curves)
    locations = identity_grid(shape, **backend(curves[0].waypoints))
    label = locations.new_zeros(shape, dtype=torch.long)

    count = 0
    while curves:
        curve = curves.pop(0)
        count += 1
        time, dist = min_dist(locations, curve, **kwargs)
        radius = curve.eval_radius(time)
        is_vessel = dist <= radius
        label.masked_fill_(is_vessel, count)

    return label > 0, label


def draw_curves_binary_fast(shape, curves, threshold, **kwargs):
    curves = list(curves)
    locations = identity_grid(shape, **backend(curves[0].waypoints))
    label = locations.new_zeros(shape, dtype=torch.long)

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
        time, dist = min_dist_table(locations, curve)
        mask = dist < threshold1
        if mask.any():
            sublocations = locations[mask, :]

            kwargs.setdefault('init_kwargs', {})
            kwargs['init_kwargs']['init'] = time[mask]
            time, dist = min_dist(sublocations, curve, **kwargs)
            radius = curve.eval_radius(time)
            is_vessel = dist <= radius
            label[mask] = torch.where(is_vessel, count, label[mask])

    return label > 0, label


def _draw_curves_inv(shape, s, mode='cosine', tiny=0):
    """prod_k (1 - p_k)"""
    s = list(s)
    x = identity_grid(shape, **backend(s[0].waypoints))
    s1 = s.pop(0)
    t, d = min_dist(x, s1)
    r = s1.eval_radius(t)
    c = dist_to_prob(d, r, mode, tiny=tiny).neg_().add_(1)
    while s:
        s1 = s.pop(0)
        t, d = min_dist(x, s1)
        r = s1.eval_radius(t)
        c.mul_(dist_to_prob(d, r, mode, tiny=tiny).neg_().add_(1))
    return c


class Brent:
    """Batched 1D optimizer (derivative-free)"""

    gold = (1 + pymath.sqrt(5))/2
    igold = 1 - (pymath.sqrt(5) - 1)/2
    tiny = 1e-8

    def __init__(self, tol=1e-9, max_iter=128, step=0.1):
        """
        Parameters
        ----------
        tol : float, default=1e-9
            Tolerance for early stopping
        max_iter : int, default=128
            Maximum number of iterations
        step : float, default=0.1
            Initial step size used to compute the initial bracket
        """
        self.tol = tol
        self.max_iter = max_iter
        self.step = step

    def fit(self, init, closure):
        """
        Parameters
        ----------
        init : tensor
            Initial estimate of the minimum
        closure : callable(tensor) -> tensor
            Function that computes the loss

        Returns
        -------
        a : tensor
            Location of the minimum
        f : tensor
            Value of the minimum
        """
        bracket = self.bracket(init, closure(init), closure)
        return self.search_in_bracket(bracket, closure)

    def bracket(self, a0, f0, closure):
        """Bracket the minimum

        Parameters
        ----------
        a0 : Initial parameter
        f0 : Initial value
        closure : callable(a) -> evaluate function at `a`

        Returns
        -------
        a0, a1, a2, f0, f1, f2
            (a1, f1) is the current estimate of the minimum location and value
            a1 is in (a0, a2) or (a2, a0)
            f1 is lower than both f0 and f2
        """
        @torch.jit.script
        def quad_min(a0, a1, a2, f0, f1, f2):
            y01 = a0 * a0 - a1 * a1
            y02 = a0 * a0 - a2 * a2
            y12 = a1 * a1 - a2 * a2
            a01 = a0 - a1
            a02 = a0 - a2
            a12 = a1 - a2
            a = (f0 * y12 + f1 * y02 + f2 * y01) / (f0 * a12 + f1 * a02 + f2 * a01)
            s = f0 / (y01 * y02) + f1 / (y01 * y12) + f2 / (y02 * y12)
            return a / 2, s

        @torch.jit.script
        def pre_closure(a0, a1, a2, f0, f1, f2, gold: float):
            delta0 = a2 - a1
            delta, s = quad_min(a0, a1, a2, f0, f1, f2)
            delta = (delta - a1).minimum((1 + gold) * delta0)
            a = torch.where(s > 0, a1 + delta, a2 + gold * delta0)
            return a

        @torch.jit.script
        def post_closure(a, a0, a1, a2, f, f0, f1, f2):
            # f2 < f1 < f0 so (assuming unicity) the minimum is in
            # (a1, a2) or (a2, inf)
            mask0 = f2 < f1
            mask1 = ((a1 < a) == (a < a2))
            mask2 = (f < f2) & mask1 & mask0
            a0, a1 = torch.where(mask2, a1, a0), torch.where(mask2, a, a1)
            f0, f1 = torch.where(mask2, f1, f0), torch.where(mask2, f, f1)
            mask2 = (f1 < f) & mask1 & mask0
            a2 = torch.where(mask2, a, a2)
            f2 = torch.where(mask2, f, f2)
            # shift by one point
            mask2 = mask0 & ~mask1
            a0, a1, a2 = (torch.where(mask2, a1, a0),
                          torch.where(mask2, a2, a1),
                          torch.where(mask2, a, a2))
            f0, f1, f2 = (torch.where(mask2, f1, f0),
                          torch.where(mask2, f2, f1),
                          torch.where(mask2, f, f2))
            return a0, a1, a2, f0, f1, f2

        a1 = a0 + self.step
        f1 = closure(a1)

        # sort such that f1 < f0
        mask = f1 > f0
        a0, a1 = torch.where(mask, a1, a0), torch.where(mask, a0, a1)
        f0, f1 = torch.where(mask, f1, f0), torch.where(mask, f0, f1)

        a2 = (a1 - a0).mul_(self.gold).add_(a1)
        f2 = closure(a2)

        for n_iter in range(self.max_iter):
            if (f1 < f2).all():
                break

            # fit quadratic polynomial
            a = pre_closure(a0, a1, a2, f0, f1, f2, float(self.gold))

            # evaluate new point
            f = closure(a)

            # check progress and update bracket
            a0, a1, a2, f0, f1, f2 = post_closure(a, a0, a1, a2, f, f0, f1, f2)

        return a0, a1, a2, f0, f1, f2

    def search_in_bracket(self, bracket, closure):
        """

        Parameters
        ----------
        bracket : tuple[tensor] = (a0, a1, a2, f0, f1, f2)
            Estimate (a1, f1) and bracket [(a0, f0), (a2, f2)] returned
            by the `bracket` function.
        closure : callable(a) -> f
            Function that computes the objective function.

        Returns
        -------
        a : tensor
            Location of the minimum
        f : tensor
            Value of the minimum

        """

        @torch.jit.script
        def quad_min(a0, a1, a2, f0, f1, f2):
            """
            Fit a quadratic to three points (a0, f0), (a1, f1), (a2, f2)
            and return the location of its minimum, and its quadratic factor
            """
            a00, a11, a22 = a0 * a0, a1 * a1, a2 * a2
            y01, y02, y12 = a00 - a11, a00 - a22, a11 - a22
            a01, a02, a12 = a0 - a1, a0 - a2, a1 - a2
            a = f0 * y12 + f1 * y02 + f2 * y01
            a = 0.5 * a / (f0 * a12 + f1 * a02 + f2 * a01)
            s = f0 / (y01 * y02) + f1 / (y01 * y12) + f2 / (y02 * y12)
            return a, s

        @torch.jit.script
        def sort(a0, a1, a2, f0, f1, f2):
            """
            Sort the pairs (a0, f0), (a1, f1), (a2, f2) such that
            f0 < f1 < f2
            """
            mask = f2 < f1
            a1, a2 = torch.where(mask, a2, a1), torch.where(mask, a1, a2)
            f1, f2 = torch.where(mask, f2, f1), torch.where(mask, f1, f2)
            mask = f1 < f0
            a0, a1 = torch.where(mask, a1, a0), torch.where(mask, a0, a1)
            f0, f1 = torch.where(mask, f1, f0), torch.where(mask, f0, f1)
            mask = f2 < f1
            a1, a2 = torch.where(mask, a2, a1), torch.where(mask, a1, a2)
            f1, f2 = torch.where(mask, f2, f1), torch.where(mask, f1, f2)
            return a0, a1, a2, f0, f1, f2

        @torch.jit.script
        def get_mask_stop(a0, b0, b1, tol: float):
            """Mask of elements that have converged"""
            return ((a0 - 0.5 * (b0 + b1)).abs() + 0.5 * (b1 - b0)) <= 2 * tol

        @torch.jit.script
        def get_mask_nomin(a0, b0, b1, s, d, d1, tiny):
            """Mask of elements that use bisection rather than interpolation"""
            # do not use extremum of the quadratic fit if:
            # - it is a maximum (s < 0), or
            # - jump is larger than half the last jump, or
            # - new point is too close from brackets
            a = a0 + d
            return (s < 0) | (d.abs() > d1.abs()/2) | ~((b0 + tiny < a) & (a < b1 - tiny))

        @torch.jit.script
        def get_side(a0, b0, b1):
            """Side of the bisection"""
            return a0 > 0.5 * (b0 + b1)

        @torch.jit.script
        def get_tiny(a0, tiny: float):
            return tiny * (1 + 2 * a0.abs())

        @torch.jit.script
        def pre_closure(a0, a1, a2, f0, f1, f2, b0, b1, d1, igold: float, tiny: float):
            # fit quadratic polynomial
            d, s = quad_min(a0, a1, a2, f0, f1, f2)
            d = d - a0

            # if quad has a minimum -> new point = minimum      (interpolation)
            # else                  -> new point = golden ratio (bisection)
            tiny1 = get_tiny(a0, tiny)
            mask_nomin = get_mask_nomin(a0, b0, b1, s, d, d1, tiny1)
            mask_side = get_side(a0, b0, b1)
            bisection = torch.where(mask_side, b0 - a0, b1 - a0) * igold
            d = torch.where(mask_nomin, bisection, d)
            a = d + a0
            return a, d

        @torch.jit.script
        def post_closure(a, a0, a1, a2, f, f0, f1, f2, b0, b1):
            mask = f < f0  # f < f0 < f1 < f2
            mask2 = a < a0
            b0, b1 = (torch.where(mask & ~mask2, a0, b0),
                      torch.where(mask & mask2, a0, b1))
            a0, a1, a2 = (torch.where(mask, a, a0),
                          torch.where(mask, a0, a1),
                          torch.where(mask, a1, a2))
            f0, f1, f2 = (torch.where(mask, f, f0),
                          torch.where(mask, f0, f1),
                          torch.where(mask, f1, f2))

            mask = f0 < f
            b0, b1 = (torch.where(mask & mask2, a, b0),
                      torch.where(mask & ~mask2, a, b1))
            mask = mask & (f < f1)          # f0 < f < f1 < f2
            a1, a2 = torch.where(mask, a, a1), torch.where(mask, a1, a2)
            f1, f2 = torch.where(mask, f, f1), torch.where(mask, f1, f2)
            mask = (f1 < f) & (f < f2)      # f0 < f1 < f < f2
            a2 = torch.where(mask, a, a2)
            f2 = torch.where(mask, f, f2)
            return a0, a1, a2, f0, f1, f2, b0, b1

        b0, b1 = bracket[0], bracket[2]
        mask = b1 < b0
        b0, b1 = torch.where(mask, b1, b0), torch.where(mask, b0, b1)
        a0, a1, a2, f0, f1, f2 = sort(*bracket)  # sort by values

        d = torch.full_like(a0, float('inf'))
        d0 = d.clone()
        for n_iter in range(1, self.max_iter+1):

            mask_stop = get_mask_stop(a0, b0, b1, float(self.tol))
            if mask_stop.all():
                break

            # d1 = delta from two iterations ago
            d1, d0 = d0, d

            # fit quadratic polynomial
            # if quad has a minimum -> new point = minimum      (interpolation)
            # else                  -> new point = golden ratio (bisection)
            a, d = pre_closure(a0, a1, a2, f0, f1, f2, b0, b1, d1,
                               float(self.igold), float(self.tiny))

            # evaluate new point
            f = closure(a)

            # update bracket
            a0, a1, a2, f0, f1, f2, b0, b1 = post_closure(
                a, a0, a1, a2, f, f0, f1, f2, b0, b1)

        return a0, f0


# ======================================================================
# Functions to estimate the length of a cubic spline
# ======================================================================

gauss_legendre_coeff = (
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
    l = 0
    for x, w in gauss_legendre_coeff:
        t = 0.5 * (1 + x)
        l += norm(g(t))
    l *= 0.5
    return l


def to_hermite(c00, c0, c1, c11):
    """Transform a "control point" cubic spline into a cubic Hermite spline"""
    # cubic spline coefficients evaluated at integer coordinates 0 and 1
    # s30, s31, d30, d31 = 4/6, 1/6, 0, +/- 0.5

    f0 = (4 * c0 + c00 + c1) / 6
    f1 = (4 * c1 + c0 + c11) / 6
    g0 = (c00 - c1) / 2
    g1 = (c0 - c11) / 2
    return f0, g0, f1, g1


def length(curve: BSplineCurve):
    """Compute the length of a cubic B-spline"""
    if curve.order != 3:
        raise ValueError("Only order 3 supported")
    if curve.bound != 'dct2':
        raise ValueError("Only bound dct2 supported")
    first = [curve.coeff[0], *curve.coeff[:3]]
    l = cubic_hermite_length(*to_hermite(*first))
    for i in range(len(curve.coeff)-3):
        l += cubic_hermite_length(*to_hermite(*curve.coeff[i:i+4]))
    last = [*curve.coeff[-3:], curve.coeff[-1]]
    l += cubic_hermite_length(*to_hermite(*last))
    return l


def discretize_equidistant(curve: BSplineCurve, delta=1):
    n = int(max(length(curve) / delta, 1) * 128)
    t = torch.linspace(0, 1, n,
                       dtype=curve.coeff.dtype,
                       device=curve.coeff.device)
    pos = curve.eval_position(t)
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