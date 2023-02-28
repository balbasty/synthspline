import math as pymath
import torch


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
            a = bracket_pre_closure(a0, a1, a2, f0, f1, f2, float(self.gold))

            # evaluate new point
            f = closure(a)

            # check progress and update bracket
            a0, a1, a2, f0, f1, f2 = bracket_post_closure(a, a0, a1, a2, f, f0, f1, f2)

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

        b0, b1 = bracket[0], bracket[2]
        mask = b1 < b0
        b0, b1 = torch.where(mask, b1, b0), torch.where(mask, b0, b1)
        a0, a1, a2, f0, f1, f2 = search_sort(*bracket)  # sort by values

        d = torch.full_like(a0, float('inf'))
        d0 = d.clone()
        for n_iter in range(1, self.max_iter+1):

            mask_stop = search_get_mask_stop(a0, b0, b1, float(self.tol))
            if mask_stop.all():
                break

            # d1 = delta from two iterations ago
            d1, d0 = d0, d

            # fit quadratic polynomial
            # if quad has a minimum -> new point = minimum      (interpolation)
            # else                  -> new point = golden ratio (bisection)
            a, d = search_pre_closure(a0, a1, a2, f0, f1, f2, b0, b1, d1,
                                      float(self.igold), float(self.tiny))

            # evaluate new point
            f = closure(a)

            # update bracket
            a0, a1, a2, f0, f1, f2, b0, b1 = search_post_closure(
                a, a0, a1, a2, f, f0, f1, f2, b0, b1)

        return a0, f0


@torch.jit.script
def bracket_quad_min(a0, a1, a2, f0, f1, f2):
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
def bracket_pre_closure(a0, a1, a2, f0, f1, f2, gold: float):
    delta0 = a2 - a1
    delta, s = bracket_quad_min(a0, a1, a2, f0, f1, f2)
    delta = (delta - a1).minimum((1 + gold) * delta0)
    a = torch.where(s > 0, a1 + delta, a2 + gold * delta0)
    return a


@torch.jit.script
def bracket_post_closure(a, a0, a1, a2, f, f0, f1, f2):
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


@torch.jit.script
def search_quad_min(a0, a1, a2, f0, f1, f2):
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
def search_sort(a0, a1, a2, f0, f1, f2):
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
def search_get_mask_stop(a0, b0, b1, tol: float):
    """Mask of elements that have converged"""
    return ((a0 - 0.5 * (b0 + b1)).abs() + 0.5 * (b1 - b0)) <= 2 * tol


@torch.jit.script
def search_get_mask_nomin(a0, b0, b1, s, d, d1, tiny):
    """Mask of elements that use bisection rather than interpolation"""
    # do not use extremum of the quadratic fit if:
    # - it is a maximum (s < 0), or
    # - jump is larger than half the last jump, or
    # - new point is too close from brackets
    a = a0 + d
    return (s < 0) | (d.abs() > d1.abs() / 2) | ~((b0 + tiny < a) & (a < b1 - tiny))


@torch.jit.script
def search_get_side(a0, b0, b1):
    """Side of the bisection"""
    return a0 > 0.5 * (b0 + b1)


@torch.jit.script
def search_get_tiny(a0, tiny: float):
    return tiny * (1 + 2 * a0.abs())


@torch.jit.script
def search_pre_closure(a0, a1, a2, f0, f1, f2, b0, b1, d1, igold: float, tiny: float):
    # fit quadratic polynomial
    d, s = search_quad_min(a0, a1, a2, f0, f1, f2)
    d = d - a0

    # if quad has a minimum -> new point = minimum      (interpolation)
    # else                  -> new point = golden ratio (bisection)
    tiny1 = search_get_tiny(a0, tiny)
    mask_nomin = search_get_mask_nomin(a0, b0, b1, s, d, d1, tiny1)
    mask_side = search_get_side(a0, b0, b1)
    bisection = torch.where(mask_side, b0 - a0, b1 - a0) * igold
    d = torch.where(mask_nomin, bisection, d)
    a = d + a0
    return a, d


@torch.jit.script
def search_post_closure(a, a0, a1, a2, f, f0, f1, f2, b0, b1):
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
    mask = mask & (f < f1)  # f0 < f < f1 < f2
    a1, a2 = torch.where(mask, a, a1), torch.where(mask, a1, a2)
    f1, f2 = torch.where(mask, f, f1), torch.where(mask, f1, f2)
    mask = (f1 < f) & (f < f2)  # f0 < f1 < f < f2
    a2 = torch.where(mask, a, a2)
    f2 = torch.where(mask, f, f2)
    return a0, a1, a2, f0, f1, f2, b0, b1

