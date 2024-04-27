"""
This file implement functions that perform symbolic calculus on
(piecewise) polynomials.

A polynomial is represented as a list of factors, from the lowest
exponent (0) to the highest exponent (n)

A piecewise polynomial is represented as a a list of 2-tuples, where
the left element of each tuple is a polyonmial, and the right element
is the upper bound of the domain on which the polynomial is defined,
with `True` meaning infinity.
"""
import torch
from fractions import Fraction
from .utils import make_list


def poly_eq(p, q):
    """Check that two polynomials are equal"""
    n = max(len(p), len(q))
    p = make_list(p, n, default=0)
    q = make_list(q, n, default=0)
    return all(pi == qj for pi, qj in zip(p, q))


def poly_prod(p, q):
    """Product of two polynomials"""
    new_poly = [0] * (len(p) + len(q) - 1)
    for i, pi in enumerate(p):
        for j, qj in enumerate(q):
            new_poly[i+j] = new_poly[i+j] + pi * qj
    return new_poly


def poly_sum(p, q):
    """Sum of two polynomials"""
    new_poly = [0] * max(len(p), len(q))
    for i, fi in enumerate(p):
        new_poly[i] = new_poly[i] + fi
    for j, gj in enumerate(q):
        new_poly[j] = new_poly[j] + gj
    return new_poly


def poly_eval(p, x):
    """Evaluate P(x)"""
    value, xn = p[0], x
    for v in p[1:]:
        value = value + v * xn
        xn = xn * x
    return value


def poly_diff(p, n=1):
    """Differentiate a polynomial: P'(x)"""
    if n == 0:
        return p
    if n > 1:
        for _ in range(n):
            p = poly_diff(p)
        return p
    if len(p) == 1:
        return [0]
    assert n == 1
    new_poly = [
        (k+1) * v for k, v in enumerate(p[1:])
    ]
    return new_poly


def poly_integral(p, minlim=-float('inf'), maxlim=float('inf')):
    r"""Integrate a polynomial: \int_a^b P(x) dx"""
    if minlim > maxlim:
        return -poly_integral(p, maxlim, minlim)
    elif minlim == maxlim:
        return 0
    q = [0] + [v/(k+1) if torch.is_tensor(v) else Fraction(v, k+1)
               for k, v in enumerate(p)]
    # drop trailing zeros
    while q and q[-1] == 0:
        q = q[:-1]
    if not q:
        return 0
    leading_sign = -1 if q[-1] < 0 else 1
    # special cases: infinite domains
    if minlim == -float('inf') and maxlim == float('inf'):
        return 0 if len(q) % 2 else float('inf') * leading_sign
    elif maxlim == float('inf'):
        return float('inf') * leading_sign
    elif minlim == -float('inf'):
        return float('inf') * leading_sign * (-1 if len(q) % 2 else 1)
    # finite domain
    return poly_eval(q, maxlim) - poly_eval(q, minlim)


def poly_outershift(p, delta):
    """P(x) + delta"""
    g = list(p)
    g[0] = g[0] + delta
    return g


def poly_outerscale(p, alpha):
    """P(x) * alpha"""
    return [v * alpha for v in p]


def poly_innershift(p, delta):
    """P(x + delta)"""
    q = p
    while True:
        if len(p) == 1 and p[0] == 0:
            break
        p = poly_outerscale(poly_diff(p), delta)
        q = poly_sum(q, p)
    return q


def poly_innerscale(p, alpha):
    """P(x * alpha)"""
    return [v * (alpha**k) for k, v in enumerate(p)]


def piecewise_poly_prod(p, q):
    """P(x)  * Q(x) """
    return piecewise_poly_op(p, q, poly_prod)


def piecewise_poly_sum(p, q):
    """P(x)  + Q(x) """
    return piecewise_poly_op(p, q, poly_sum)


def piecewise_poly_outershift(p, delta):
    """P(x) + delta"""
    return [(poly_outershift(pi, delta), ci) for pi, ci in p]


def piecewise_poly_outerscale(p, alpha):
    """P(x) * alpha"""
    return [(poly_outerscale(pi, alpha), ci) for pi, ci in p]


def piecewise_poly_innershift(p, delta):
    """P(x + delta)"""
    return [(poly_innershift(pi, delta), True if ci is True else ci + delta)
            for pi, ci in p]


def piecewise_poly_innerscale(p, alpha):
    """P(x * alpha)"""
    return [(poly_innerscale(pi, alpha), True if ci is True else ci * alpha)
            for pi, ci in p]


def piecewise_poly_integral(p, minlim=-float('inf'), maxlim=float('inf')):
    """Integrate P"""
    if minlim > maxlim:
        return -poly_integral(p, maxlim, minlim)
    elif minlim == maxlim:
        return 0
    if minlim > -float('inf') and maxlim < float('inf'):
        p = piecewise_poly_prod(p, [([0], minlim), ([1], maxlim), ([0], True)])
    elif minlim > -float('inf'):
        p = piecewise_poly_prod(p, [([0], minlim), ([1], True)])
    elif maxlim < float('inf'):
        p = piecewise_poly_prod(p, [([1], maxlim), ([0], True)])
    value = 0
    minlim = -float('inf')
    for pi, maxlim in p:
        if maxlim is True:
            maxlim = float('inf')
        value1 = poly_integral(pi, minlim, maxlim)
        value += value1
        minlim = maxlim
    return value


def piecewise_poly_op(f, g, op):
    """
    Pointwise operation between two piecewise polynomials,
    whose conditions are of the form `x < number`.
    """
    fargs, gargs = list(f), list(g)
    args = []
    while True:
        if fargs and not isinstance(fargs[0][1], bool):
            farg = (fargs[0][0], fargs[0][1])
            flim = float(farg[1])
            if gargs and not isinstance(gargs[0][1], bool):
                garg = (gargs[0][0], gargs[0][1])
                glim = float(garg[1])
                if flim < glim:
                    arg = (op(farg[0], garg[0]), farg[1])
                    if args and poly_eq(arg[0], args[-1][0]):
                        args.pop(-1)
                    args.append(arg)
                    fargs.pop(0)
                    continue
                if glim < flim:
                    arg = (op(farg[0], garg[0]), garg[1])
                    if args and poly_eq(arg[0], args[-1][0]):
                        args.pop(-1)
                    args.append(arg)
                    gargs.pop(0)
                    continue
                assert flim == glim
                arg = (op(farg[0], garg[0]), farg[1])
                if args and poly_eq(arg[0], args[-1][0]):
                    args.pop(-1)
                args.append(arg)
                fargs.pop(0)
                gargs.pop(0)
                continue
            elif gargs:
                arg = (op(farg[0], gargs[0][0]), farg[1])
                if args and poly_eq(arg[0], args[-1][0]):
                    args.pop(-1)
                args.append(arg)
                fargs.pop(0)
                continue
            else:
                if args and poly_eq(farg[0], args[-1][0]):
                    args.pop(-1)
                args.append(farg)
                fargs.pop(0)
                continue
        elif gargs and not isinstance(gargs[0][1], bool):
            garg = (gargs[0][0], gargs[0][1])
            if fargs:
                arg = (op(fargs[0][0], garg[0]), garg[1])
                if args and poly_eq(arg[0], args[-1][0]):
                    args.pop(-1)
                args.append(arg)
                gargs.pop(0)
                continue
            else:
                if args and poly_eq(garg[0], args[-1][0]):
                    args.pop(-1)
                args.append(garg)
                gargs.pop(0)
                continue
        elif fargs and gargs:
            arg = (op(fargs[0][0], gargs[0][0]), fargs[0][1])
            if args and poly_eq(arg[0], args[-1][0]):
                args.pop(-1)
            args.append(arg)
            fargs.pop(0)
            gargs.pop(0)
            continue
        elif fargs:
            if args and poly_eq(fargs[0][0], args[-1][0]):
                args.pop(-1)
            args.append(fargs.pop(0))
            continue
        elif gargs:
            if args and poly_eq(gargs[0][0], args[-1][0]):
                args.pop(-1)
            args.append(gargs.pop(0))
            continue
        break
    return args
