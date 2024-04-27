from functools import reduce
import operator
import torch


def prod(x):
    return reduce(operator.mul, x, 1)


def hypergeometric_matrix(a, b, X, alpha=1, max_iter=1000, tol=1e-6):
    """
    Hypergeometric function of a matrix argument

    !!! quote "Reference"
        - https://en.wikipedia.org/wiki/Hypergeometric_function_of_a_matrix_argument
        - Plamen Koev, Alan Edelman. "The Efficient Evaluation of the
          Hypergeometric Function of a Matrix Argument."
          https://arxiv.org/pdf/math/0505344

    """
    eig = torch.linalg.eigvalsh(X)
    if not isinstance(a, (list, tuple)):
        a = [a]
    if not isinstance(b, (list, tuple)):
        b = [b]
    acc = 0
    for k in range(max_iter):
        for kappa in partitions(k):
            acc1 = (
                jack_cnorm(*eig.unbind(-1), alpha, kappa) *
                prod(pochhammer(ai, alpha, kappa) for ai in a) *
                prod(pochhammer(bi, alpha, kappa) for bi in b) *
                1 / prod(range(1, k))
            )
            acc += acc1
            if acc1.abs().mean() < tol:
                break
    return acc


def pochhammer(a, alpha, kappa):
    """
    Generalized Pochhammer symbol

    !!! quote "Reference"
        https://en.wikipedia.org/wiki/Generalized_Pochhammer_symbol
    """
    m = len(kappa)
    acc = 1
    for i in range(m):
        for j in range(kappa[i]):
            acc *= (a - (i-1) / alpha + j - 1)
    return acc


def jack_cnorm(*x, alpha, kappa):
    """C-normalization of the Jack function"""
    return NotImplemented


def jack(*x, alpha, kappa):
    """Jack function"""
    return NotImplemented


def is_strip_partition(mu, kappa):
    """
    Check that mu is a strip partition of kappa

    Return true if
        k_1 ≤ μ_1 ≤ k_2 ≤ μ_1 ≤ ...≤ k_{N-1} ≤ μ_{N-1} ≤ k_{N} <= μ_{N} = 0
    """
    *mu, m = mu
    if m != 0:
        return False
    k, *kappa = kappa
    while mu:
        m, *mu = mu
        if m > k:
            return False
        k, *kappa = kappa
        if m < k:
            return False
    return True


def conjugate_partition(kappa):
    """
    Compute the conjugate of a partition

    !!! quote "Reference"
        https://en.wikipedia.org/wiki/Integer_partition#Conjugate_and_self-conjugate_partitions
    """
    return partition_from_diagram(partition_diagram(kappa).T)


def partition_from_diagram(diagram):
    """
    Convert a diagram into a partition

    !!! quote "Reference"
        https://en.wikipedia.org/wiki/Young_tableau#Diagrams

    Parameters
    ----------
    diagram : (M, N) tensor

    Returns
    -------
    kappa : list[int]
    """
    return [row.sum().item() for row in diagram]


def partition_diagram(kappa):
    """
    Compute the diagram of a partition

    !!! quote "Reference"
        https://en.wikipedia.org/wiki/Young_tableau#Diagrams

    Parameters
    ----------
    kappa : list[int]
        A partition of an integer

    Returns
    -------
    diagram : (len(kappa), max(kappa)) tensor[bool]
        Diagram of the partition, as a binary matrix
    """
    d = torch.zeros([len(kappa), max(kappa)], dtype=bool)
    for i, k in enumerate(kappa):
        d[i, :k] = True
    return d


def partitions(n: int):
    """Compute the partitions of the integer n

    !!! quote "Reference"
        - https://en.wikipedia.org/wiki/Integer_partition
        - https://jeromekelleher.net/generating-integer-partitions.html

    Parameters
    ----------
    n : int
        Integer to partition

    Yields
    ------
    kappa : list[int]
        A (sorted) partition of the integer
    """
    a = [0 for i in range(n + 1)]
    i = 1
    y = n - 1
    while i != 0:
        x = a[i - 1] + 1
        i -= 1
        while 2 * x <= y:
            a[i] = x
            y -= x
            i += 1
        j = i + 1
        while x <= y:
            a[i] = x
            a[j] = y
            yield a[:i + 2]
            x += 1
            y -= 1
        a[i] = x + y
        y = x + y - 1
        yield a[:i + 1]


def betaln(x, y):
    """
    Log of the Beta function

    !!! quote "Reference"
        https://en.wikipedia.org/wiki/Beta_function
    """
    gammaln = torch.special.gammaln
    return gammaln(x) + gammaln(y) - gammaln(x+y)


def beta(x, y):
    """
    Beta function

    !!! quote "Reference"
        https://en.wikipedia.org/wiki/Beta_function
    """
    return betaln(x, y).exp()


def gamma(x):
    """
    Gamma function

    !!! quote "Reference"
        https://en.wikipedia.org/wiki/Gamma_function
    """
    return torch.special.gammaln(x).exp()


gammaln = torch.special.gammaln
