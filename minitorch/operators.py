"""
Collection of the core mathematical operators used throughout the code base.
"""

import math
from typing import Callable, Iterable

# ## Task 0.1
#
# Implementation of a prelude of elementary functions.


def mul(x: float, y: float) -> float:
    """Multiplies x times y."""
    return x * y


def id(x: float) -> float:
    """Identity function"""
    return x


def add(x: float, y: float) -> float:
    """Add x plus y"""
    return x + y


def neg(x: float) -> float:
    """Negation"""
    # have to cast because some tests failed otherwise since ints were called
    return -float(x)


def lt(x: float, y: float) -> float:
    """Less than.

    Returns 1.0 if x is less than y, otherwise 0.0.
    """
    return 1.0 if x < y else 0.0


def eq(x: float, y: float) -> float:
    """Equality"""
    return 1.0 if x == y else 0.0


def max(x: float, y: float) -> float:
    """Maximium of two values"""
    return x if x > y else y


def is_close(x: float, y: float) -> float:
    """Is-close

    Returns 1.0 if x and y are within a tolerance of each other, otherwise 0.0.
    """
    return 1.0 if abs(x - y) < 1e-2 else 0.0


def sigmoid(x: float) -> float:
    r"""Logistic sigmoid function

    Args:
        x: input value

    Returns:
        Value of sigmoid evaluated at x

    """
    # different code paths for numerical stability
    if x >= 0:
        ret = 1.0 / (1.0 + math.exp(-x))
    else:
        ret = math.exp(x) / (1.0 + math.exp(x))
    return ret


def relu(x: float) -> float:
    """Rectified linear unit

    Inputs:
        x: value

    Returns:
        Value of ReLU at x
    """
    # This wasn't numba jit-able
    #return max(x, 0.0)
    # But this is
    return x if x > 0 else 0.0


EPS = 1e-6


def log(x: float) -> float:
    """Natural logarithm of x"""
    return math.log(x + EPS)


def exp(x: float) -> float:
    """Exponential function of x"""
    return math.exp(x)


def log_back(x: float, d: float) -> float:
    """Backward pass for log(x).

    Inputs:
        x: value

    Returns:
        d * dlog(x)dx
    """
    return d * 1.0 / x


def inv(x: float) -> float:
    """Reciprocal of x"""
    return 1.0 / x


def inv_back(x: float, d: float) -> float:
    """Backward pass for 1/x

    Inputs:
        x: value

    Returns:
        d * dx^-1/dx
    """
    return d * -(x**-2)


def relu_back(x: float, d: float) -> float:
    """Backward pass for relu.

    Inputs:
        x: value
        d: gradient of output

    Returns:
        d * drelu(x)/dx
    """
    return d * (1.0 if x > 0.0 else 0.0)


# ## Task 0.3

# Small practice library of elementary higher-order functions.


def map(fn: Callable[[float], float]) -> Callable[[Iterable[float]], Iterable[float]]:
    """
    Higher-order map.

    See https://en.wikipedia.org/wiki/Map_(higher-order_function)

    Args:
        fn: Function from one value to one value.

    Returns:
         A function that takes a list, applies `fn` to each element, and returns a
         new list
    """

    def apply(ls: Iterable[float]) -> Iterable[float]:
        return [fn(x) for x in ls]

    return apply


def negList(ls: Iterable[float]) -> Iterable[float]:
    "Use `map` and `neg` to negate each element in `ls`"
    return map(neg)(ls)


def zipWith(
    fn: Callable[[float, float], float],
) -> Callable[[Iterable[float], Iterable[float]], Iterable[float]]:
    """
    Higher-order zipwith (or map2).

    See https://en.wikipedia.org/wiki/Map_(higher-order_function)

    Args:
        fn: combine two values

    Returns:
         Function that takes two equally sized lists `ls1` and `ls2`, produce a new list by
         applying fn(x, y) on each pair of elements.

    """

    def apply(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
        return [fn(x, y) for x, y in zip(ls1, ls2)]

    return apply


def addLists(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
    "Add the elements of `ls1` and `ls2` using `zipWith` and `add`"
    return zipWith(add)(ls1, ls2)


def reduce(
    fn: Callable[[float, float], float], start: float
) -> Callable[[Iterable[float]], float]:
    r"""
    Higher-order reduce.

    Args:
        fn: combine two values
        start: start value $x_0$

    Returns:
         Function that takes a list `ls` of elements
         $x_1 \ldots x_n$ and computes the reduction :math:`fn(x_3, fn(x_2,
         fn(x_1, x_0)))`
    """
    res = start

    def apply(ls: Iterable[float]) -> float:
        nonlocal res
        for x in ls:
            res = fn(x, res)
        return res

    return apply


def sum(ls: Iterable[float]) -> float:
    "Sum up a list using `reduce` and `add`."
    return reduce(add, 0.0)(ls)


def prod(ls: Iterable[float]) -> float:
    "Product of a list using `reduce` and `mul`."
    return reduce(mul, 1.0)(ls)
