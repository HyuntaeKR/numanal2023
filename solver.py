import numpy as np
import typing

from differentiation import diff


def bisection(
    func: typing.Callable[[float], float],
    boundary: list,
    maxiter: int = 500,
    tol: float = 1e-5,
) -> float:
    """
    Returns root of given function in given boundary found by bisection method.
    Function should be scalar-valued and single-variable.
    """
    a = boundary[0]  # lower bound
    b = boundary[1]  # upper bound

    assert func(a) * func(b) < 0, "No root in given range."

    if func(a) * func(b) == 0:
        if func(a) == 0:
            return a
        else:
            return b
    else:
        for _ in range(maxiter):
            m = (a + b) / 2  # midpoint

            if abs(func(m)) < tol:
                return m

            if func(a) * func(m) > 0:
                a = m
            else:  # func(b)*func(m) > 0
                b = m

        return m


def newton_raphson(
    func: typing.Callable[[float], float],
    diff_func: typing.Callable[[float], float] = None,
    x0: float = 0.1,
    tol: float = 1e-8,
    maxiter: int = 500,
) -> float:
    """
    Returns root of given function found by Newton-Raphson method.
    Function should be scalar-valued and single-variable.
    """
    if diff_func is None:  # Numerically differentiate
        x = x0
        for _ in range(maxiter):
            if abs(func(x)) < tol:
                return x
            x -= func(x) / diff(func, x)

        return x
    else:
        x = x0
        for _ in range(maxiter):
            if abs(func(x)) < tol:
                return x
            x -= func(x) / diff_func(x)

        return x
