import numpy as np
import typing
from numpy import typing as nptyping

from differentiation import diff
from differentiation import jacobi


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


def newton_raphson_multivar(
    func: typing.Callable[[nptyping.ArrayLike], nptyping.ArrayLike],
    x0: nptyping.ArrayLike,
    jacobian: typing.Callable[[nptyping.ArrayLike], np.ndarray] = None,
    tol: float = 1e-8,
    maxiter: int = 500,
) -> np.ndarray:
    if jacobian is None:
        x = x0
        for _ in range(maxiter):
            if np.linalg.norm(func(x)) < tol:
                return x
            jacobian = jacobi(func, x)
            # Can use any system of equations solving method here
            delta_x = -np.matmul(np.linalg.inv(jacobian), func(x))

            x += delta_x
        print("Warning: Did not converge within the specified number of iterations.")
        return x
    else:
        x = x0

        for _ in range(maxiter):
            if np.linalg.norm(func(x)) < tol:
                return x
            # Can use any system of equations solving method here
            delta_x = -np.matmul(np.linalg.inv(jacobian(x)), func(x))

            x += delta_x
        return x
