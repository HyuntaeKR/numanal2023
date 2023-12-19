import numpy as np
import typing
from numpy import typing as nptyping


def diff(func: typing.Callable[[float], float], x: float):
    """
    Return numerically differnetiated value of given scaler function at point x.
    """
    inc = 1e-5
    result = (func(x + inc) - func(x)) / inc
    return result


def jacobi(
    func: typing.Callable[[nptyping.ArrayLike], nptyping.ArrayLike],
    x: nptyping.ArrayLike,
) -> np.ndarray:
    """
    Return numerically calculated Jacobian of given vector function at vector x.
    """
    num_var = len(x)
    num_eqn = len(func(x))
    jacobian = np.zeros((num_eqn, num_var))
    inc = 1e-5

    for i in range(num_var):
        x_plus_inc = x.copy()
        x_plus_inc[i] += inc
        jacobian[:, i] = (func(x_plus_inc) - func(x)) / inc

    return jacobian


def gradient(
    f: typing.Callable[[nptyping.ArrayLike], float], x: nptyping.ArrayLike
) -> np.ndarray:
    """
    Return numerically calculated gradient of given scaler function at vector x.
    """
    dim = len(x)
    grad_f = np.zeros(dim)
    inc = 1e-5

    for i in range(dim):
        x_plus_inc = x.copy()
        x_plus_inc[i] += inc
        grad_f[i] = (f(x_plus_inc) - f(x)) / inc

    return grad_f


def finite_difference(
    f: typing.Callable[[float], float], x: float, h: float = 1e-2, order: int = 1
):
    """
    Returns derivative of function f at point x, based on central finite difference.
    Order of remaining terms is O(h^2).
    Maximum differentiation order of 4.
    """
    coeff = np.array(
        [
            [0, 0, 0, -1 / 2, 0, 1 / 2, 0, 0, 0],
            [0, 0, 0, 1, -2, 1, 0, 0, 0],
            [0, 0, -1 / 2, 1, 0, -1, 1 / 2, 0, 0],
            [0, 0, 1, -4, 6, -4, 1, 0, 0],
        ]
    )
    x_grid = np.array(
        [
            [
                x - 4 * h,
                x - 3 * h,
                x - 2 * h,
                x - h,
                x,
                x + h,
                x + 2 * h,
                x + 3 * h,
                x + 4 * h,
            ]
        ]
    ).T
    f_diff = np.sum(np.dot(coeff[order - 1, :], f(x_grid))) / h**order

    return f_diff
