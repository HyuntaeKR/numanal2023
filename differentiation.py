import numpy as np
import typing
from numpy import typing as nptyping


def diff(func: typing.Callable[[float], float], x: float):
    """
    Return numerically differnetiated value of given function at point x.
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


def gradient(f, x) -> np.ndarray:
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
