import numpy as np
import typing
from numpy import typing as nptyping


def diff(func: typing.Callable[[float], float], x: float):
    """
    Return numerically differnetiated value of given function at point x.
    """
    increment = 1e-5
    result = (func(x + increment) - func(x)) / increment
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
    increment = 1e-5

    for i in range(num_var):
        x_plus_inc = x.copy()
        x_plus_inc[i] += increment
        jacobian[:, i] = (func(x_plus_inc) - func(x)) / increment

    return jacobian
