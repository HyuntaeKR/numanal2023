import numpy as np
import typing


def diff(func: typing.Callable[[float], float], x: float):
    """
    Return numerically differnetiated value of given function at point x.
    """
    increment = 1e-5
    result = (func(x + increment) - func(x)) / increment
    return result
