import numpy as np
import typing


def diff(func: typing.callable[[float], float], x: float, increment=1e-5):
    """
    Return numerically differnetiated value of given function at point x.
    """
    result = (func(x + increment) - func(x)) / increment
    return result
