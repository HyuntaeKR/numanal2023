import numpy as np
import typing
from numpy import typing as nptyping

from differentiation import gradient


def golden_section(
    f: typing.Callable[[float], float],
    boundary: nptyping.ArrayLike,
    maxiter=500,
    tol=1e-5,
) -> float:
    r = (-1 + np.sqrt(5)) / 2  # golden ratio
    a = boundary[0]  # lower bound
    b = boundary[1]  # upper bound

    for _ in range(maxiter):
        h = b - a
        if abs(h) < tol:
            return (a + b) / 2
        lower = b - r * h
        upper = a + r * h

        if f(lower) > f(upper):
            a = lower
        else:
            b = upper

    print("Warning: Did not converge within given number of iteration.")
    return (a + b) / 2


def gradient_descent(
    f: typing.Callable[[nptyping.ArrayLike], float],
    x0: nptyping.ArrayLike,
    grad_f=None,
    step=0.01,
    maxiter=10000,
    tol=1e-6,
    traj: bool = False,
) -> float:
    x = x0
    if grad_f is None:
        if traj:
            trajectory = np.empty((maxiter, 3))
            for i in range(maxiter):
                trajectory[i, 0:2] = x
                trajectory[i, 2] = f(x)
                if abs(np.linalg.norm(gradient(f, x), 2)) < tol:
                    return x, trajectory[: i + 1, :]

                x -= step * gradient(f, x)
            print("Warning: Did not converge within given number of iteration.")
            return x, trajectory
        else:
            for i in range(maxiter):
                if abs(np.linalg.norm(gradient(f, x), 2)) < tol:
                    return x

                x -= step * gradient(f, x)
            print("Warning: Did not converge within given number of iteration.")
            return x
    else:
        for _ in range(maxiter):
            if abs(np.linalg.norm(grad_f(x), 2)) < tol:
                return x

            x -= step * grad_f(f, x)
        print("Warning: Did not converge within given number of iteration.")
        return x


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    print("HELLO HAHA")
