import sympy as sy
from sympy import symbols, diff, Matrix, simplify


def maclaurin(expr, order: int = 2):
    """
    Returns maclaurin series of expression with given order.
    """
    x = symbols("x")
    series = 0
    for i in range(order + 1):
        series += diff(expr, x, i).subs(x, 0) * x**i / sy.factorial(i)

    return series


def quad_taylor(expr, center=list):
    """
    Returns the quadratic taylor expansion of a two variable form expression
    about the given center.
    """
    x, y = symbols("x y")
    z = Matrix([[x, y]]).T  # z is the root vector [[x], [y]]
    a = Matrix([center]).T
    term1 = expr.subs([(x, center[0]), (y, center[1])])
    term2 = Matrix(
        [
            [
                diff(expr, x).subs([(x, center[0]), (y, center[1])]),
                diff(expr, y).subs([(x, center[0]), (y, center[1])]),
            ]
        ]
    ).dot(z - a)
    term3_1 = Matrix(
        [
            [
                diff(expr, x, 2).subs([(x, center[0]), (y, center[1])]),
                diff(diff(expr, x), y).subs([(x, center[0]), (y, center[1])]),
            ],
            [
                diff(diff(expr, y), x).subs([(x, center[0]), (y, center[1])]),
                diff(expr, y, 2).subs([(x, center[0]), (y, center[1])]),
            ],
        ]
    ) * (z - a)
    term3_2 = ((z - a).T) * (term3_1) / 2

    series = term1 + term2 + term3_2[0]
    series = simplify(series)

    return series


if __name__ == "__main__":
    from IPython.display import display

    x, y = symbols("x y")
    expr = sy.exp(x * y) * sy.sin(x + y)
    display(quad_taylor(expr, [1, 2]))
