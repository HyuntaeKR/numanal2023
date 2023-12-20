import typing
import numpy as np
import numpy.linalg as la


def partial_pivot(mat: np.ndarray, column: int) -> np.ndarray:
    """
    Partially pivots matrix(mat) with respect to input column.
    """
    # Should consider the absolute maximum value.
    # Should add 'column' because the argmax returns relative index.
    max_row = np.argmax(abs(mat[column:, column])) + column

    if max_row == column:
        return mat
    else:
        # How to switch rows!
        mat[[column, max_row], :] = mat[[max_row, column], :]
        return mat


def row_echelon_form(mat: np.ndarray, naive: bool = False) -> np.ndarray:
    """
    Returns row echelon form of a matrix.
    """
    row = len(mat)
    # Copy of matrix to return
    # Note that simply assigning mat to ref will modify the original matrix too!
    # =====
    # When you assign an ndarray to another variable,
    # it creates a reference to the same data rather than making a copy of the data.
    # ===== By. ChatGPT

    ref = np.copy(mat)  # ref: row echelon form

    for i in range(row - 1):
        if ~naive:
            ref = partial_pivot(ref, i)
        for j in range(i + 1, row):
            if ref[j, i] != 0:
                multiplier = ref[j, i] / ref[i, i]
                ref[j, :] = ref[j, :] - multiplier * ref[i, :]

    return ref


def get_sys_info(A: np.ndarray, b: np.ndarray) -> int:
    """
    Prints out the info for system of linear equations
    and returns number of solutions.

    System >> Ax=b
    ---
    """
    row, column = np.shape(A)
    rank_of_A = la.matrix_rank(A)
    # Augmented matrix
    aug = np.hstack((A, b))
    rank_of_aug = la.matrix_rank(aug)

    if rank_of_aug == rank_of_A + 1:
        print("Inconsistent system. No solution.")
        num_sol = 0
        return num_sol
    elif rank_of_aug == rank_of_A:
        if row == column:
            print("Consistent and independent system. Unique solution.")
            num_sol = 1
            return num_sol
        else:
            print("Consistent and dependent system. Infinite solutions.")
            num_sol = np.inf
            return num_sol
    else:
        print("Unvalid system.")
        return np.nan


def gaussian_elim(A: np.ndarray, b: np.ndarray, naive: bool = False) -> np.ndarray:
    """
    Implements Gaussian elimination.
    Returns the root of the system.

    System >> Ax=b
    ---
    """
    row = len(A)
    num_sol = get_sys_info(A, b)
    if num_sol != 1:
        raise Exception("Cannot get root of given system.")
    aug_ref = row_echelon_form(np.hstack((A, b)), naive)
    x = np.zeros((row, 1))
    # Backward substitution
    # Can iterate through range in reverse order.
    for i in range(row - 1, -1, -1):
        x[i] = (
            aug_ref[i, -1]
            - np.dot(
                aug_ref[i, i + 1 : -1],
                x[i + 1 :, 0],
            )
        ) / aug_ref[i, i]

    return x


def rev_partial_pivot(mat: np.ndarray, column: int) -> np.ndarray:
    """
    Partially pivots matrix(mat) with respect to input column.
    """
    # Should consider the absolute maximum value.
    # Should add 'column' because the argmax returns relative index.
    max_row = np.argmax(abs(mat[: column + 1, column]))

    if max_row == column:
        return mat
    else:
        # How to switch rows!
        mat[[column, max_row], :] = mat[[max_row, column], :]
        return mat


# Use backward_elim instead of row_echelon_form
def backward_elim(mat: np.ndarray, naive: bool = False) -> np.ndarray:
    """
    Makes the input matrix into lower triangluar form
    by elemetary row operation.
    """
    row = len(mat)
    lowtri = np.copy(mat)

    for i in range(row - 1, 0, -1):
        if ~naive:
            lowtri = rev_partial_pivot(lowtri, i)
        for j in range(i - 1, -1, -1):
            multiplier = lowtri[j, i] / lowtri[i, i]
            lowtri[j, :] = lowtri[j, :] - multiplier * lowtri[i, :]

    return lowtri


def rev_gaussian_elim(A: np.ndarray, b: np.ndarray, naive: bool = False) -> np.ndarray:
    """
    Reversed Gaussian elimination.
    Implements backward elimination and forward substitution.

    System >> Ax=b
    ---
    """
    row = len(A)
    num_sol = get_sys_info(A, b)
    if num_sol != 1:
        raise Exception("Cannot get root!")
    aug_lowtri = backward_elim(np.hstack((A, b)), naive)
    x = np.zeros((row, 1))
    # Forward substitution
    for i in range(row):
        x[i] = (aug_lowtri[i, -1] - np.dot(aug_lowtri[i, :i], x[:i, 0])) / aug_lowtri[
            i, i
        ]

    return x


def lu_decomp(A: np.ndarray, b: np.ndarray) -> typing.List[np.ndarray]:
    """
    Implements LU decomposition based on Doolittle's method.
    Returns list of [L, U, b'].
    Basically the same algorithm with `row_echelon_form` but
    records the multiplier value in L.
    """
    row = len(A)
    # Create identity matrix for L
    L = np.eye(row)
    U = np.copy(A)
    pivot_b = np.copy(b)

    for i in range(row - 1):
        # Implement partial pivoting by default
        max_row = np.argmax(abs(U[i:, i])) + i
        if max_row != i:
            U[[i, max_row], :] = U[[max_row, i], :]
            pivot_b[[i, max_row]] = pivot_b[[max_row, i]]

        for j in range(i + 1, row):
            multiplier = U[j, i] / U[i, i]
            U[j, :] = U[j, :] - multiplier * U[i, :]
            L[j, i] = multiplier

    return [L, U, pivot_b]


def solve_lud(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Returns the root of a system of linear equations using LU decomposition.
    """
    row = len(A)
    L, U, pivot_b = lu_decomp(A, b)
    # Forward substitution to get vector y where,
    # Ly = b
    y = np.zeros((row, 1))
    for i in range(row):
        y[i] = (pivot_b[i] - np.dot(L[i, :i], y[:i, 0])) / L[i, i]
    # Backward substitution to get root vector where,
    # Ux = y
    x = np.zeros((row, 1))
    for i in range(row - 1, -1, -1):
        x[i] = (y[i, 0] - np.dot(U[i, i + 1 :], x[i + 1 :])) / U[i, i]

    return x


def tridiagonal(size: int, main: float, sub: float, sup: float) -> np.ndarray:
    """
    Returns a tridiagonal array with (size)x(size).
    Diagonal values are taken as parameters.
    """
    tridiag = np.zeros((size, size))

    for i in range(size):
        for j in range(size):
            if i == j:
                tridiag[i][j] = main
            if j == i + 1:
                if i + 1 >= size:
                    pass
                else:
                    tridiag[i][j] = sup
            if j == i - 1:
                if i - 1 < 0:
                    pass
                else:
                    tridiag[i][j] = sub

    return tridiag


def thomas(T: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Returns the root of a system of linear equations using the
    Thomas algorithm.

    System >> Tx = b
    ---
    """
    row = len(T)
    for i in range(row):
        # Normalize by diagonal compound
        T[i, :] = T[i, :] / T[i, i]
        b[i] = b[i] / T[i, i]
        # Eliminate non-zero entries below the diagonal compound
        for j in range(i + 1, row):
            if T[j, i] != 0:
                T[j, :] -= T[i, :] * T[j, i]
                # Subtract corresponding constant vector entries
                b[j] -= b[i] * T[j, i]

    # Use backward substitution to get the root vector
    x = np.zeros((row, 1))
    # Last entry is simply equated
    x[-1] = b[-1]
    for i in range(row - 2, -1, -1):
        x[i] = b[i] - T[i, i + 1] * x[i + 1]

    return x


def jacobi(
    A: np.ndarray,
    b: np.ndarray,
    x0: np.ndarray = None,
    maxiter: int = 100,
    tol: float = 1e-6,
):
    """
    Returns the root of a system of linear equations solved by Jacobi method.

    System >> Ax=b
    ---
    """
    row = len(A)
    if x0 is None:
        x0 = np.ones(row)

    # Decompose the coefficient matrix into D, L0, U0
    D = np.diag(np.diag(A))
    # Should set the 'k' parameter to exclude main diagonal
    L0 = np.tril(A, -1)
    U0 = np.triu(A, 1)

    x = x0
    for _ in range(maxiter):
        part1 = np.linalg.inv(D)
        part2 = b - np.dot((L0 + U0), x)
        x_new = np.dot(part1, part2)
        if np.linalg.norm(np.matmul(A, x_new) - b, 2) < tol:
            return x_new
        x = x_new

    return x


def gauss_seidal(
    A: np.ndarray,
    b: np.ndarray,
    x0: np.ndarray = None,
    maxiter: int = 100,
    tol: float = 1e-6,
):
    """
    Returns the root of a system of linear equations solved by Gauss-Seidal method.

    System >> Ax=b
    ---
    """
    row = len(A)
    if x0 is None:
        x0 = np.ones(row)

    x = x0

    for _ in range(maxiter):
        for i in range(row):
            x[i] = (b[i] - np.dot(A[i, :], x) + A[i, i] * x[i]) / A[i, i]
        if np.linalg.norm(np.matmul(A, x) - b, 2) < tol:
            return x

    return x
