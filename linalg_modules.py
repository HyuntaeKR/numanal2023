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
    # Note that simply assigning mat to mat_ref will modify the original matrix too!
    # =====
    # When you assign an ndarray to another variable,
    # it creates a reference to the same data rather than making a copy of the data.
    # ===== By. ChatGPT

    mat_ref = np.copy(mat)

    for row_idx in range(row - 1):
        if ~naive:
            mat_ref = partial_pivot(mat_ref, row_idx)
        for row_idx2 in range(row_idx + 1, row):
            if mat_ref[row_idx2, row_idx] != 0:
                multiplier = mat_ref[row_idx2, row_idx] / mat_ref[row_idx, row_idx]
                mat_ref[row_idx2, :] = (
                    mat_ref[row_idx2, :] - multiplier * mat_ref[row_idx, :]
                )

    return mat_ref


def get_sys_info(coeff_mat: np.ndarray, const_vect: np.ndarray) -> int:
    """
    Prints out the info for system of linear equations
    and returns number of solutions.
    """
    row, column = np.shape(coeff_mat)
    rank_coeff = la.matrix_rank(coeff_mat)
    # Augmented matrix
    aug_mat = np.hstack((coeff_mat, const_vect))
    rank_aug = la.matrix_rank(aug_mat)

    if rank_aug == rank_coeff + 1:
        print("Inconsistent system. No solution.")
        num_sol = 0
        return num_sol
    elif rank_aug == rank_coeff:
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


def gaussian_elim(
    coeff_mat: np.ndarray, const_vect: np.ndarray, naive: bool = False
) -> np.ndarray:
    """
    Implements Gaussian elimination.
    Returns the root of the system.
    """
    row = len(coeff_mat)
    num_sol = get_sys_info(coeff_mat, const_vect)
    if num_sol != 1:
        raise Exception("Cannot get root!")
    aug_mat_ref = row_echelon_form(np.hstack((coeff_mat, const_vect)), naive)
    root_vect = np.zeros((row, 1))
    # Backward substitution
    # Can iterate through range in reverse order.
    for row_idx in range(row - 1, -1, -1):
        root_vect[row_idx] = (
            aug_mat_ref[row_idx, -1]
            - np.dot(
                # This part should use aug_mat_ref instead of aug_mat!
                aug_mat_ref[row_idx, row_idx + 1 : -1],
                root_vect[row_idx + 1 :, 0],
            )
        ) / aug_mat_ref[row_idx, row_idx]

    return root_vect


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
    mat_lowtri = np.copy(mat)

    for row_idx in range(row - 1, 0, -1):
        if ~naive:
            mat_lowtri = rev_partial_pivot(mat_lowtri, row_idx)
        for row_idx2 in range(row_idx - 1, -1, -1):
            multiplier = mat_lowtri[row_idx2, row_idx] / mat_lowtri[row_idx, row_idx]
            mat_lowtri[row_idx2, :] = (
                mat_lowtri[row_idx2, :] - multiplier * mat_lowtri[row_idx, :]
            )

    return mat_lowtri


def rev_gaussian_elim(
    coeff_mat: np.ndarray, const_vect: np.ndarray, naive: bool = False
) -> np.ndarray:
    """
    Reversed Gaussian elimination.
    Implements backward elimination and forward substitution.
    """
    row, column = np.shape(coeff_mat)
    num_sol = get_sys_info(coeff_mat, const_vect)
    if num_sol != 1:
        raise Exception("Cannot get root!")
    aug_mat_lowtri = backward_elim(np.hstack((coeff_mat, const_vect)), naive)
    root_vect = np.zeros((row, 1))
    # Forward substitution
    for row_idx in range(row):
        root_vect[row_idx] = (
            aug_mat_lowtri[row_idx, -1]
            - np.dot(aug_mat_lowtri[row_idx, :row_idx], root_vect[:row_idx, 0])
        ) / aug_mat_lowtri[row_idx, row_idx]

    return root_vect


def lu_decomp(coeff_mat: np.ndarray, const_vect: np.ndarray) -> typing.List[np.ndarray]:
    """
    Implements LU decomposition based on Doolittle's method.
    Returns list of [L, U, b'].
    Basically the same algorithm with row_echelon_form but
    records the multiplier value in L.
    """
    row = len(coeff_mat)
    # Create identity matrix for L
    lowertri = np.eye(row)
    uppertri = np.copy(coeff_mat)
    pivot_const = np.copy(const_vect)

    for row_idx in range(row - 1):
        # Implement partial pivoting by default
        max_row = np.argmax(abs(uppertri[row_idx:, row_idx])) + row_idx
        if max_row != row_idx:
            uppertri[[row_idx, max_row], :] = uppertri[[max_row, row_idx], :]
            pivot_const[[row_idx, max_row]] = pivot_const[[max_row, row_idx]]

        for row_idx2 in range(row_idx + 1, row):
            multiplier = uppertri[row_idx2, row_idx] / uppertri[row_idx, row_idx]
            uppertri[row_idx2, :] = (
                uppertri[row_idx2, :] - multiplier * uppertri[row_idx, :]
            )
            lowertri[row_idx2, row_idx] = multiplier

    return [lowertri, uppertri, pivot_const]


def solve_lud(coeff_mat: np.ndarray, const_vect: np.ndarray) -> np.ndarray:
    """
    Returns the root of a system of linear equations using LU decomposition.
    """
    # Use len() instead of using np.shape() and not using the column variable
    row = len(coeff_mat)
    lowertri, uppertri, pivot_const = lu_decomp(coeff_mat, const_vect)
    # Forward substitution to get vector y where,
    # Ly = b
    vect_y = np.zeros((row, 1))
    for row_idx in range(row):
        vect_y[row_idx] = (
            pivot_const[row_idx]
            - np.dot(lowertri[row_idx, :row_idx], vect_y[:row_idx, 0])
        ) / lowertri[row_idx, row_idx]
    # print(uppertri)
    # print(vect_y)
    # Backward substitution to get root vector where,
    # Ux = y
    root_vect = np.zeros((row, 1))
    for row_idx in range(row - 1, -1, -1):
        root_vect[row_idx] = (
            vect_y[row_idx, 0]
            - np.dot(uppertri[row_idx, row_idx + 1 :], root_vect[row_idx + 1 :])
        ) / uppertri[row_idx, row_idx]

    return root_vect


def tridiagonal(
    size: int, main_diag: float, sub_diag: float, sup_diag: float
) -> np.ndarray:
    """
    Returns a tridiagonal array with (size)x(size).
    Diagonal values are taken as parameters.
    """
    tridiag = np.zeros((size, size))

    for i in range(size):
        for j in range(size):
            if i == j:
                tridiag[i][j] = main_diag
            if j == i + 1:
                if i + 1 >= size:
                    pass
                else:
                    tridiag[i][j] = sup_diag
            if j == i - 1:
                if i - 1 < 0:
                    pass
                else:
                    tridiag[i][j] = sub_diag

    return tridiag


def thomas(tridiag: np.ndarray, const_vect: np.ndarray) -> np.ndarray:
    """
    Returns the root of a system of linear equations using the
    Thomas algorithm.
    """
    row = len(tridiag)
    for i in range(row):
        # Normalize by diagonal compound
        tridiag[i, :] = tridiag[i, :] / tridiag[i, i]
        const_vect[i] = const_vect[i] / tridiag[i, i]
        # Eliminate non-zero entries below the diagonal compound
        for i2 in range(i + 1, row):
            if tridiag[i2, i] != 0:
                tridiag[i2, :] -= tridiag[i, :] * tridiag[i2, i]
                # Subtract corresponding constant vector entries
                const_vect[i2] -= const_vect[i] * tridiag[i2, i]

    # Use backward substitution to get the root vector
    root_vect = np.zeros((row, 1))
    # Last entry is simply equated
    root_vect[-1] = const_vect[-1]
    for i in range(row - 2, -1, -1):
        root_vect[i] = const_vect[i] - tridiag[i, i + 1] * root_vect[i + 1]

    return root_vect


def jacobi(
    coeff_mat: np.ndarray,
    const_vect: np.ndarray,
    init_guess: np.ndarray = None,
    iterate: int = 100,
    tol: float = 1e-6,
):
    """
    Returns the root of a system of linear equations solved by Jacobi method.
    """
    row = len(coeff_mat)
    if init_guess == None:
        init_guess = np.ones(row)

    # Decompose the coefficient matrix
    diag_mat = np.diag(np.diag(coeff_mat))
    # Should set the 'k' parameter to exclude main diagonal
    lowertri = np.tril(coeff_mat, -1)
    uppertri = np.triu(coeff_mat, 1)

    root_vect = init_guess
    for _ in range(iterate):
        part1 = np.linalg.inv(diag_mat)
        part2 = const_vect - np.dot((lowertri + uppertri), root_vect)
        root_vect_new = np.dot(part1, part2)
        if np.linalg.norm(np.matmul(coeff_mat, root_vect_new) - const_vect, 2) < tol:
            return root_vect_new
        root_vect = root_vect_new

    return root_vect


def gauss_seidal(
    coeff_mat: np.ndarray,
    const_vect: np.ndarray,
    init_guess: np.ndarray = None,
    iterate: int = 100,
    tol: float = 1e-6,
):
    """
    Returns the root of a system of linear equations solved by Gauss-Seidal method.
    """
    row = len(coeff_mat)
    if init_guess == None:
        init_guess = np.ones(row)

    root_vect = init_guess

    for _ in range(iterate):
        for i in range(row):
            root_vect[i] = (
                const_vect[i]
                - np.dot(coeff_mat[i, :], root_vect)
                + coeff_mat[i, i] * root_vect[i]
            ) / coeff_mat[i, i]
        if np.linalg.norm(np.matmul(coeff_mat, root_vect) - const_vect, 2) < tol:
            return root_vect

    return root_vect
