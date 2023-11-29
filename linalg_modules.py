import numpy as np


class Chapter2_3:
    def __init__(self):
        """
        Functions from chapter2-3.
        """

    def partial_pivot(self, mat: np.ndarray, column: int) -> np.ndarray:
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

    def row_echelon_form(self, mat: np.ndarray, naive: bool = False) -> np.ndarray:
        """
        Returns row echelon form of a matrix.
        """
        row, column = np.shape(mat)
        # Copy of matrix to return
        # Note that simply assigning mat to mat_ref will modify the original matrix too!
        # =====
        # When you assign an ndarray to another variable,
        # it creates a reference to the same data rather than making a copy of the data.
        # ===== By. ChatGPT

        mat_ref = np.copy(mat)

        for row_idx in range(row - 1):
            if ~naive:
                mat_ref = self.partial_pivot(mat_ref, row_idx)
            for row_idx2 in range(row_idx + 1, row):
                if mat_ref[row_idx2, row_idx] != 0:
                    multiplier = mat_ref[row_idx2, row_idx] / mat_ref[row_idx, row_idx]
                    mat_ref[row_idx2, :] = (
                        mat_ref[row_idx2, :] - multiplier * mat_ref[row_idx, :]
                    )

        return mat_ref

    def get_sys_info(self, coeff_mat: np.ndarray, const_vect: np.ndarray) -> int:
        """
        Prints out the info for system of linear equations
        and returns number of solutions.
        """
        row, column = np.shape(coeff_mat)
        rank_coeff = np.linalg.matrix_rank(coeff_mat)
        # Augmented matrix
        aug_mat = np.hstack((coeff_mat, const_vect))
        rank_aug = np.linalg.matrix_rank(aug_mat)

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
        self, coeff_mat: np.ndarray, const_vect: np.ndarray, naive: bool = False
    ) -> np.ndarray:
        """
        Implements Gaussian elimination.
        Returns the root of the system.
        """
        row, column = np.shape(coeff_mat)
        num_sol = self.get_sys_info(coeff_mat, const_vect)
        if num_sol != 1:
            raise Exception("Cannot get root!")
        aug_mat_ref = self.row_echelon_form(np.hstack((coeff_mat, const_vect)), naive)
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

    def forward_subs():
        pass

    def rev_partial_pivot(self, mat: np.ndarray, column: int) -> np.ndarray:
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
    def backward_elim(self, mat: np.ndarray, naive: bool = False) -> np.ndarray:
        """
        Makes the input matrix into lower triangluar form
        by elemetary row operation.
        """
        row, column = np.shape(mat)
        mat_lowtri = np.copy(mat)

        for row_idx in range(row - 1, 0, -1):
            if ~naive:
                mat_lowtri = self.rev_partial_pivot(mat_lowtri, row_idx)
            for row_idx2 in range(row_idx - 1, -1, -1):
                multiplier = (
                    mat_lowtri[row_idx2, row_idx] / mat_lowtri[row_idx, row_idx]
                )
                mat_lowtri[row_idx2, :] = (
                    mat_lowtri[row_idx2, :] - multiplier * mat_lowtri[row_idx, :]
                )

        return mat_lowtri

    def rev_gaussian_elim(
        self, coeff_mat: np.ndarray, const_vect: np.ndarray, naive: bool = False
    ) -> np.ndarray:
        """
        Reversed Gaussian elimination.
        Implements backward elimination and forward substitution.
        """
        row, column = np.shape(coeff_mat)
        num_sol = self.get_sys_info(coeff_mat, const_vect)
        if num_sol != 1:
            raise Exception("Cannot get root!")
        aug_mat_lowtri = self.backward_elim(np.hstack((coeff_mat, const_vect)), naive)
        root_vect = np.zeros((row, 1))
        # Forward substitution
        for row_idx in range(row):
            root_vect[row_idx] = (
                aug_mat_lowtri[row_idx, -1]
                - np.dot(aug_mat_lowtri[row_idx, :row_idx], root_vect[:row_idx, 0])
            ) / aug_mat_lowtri[row_idx, row_idx]

        return root_vect
    
    def backward_subs():
        pass
