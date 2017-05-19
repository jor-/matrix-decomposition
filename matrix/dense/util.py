import numpy as np

import matrix.errors


def equal(A, B):
    return A.shape == B.shape and not np.any(A != B)


def check_finite_matrix(A):
    if not np.all(np.isfinite(A)):
        raise matrix.errors.MatrixNotFiniteError(matrix=A)
