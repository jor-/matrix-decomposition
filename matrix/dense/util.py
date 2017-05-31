import numpy as np

import matrix.errors


def equal(A, B):
    return A.shape == B.shape and not np.any(A != B)


def is_finite(A):
    return np.all(np.isfinite(A))


def check_finite(A, check_finite=True):
    if check_finite and not is_finite(A):
        raise matrix.errors.MatrixNotFiniteError(matrix=A)
