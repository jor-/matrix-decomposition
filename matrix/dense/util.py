import numpy as np
import scipy.linalg

import matrix.errors


def is_equal(A, B):
    return A.shape == B.shape and not np.any(A != B)


def is_almost_equal(A, B, rtol=1e-05, atol=1e-08):
    return A.shape == B.shape and np.allclose(A, B, rtol=rtol, atol=atol)


def is_finite(A):
    return np.all(np.isfinite(A))


def check_finite(A, check_finite=True):
    if check_finite and not is_finite(A):
        raise matrix.errors.MatrixNotFiniteError(A)


def solve_triangular(A, b, lower=True, unit_diagonal=False, overwrite_b=False, check_finite=True):
    return scipy.linalg.solve_triangular(A, b, lower=lower, unit_diagonal=unit_diagonal, overwrite_b=overwrite_b, check_finite=check_finite)
