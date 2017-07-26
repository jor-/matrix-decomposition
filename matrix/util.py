import numpy as np
import scipy.sparse

import matrix.dense.util
import matrix.sparse.util

import matrix.errors


def as_matrix_or_array(A, check_ndim_values=None):
    if not scipy.sparse.issparse(A):
        A = np.asanyarray(A)
    if check_ndim_values is not None and A.ndim not in check_ndim_values:
        raise ValueError('The number of dimensions of the input should be in '
                         '{} but it is {}.'.format(check_ndim_values, A.ndim))
    return A


def as_matrix_or_vector(A):
    return as_matrix_or_array(A, check_ndim_values=(1, 2))


def as_matrix(A):
    return as_matrix_or_array(A, check_ndim_values=(2,))


def as_vector(A):
    A = np.asarray(A)
    if A.ndim != 1:
        raise ValueError('The number of dimensions of the input should be 1 '
                         'but it is {}.'.format(A.ndim))
    return A


def equal(A, B):
    A_is_sparse = matrix.sparse.util.is_sparse(A)
    B_is_sparse = matrix.sparse.util.is_sparse(B)
    if A_is_sparse != B_is_sparse:
        return False
    if A_is_sparse:
        assert B_is_sparse
        return matrix.sparse.util.equal(A, B)
    else:
        assert not B_is_sparse
        return matrix.dense.util.equal(A, B)


def almost_equal(A, B):
    A_is_sparse = matrix.sparse.util.is_sparse(A)
    B_is_sparse = matrix.sparse.util.is_sparse(B)
    if A_is_sparse != B_is_sparse:
        return False
    if A_is_sparse:
        assert B_is_sparse
        return matrix.sparse.util.almost_equal(A, B)
    else:
        assert not B_is_sparse
        return matrix.dense.util.almost_equal(A, B)


def check_square_matrix(A):
    if len(A.shape) != 2 or A.shape[0] != A.shape[1]:
        raise matrix.errors.MatrixNotSquareError(matrix=A)


def is_finite(A):
    if scipy.sparse.issparse(A):
        return matrix.sparse.util.is_finite(A)
    else:
        return matrix.dense.util.is_finite(A)


def check_finite(A, check_finite=True):
    if check_finite and not is_finite(A):
        raise matrix.errors.MatrixNotFiniteError(matrix=A)


def solve_triangular(A, b, lower=True, unit_diagonal=False, overwrite_b=False, check_finite=True):
    if scipy.sparse.issparse(A):
        return matrix.sparse.util.solve_triangular(
            A, b,
            lower=lower, unit_diagonal=unit_diagonal,
            overwrite_b=overwrite_b, check_finite=check_finite)
    else:
        return matrix.dense.util.solve_triangular(
            A, b,
            lower=lower, unit_diagonal=unit_diagonal,
            overwrite_b=overwrite_b, check_finite=check_finite)
