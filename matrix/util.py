import numpy as np
import scipy.sparse

import matrix
import matrix.errors
import matrix.dense.util
import matrix.sparse.util


def as_matrix_or_array(A, check_ndim_values=None):
    if not scipy.sparse.issparse(A):
        A = np.asarray(A)
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


def is_equal(A, B):
    A_is_sparse = matrix.sparse.util.is_sparse(A)
    B_is_sparse = matrix.sparse.util.is_sparse(B)
    if A_is_sparse != B_is_sparse:
        return False
    if A_is_sparse:
        assert B_is_sparse
        return matrix.sparse.util.is_equal(A, B)
    else:
        assert not B_is_sparse
        return matrix.dense.util.is_equal(A, B)


def is_almost_equal(A, B, rtol=1e-05, atol=1e-08):
    A_is_sparse = matrix.sparse.util.is_sparse(A)
    B_is_sparse = matrix.sparse.util.is_sparse(B)
    if A_is_sparse and B_is_sparse:
        is_almost_equal = matrix.sparse.util.is_almost_equal
    else:
        if A_is_sparse:
            A = A.toarray()
        if B_is_sparse:
            B = B.toarray()
        is_almost_equal = matrix.dense.util.is_almost_equal
    return is_almost_equal(A, B, rtol=rtol, atol=atol)


def check_square_matrix(A):
    if len(A.shape) != 2 or A.shape[0] != A.shape[1]:
        raise matrix.errors.MatrixNotSquareError(A)


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


def set_nearly_zero_to_zero(A, min_abs_value=None):
    # determine dtype resolution
    dtype_resolution = np.finfo(A.dtype).resolution

    # check min_abs_value
    if min_abs_value is None:
        min_abs_value = dtype_resolution
    else:
        if min_abs_value < 0:
            raise ValueError('min_abs_value {} has to be greater or equal zero.'.format(min_abs_value))
        if min_abs_value < dtype_resolution:
            matrix.logger.warning('Setting min_abs_value to resolution {} of matrix data type {}.'.format(dtype_resolution, A.dtype))
            min_abs_value = dtype_resolution

    # apply min_abs_value
    if min_abs_value > 0:
        if scipy.sparse.issparse(A):
            A.data[np.abs(A.data) < min_abs_value] = 0
            A.eliminate_zeros()
        else:
            A[np.abs(A) < min_abs_value] = 0

    return A


def set_diagonal_nearly_real_to_real(A, min_abs_value=None):
    # check if complex dtype
    dtype = A.dtype
    if np.iscomplexobj(np.array([], dtype=dtype)):

        # determine min_abs_value
        dtype_resolution = np.finfo(dtype).resolution
        if min_abs_value is None:
            min_abs_value = dtype_resolution
        else:
            if min_abs_value < 0:
                raise ValueError('min_abs_value {} has to be greater or equal zero.'.format(min_abs_value))
            if min_abs_value < dtype_resolution:
                matrix.logger.warning('Setting min_abs_value to resolution {} of matrix data type {}.'.format(dtype_resolution, A.dtype))
                min_abs_value = dtype_resolution

        # apply min_abs_value
        if min_abs_value > 0:
            for i in range(A.shape[0]):
                A_ii = A[i, i]
                if np.iscomplex(A_ii) and np.abs(A_ii.imag) < min_abs_value:
                    A_ii = A_ii.real
                    A[i, i] = A_ii

    return A
