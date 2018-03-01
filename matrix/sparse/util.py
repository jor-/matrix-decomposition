import warnings

import numpy as np
import scipy.sparse
import scipy.sparse.linalg

import matrix
import matrix.errors
import matrix.dense.util
import matrix.sparse.util


def is_sparse(A):
    return scipy.sparse.issparse(A)


def is_equal(A, B):
    return A.shape == B.shape and A.nnz == B.nnz and (A != B).nnz == 0


def is_almost_equal(A, B, rtol=1e-05, atol=1e-08):
    if A.shape != B.shape:
        return False
    D = A - B
    if D.format not in ('coo', 'csc', 'csr'):
        D.tocsr(copy=False)
    return np.all(np.isclose(D.data, 0, rtol=rtol, atol=atol))


def is_finite(A):
    return np.all(np.isfinite(A.data))


def check_finite(A, check_finite=True):
    if check_finite and not is_finite(A):
        raise matrix.errors.MatrixNotFiniteError(A)


def convert_to_csc_or_csr(A, matrix_format, warn_if_wrong_format=True, sort_indices=False, eliminate_zeros=False, overwrite_A=False, copy=False):
    # check input
    matrix_format = matrix_format.upper()
    POSSIBLE_MATRIX_FORMATS = ('CSC', 'CSR')
    if matrix_format not in POSSIBLE_MATRIX_FORMATS:
        raise ValueError('Matrix format has to be in {} but it is {}.'.format(POSSIBLE_MATRIX_FORMATS, matrix_format))

    # convert to sparse matrix
    copied = False
    if not scipy.sparse.isspmatrix_csc(A):
        if warn_if_wrong_format:
            matrix.logger.warning('{} matrix format is required. Converting to CSC matrix format.'.format(matrix_format), scipy.sparse.SparseEfficiencyWarning)
        if matrix_format == 'CSC':
            A = scipy.sparse.csc_matrix(A)
        else:
            A = scipy.sparse.csr_matrix(A)
        copied = True

    # sort indices
    A_old = A
    A = matrix.sparse.util.sort_indices(A, sort_indices=sort_indices, overwrite_A=overwrite_A or copied)
    if A_old is not A:
        copied = True

    # eliminate zeros
    A_old = A
    A = matrix.sparse.util.eliminate_zeros(A, eliminate_zeros=eliminate_zeros, overwrite_A=overwrite_A or copied)
    if A_old is not A:
        copied = True

    # copy
    if copy and not copied:
        A = A.copy()

    # return
    return A


def convert_to_csc(A, warn_if_wrong_format=True, sort_indices=False, eliminate_zeros=False, overwrite_A=False, copy=False):
    return convert_to_csc_or_csr(A, 'CSC', warn_if_wrong_format=warn_if_wrong_format, sort_indices=sort_indices, eliminate_zeros=eliminate_zeros, overwrite_A=overwrite_A, copy=copy)


def convert_to_csr(A, warn_if_wrong_format=True, sort_indices=False, eliminate_zeros=False, overwrite_A=False, copy=False):
    return convert_to_csc_or_csr(A, 'CSR', warn_if_wrong_format=warn_if_wrong_format, sort_indices=sort_indices, eliminate_zeros=eliminate_zeros, overwrite_A=overwrite_A, copy=copy)


def sort_indices(A, sort_indices=True, overwrite_A=False):
    if sort_indices:
        if not A.has_sorted_indices:
            if not overwrite_A:
                A = A.copy()
            A.sort_indices()
    return A


def eliminate_zeros(A, eliminate_zeros=True, overwrite_A=False):
    if eliminate_zeros:
        if not overwrite_A:
            A = A.copy()
        A.eliminate_zeros()
    return A


def convert_index_dtype(A, dtype, overwrite_A=False):
    if not (scipy.sparse.isspmatrix_csc(A) or scipy.sparse.isspmatrix_csr(A)):
        raise NotImplementedError("Only CSR and CSC are supported yet.")
    if A.indices.dtype != dtype or A.indptr.dtype != dtype:
        if not overwrite_A:
            A = A.copy()
        A.indices = np.asanyarray(A.indices, dtype=dtype)
        A.indptr = np.asanyarray(A.indptr, dtype=dtype)
    return A


def set_diagonal(A, diagonal_value):
    assert A.ndim == 2
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', scipy.sparse.SparseEfficiencyWarning)
        for i in range(min(A.shape)):
            A[i, i] = diagonal_value


def solve_triangular(A, b, lower=True, unit_diagonal=False, overwrite_b=False, check_finite=True):
    if check_finite:
        matrix.sparse.util.check_finite(A)
        if is_sparse(b):
            matrix.sparse.util.check_finite(b)
        else:
            matrix.dense.util.check_finite(b)
    A = A.tocsr(copy=True)
    if unit_diagonal:
        set_diagonal(A, 1)
    b = b.astype(np.result_type(A.data, b, np.float), copy=not overwrite_b)  # this has to be done due to a bug in scipy (see pull reqeust #7449)
    return scipy.sparse.linalg.spsolve_triangular(A, b, lower=lower, overwrite_A=True, overwrite_b=overwrite_b)


def compressed_matrix_indices(A, i, A_i_start_index=None, A_i_stop_index=None, A_ii_index=None, A_ii=None):
    if A_i_start_index is None:
        A_i_start_index = A.indptr[i]
    if A_i_stop_index is None:
        A_i_stop_index = A.indptr[i + 1]
    assert A_i_stop_index >= A_i_start_index

    if A_ii_index is None:
        A_ii_index_mask = np.where(A.indices[A_i_start_index:A_i_stop_index] == i)[0]
        if len(A_ii_index_mask) == 1:
            A_ii_index = A_i_start_index + A_ii_index_mask[0]
            assert A_i_start_index <= A_ii_index <= A_i_stop_index
        else:
            assert len(A_ii_index_mask) == 0
            A_ii_index = None

    if A_ii is None:
        if A_ii_index is not None:
            A_ii = A.data[A_ii_index]
        else:
            A_ii = 0

    return A_i_start_index, A_i_stop_index, A_ii_index, A_ii
