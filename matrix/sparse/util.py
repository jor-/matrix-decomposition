import warnings

import numpy as np
import scipy.sparse

import matrix.errors
import matrix.sparse.util


def is_sparse(A):
    return scipy.sparse.issparse(A)


def equal(A, B):
    return A.shape == B.shape and (A != B).nnz == 0


def check_finite_matrix(A):
    if not np.all(np.isfinite(A.data)):
        raise matrix.errors.MatrixNotFiniteError(matrix=A)


def convert_to_csc(A, warn_if_wrong_format=True, sort_indices=False, eliminate_zeros=False):
    if not scipy.sparse.isspmatrix_csc(A):
        if warn_if_wrong_format:
            warnings.warn('CSC matrix format is required. Converting to CSC matrix format.', scipy.sparse.SparseEfficiencyWarning)
        A = scipy.sparse.csc_matrix(A)
    A = matrix.sparse.util.sort_indices(A, sort_indices=sort_indices)
    A = matrix.sparse.util.eliminate_zeros(A, eliminate_zeros=eliminate_zeros)
    return A


def convert_to_csr(A, warn_if_wrong_format=True, sort_indices=False, eliminate_zeros=False):
    if not scipy.sparse.isspmatrix_csr(A):
        if warn_if_wrong_format:
            warnings.warn('CSR matrix format is required. Converting to CSR matrix format.', scipy.sparse.SparseEfficiencyWarning)
        A = scipy.sparse.csr_matrix(A)
    A = matrix.sparse.util.sort_indices(A, sort_indices=sort_indices)
    A = matrix.sparse.util.eliminate_zeros(A, eliminate_zeros=eliminate_zeros)
    return A


def sort_indices(A, sort_indices=True):
    if sort_indices:
        A.sort_indices()
    return A


def eliminate_zeros(A, eliminate_zeros=True):
    if eliminate_zeros:
        A.eliminate_zeros()
    return A


def convert_index_dtype(A, dtype):
    if not (scipy.sparse.isspmatrix_csc(A) or scipy.sparse.isspmatrix_csr(A)):
        raise NotImplementedError("Only CSR and CSC are supported yet.")
    A.indices = np.asanyarray(A.indices, dtype=dtype)
    A.indptr = np.asanyarray(A.indptr, dtype=dtype)
    return A
