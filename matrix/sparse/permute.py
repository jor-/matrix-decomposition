import numpy as np
import scipy.sparse

import matrix
import matrix.permute


def _indices_of_compressed_matrix(A, p):
    if p is not None:
        # chose same dtype of indices and indptr
        try:
            p = p.astype(A.indices.dtype, casting='safe')
        except TypeError:
            A.indptr = A.indptr.astype(p.dtype, casting='safe')

        # apply permutation
        p_inverse = matrix.permute.invert_permutation_vector(p)
        A.indices = p_inverse[A.indices]
        A.has_sorted_indices = False
    return A


def rows(A, p, inplace=False, warn_if_wrong_format=True):
    if p is not None:
        if not scipy.sparse.isspmatrix_csc(A):
            if warn_if_wrong_format:
                matrix.logger.warning('CSC matrix format is required. Converting to CSC matrix format.', scipy.sparse.SparseEfficiencyWarning)
            A = scipy.sparse.csc_matrix(A)
        elif not inplace:
            A = A.copy()
        A = _indices_of_compressed_matrix(A, p)
    return A


def colums(A, p, inplace=False, warn_if_wrong_format=True):
    if p is not None:
        if not scipy.sparse.isspmatrix_csr(A):
            if warn_if_wrong_format:
                matrix.logger.warning('CSR matrix format is required. Converting to CSC matrix format.', scipy.sparse.SparseEfficiencyWarning)
            A = scipy.sparse.csr_matrix(A)
        elif not inplace:
            A = A.copy()
        A = _indices_of_compressed_matrix(A, p)
    return A


def symmetric(A, p, inplace=False, warn_if_wrong_format=True):
    """ Permute symmetrically a matrix.

    Parameters
    ----------
    A : scipy.sparse.spmatrix
        The matrix that should be permuted.
        It must have the shape (m, m).
    p : numpy.ndarray
        The permutation vector.
        It must have the shape (m,).
    inplace : bool
        Whether the permutation should be done inplace or not.
        optional, default : False
    warn_if_wrong_format : bool
        Whether the warn if the matrix `A` is not in the needed sparse format.
        optional, default : True

    Returns
    -------
    numpy.ndarray
        The matrix `A` symmetrically permuted by the permutation vector `p`.
        For the returned matrix `B` holds for all i, j in range(m):
        B[i,j] == A[p[i],p[j]]
        It has the shape (m, m).
    """

    if p is not None:
        if scipy.sparse.isspmatrix_csc(A):
            A = rows(A, p, inplace=inplace, warn_if_wrong_format=False)
            A = colums(A, p, inplace=True, warn_if_wrong_format=False)
        else:
            if warn_if_wrong_format and not scipy.sparse.isspmatrix_csr(A):
                matrix.logger.warning('CSC or CSR matrix format is required. Converting to needed matrix format.', scipy.sparse.SparseEfficiencyWarning)
            A = colums(A, p, inplace=inplace, warn_if_wrong_format=False)
            A = rows(A, p, inplace=True, warn_if_wrong_format=False)
    return A


def fill_reducing_permutation_vector(A, permutation_method=None, use_long=False):
    """
    Computes a permutation vector for a fill reducing permutation mwthod.

    Parameters
    ----------
    A : scipy.sparse.spmatrix
        Matrix that is supposed to be decomposed.
        It is assumed, that A is Hermitian.
        The matrix must be a squared matrix.
    permutation_method : str
        The symmetric permutation method that is applied to the matrix before
        it is decomposed. It has to be a value in
        :const:`matrix.sparse.constants.FILL_REDUCE_PERMUTATION_METHODS`.
        optional, default: no permutation
    use_long: bool
        Specifies if the long type (64 bit) or the int type (32 bit)
        should be used for the indices of the sparse matrices.
        If use_long is None try to estimate if long type is needed.
        optional, default: False

    Returns
    -------
    numpy.ndarray
        The permutation vector.
    """

    # check matrix A
    matrix.util.check_square_matrix(A)

    # if no permutation
    if permutation_method is None:
        return np.arange(A.shape[0])

    # apply supported permutation method
    elif permutation_method in matrix.sparse.constants.FILL_REDUCE_PERMUTATION_METHODS:

        # convert permutation method for cholmod
        assert permutation_method in matrix.sparse.constants.FILL_REDUCE_PERMUTATION_METHODS
        assert permutation_method.startswith(matrix.sparse.constants.FILL_REDUCE_PERMUTATION_METHOD_PREFIX)
        permutation_method = permutation_method[len(matrix.sparse.constants.FILL_REDUCE_PERMUTATION_METHOD_PREFIX):]
        assert permutation_method in matrix.sparse.constants.CHOLMOD_PERMUTATION_METHODS

        # try to import cholmod
        try:
            import sksparse.cholmod
        except ImportError as e:
            raise Exception('scikit-sparse is not installed.') from e

        # calculate permutation vector
        f = sksparse.cholmod.analyze(A, mode='simplicial', ordering_method=permutation_method, use_long=use_long)
        p = f.P()
        assert np.all(np.sort(p) == np.arange(len(p)))
        return p

    # unsupported permutation method
    else:
        raise ValueError('Permutation method {} is unknown. Only the following methods are supported {}.'.format(permutation_method, matrix.sparse.constants.FILL_REDUCE_PERMUTATION_METHODS))
