import numpy as np
import scipy.sparse

import matrix
import matrix.permute


def _permute_indices(indices, p):
    if p is not None:
        # make p as same array type as indices
        p = np.asarray(p, dtype=indices.dtype)
        # apply permutation
        p_inverse = matrix.permute.invert_permutation_vector(p)
        assert p_inverse.dtype == indices.dtype
        indices = p_inverse[indices]
    return indices


def _indices_of_compressed_matrix(A, p):
    if p is not None:
        A.indices = _permute_indices(A.indices, p)
        A.has_sorted_indices = False
    return A


def rows(A, p, inplace=False, warn_if_wrong_format=True):
    if p is not None:
        is_coo = scipy.sparse.isspmatrix_coo(A)
        is_csc = scipy.sparse.isspmatrix_csc(A)
        # convert and copy
        if not is_coo or is_csc:
            if warn_if_wrong_format:
                matrix.logger.warning('COO or CSC matrix format is required. Converting to CSC matrix format.', scipy.sparse.SparseEfficiencyWarning)
            A = scipy.sparse.csc_matrix(A)
            is_csc = True
        elif not inplace:
            A = A.copy()
        # permute
        if is_coo:
            A.row = _permute_indices(A.row, p)
        else:
            assert is_csc
            A = _indices_of_compressed_matrix(A, p)
    return A


def colums(A, p, inplace=False, warn_if_wrong_format=True):
    if p is not None:
        is_coo = scipy.sparse.isspmatrix_coo(A)
        is_csr = scipy.sparse.isspmatrix_csr(A)
        # convert and copy
        if not is_coo or is_csr:
            if warn_if_wrong_format:
                matrix.logger.warning('COO or CSR matrix format is required. Converting to CSR matrix format.', scipy.sparse.SparseEfficiencyWarning)
            A = scipy.sparse.csr_matrix(A)
            is_csr = True
        elif not inplace:
            A = A.copy()
        # permute
        if is_coo:
            A.col = _permute_indices(A.col, p)
        else:
            assert is_csr
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
        if not scipy.sparse.isspmatrix_coo(A):
            if warn_if_wrong_format:
                matrix.logger.warning('COO matrix format is required. Converting to needed matrix format.', scipy.sparse.SparseEfficiencyWarning)

        if scipy.sparse.isspmatrix_csc(A):
            A = rows(A, p, inplace=inplace, warn_if_wrong_format=False)
            A = colums(A, p, inplace=True, warn_if_wrong_format=False)
        else:
            A = colums(A, p, inplace=inplace, warn_if_wrong_format=False)
            A = rows(A, p, inplace=True, warn_if_wrong_format=False)
    return A


def fill_reducing_permutation_vector(A, permutation_method=None, use_long=None):
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
        optional, default: None

    Returns
    -------
    numpy.ndarray
        The permutation vector.
    """

    # check matrix A
    matrix.util.check_square_matrix(A)

    # perpare permutaiton method
    if permutation_method is not None:
        permutation_method = permutation_method.lower()
    else:
        permutation_method = matrix.constants.NO_PERMUTATION_METHOD

    # if no permutation
    if permutation_method == matrix.constants.NO_PERMUTATION_METHOD:
        n = A.shape[0]
        return np.arange(n, dtype=np.min_scalar_type(n))

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
            raise ImportError('scikit-sparse is not installed.') from e

        # convert to csc matrix
        if not scipy.sparse.isspmatrix_csc(A):
            matrix.logger.warning('CSC matrix format is required. Converting to CSC matrix format.', scipy.sparse.SparseEfficiencyWarning)
            A = A.tocsc(copy=False)

        # calculate permutation vector
        try:
            f = sksparse.cholmod.analyze(A, mode='simplicial', ordering_method=permutation_method, use_long=use_long)
        except sksparse.cholmod.CholmodTooLargeError as cholmod_exception:
            if A.indices.dtype != np.int64 or A.indptr.dtype != np.int64:
                matrix.logger.warning('Problem to large for index type. Index type is switched to long.')
                A = matrix.sparse.util.convert_index_dtype(A, np.int64, overwrite_A=True)
                f = sksparse.cholmod.analyze(A, mode='simplicial', ordering_method=permutation_method, use_long=True)
            else:
                matrix.logger.error(cholmod_exception)
                raise

        p = f.P()
        assert np.all(np.sort(p) == np.arange(len(p)))
        return p

    # unsupported permutation method
    else:
        raise ValueError('Permutation method {} is unknown. Only the following methods are supported {}.'.format(permutation_method, matrix.sparse.constants.FILL_REDUCE_PERMUTATION_METHODS))
