import numpy as np

import matrix
import matrix.constants
import matrix.dense.permute
import matrix.sparse.permute
import matrix.sparse.util


def permutation_vector(A, permutation_method=None):
    """
    Computes a permutation vector for a permutation method.

    Parameters
    ----------
    A : numpy.ndarray or scipy.sparse.spmatrix
        Matrix to be decomposed.
        It is assumed, that A is Hermitian.
        The matrix must be a squared matrix.
    permutation_method : str
        The symmetric permutation method that is applied to the matrix before
        it is decomposed. It has to be a value in
        :const:`matrix.UNIVERSAL_PERMUTATION_METHODS`.
        If `A` is sparse, it can also be a value in
        :const:`matrix.SPARSE_ONLY_PERMUTATION_METHODS`.
        optional, default: no permutation

    Returns
    -------
    numpy.ndarray
        The permutation vector.
    """

    matrix.logger.debug(('Calculating permutation vector with method "{}".'
                         '').format(permutation_method))

    DIAGONAL_VALUES_PERMUATION_METHODS = (
        matrix.constants.INCREASING_ABSOLUTE_DIAGONAL_VALUES_PERMUTATION_METHOD,
        matrix.constants.DECREASING_ABSOLUTE_DIAGONAL_VALUES_PERMUTATION_METHOD,
        matrix.constants.INCREASING_DIAGONAL_VALUES_PERMUTATION_METHOD,
        matrix.constants.DECREASING_DIAGONAL_VALUES_PERMUTATION_METHOD)

    if permutation_method == matrix.constants.NO_PERMUTATION_METHOD:
        n = A.shape[0]
        p = np.arange(n, dtype=np.min_scalar_type(n))
    elif permutation_method in DIAGONAL_VALUES_PERMUATION_METHODS:
        d = A.diagonal()
        if isinstance(d, np.matrix):
            d = d.A1
        if permutation_method in (
                matrix.constants.INCREASING_ABSOLUTE_DIAGONAL_VALUES_PERMUTATION_METHOD,
                matrix.constants.DECREASING_ABSOLUTE_DIAGONAL_VALUES_PERMUTATION_METHOD):
            d = np.abs(d)
        if permutation_method in (
                matrix.constants.INCREASING_DIAGONAL_VALUES_PERMUTATION_METHOD,
                matrix.constants.DECREASING_DIAGONAL_VALUES_PERMUTATION_METHOD):
            if np.iscomplexobj(d):
                if np.all(np.isreal(d)):
                    d = d.real
                else:
                    raise matrix.errors.MatrixComplexDiagonalValueError(A)
        if permutation_method in (
                matrix.constants.DECREASING_DIAGONAL_VALUES_PERMUTATION_METHOD,
                matrix.constants.DECREASING_ABSOLUTE_DIAGONAL_VALUES_PERMUTATION_METHOD):
            d = -d
        p = np.argsort(d)
    elif permutation_method in matrix.constants.SPARSE_ONLY_PERMUTATION_METHODS:
        if matrix.sparse.util.is_sparse(A):
            p = matrix.sparse.permute.fill_reducing_permutation_vector(
                A, permutation_method=permutation_method)
        else:
            error = ValueError(
                ('Permutation method {} is allowed only for sparse matrices.'
                 'But the matrix is dense. Only the following methods are supported '
                 'for dense matrices: {}.'
                 ).format(permutation_method, matrix.UNIVERSAL_PERMUTATION_METHODS))
            matrix.logger.error(error)
            raise error
    else:
        error = ValueError(
            ('Permutation method {} is unknown. Only the following methods are supported '
             'for dense and sparse matrices: {}. For sparse matrices the following methods '
             'are also supported: {}.'
             ).format(permutation_method, DIAGONAL_VALUES_PERMUATION_METHODS,
                      matrix.constants.SPARSE_ONLY_PERMUTATION_METHODS))
        matrix.logger.error(error)
        raise error
    return p


def invert_permutation_vector(p):
    p_inverse = np.empty_like(p)
    for i, p_i in enumerate(p):
        p_inverse[p_i] = i
    return p_inverse


def concatenate_permutation_vectors(p_previous, p_next):
    if p_previous is None:
        return p_next
    elif p_next is None:
        return p_previous
    else:
        return p_previous[p_next]


def symmetric(A, p):
    """ Permute symmetrically a matrix.

    Parameters
    ----------
    A : numpy.ndarray or scipy.sparse.spmatrix (with shape (m, m))
        The matrix that should be permuted.
    p : numpy.ndarray (with shape (m,))
        The permutation vector.

    Returns
    -------
    numpy.ndarray (with shape (m, m))
        The matrix `A` symmetrically permuted by the permutation vector `p`.
        For the returned matrix `B` holds for all i, j in range(m):
        B[i,j] == A[p[i],p[j]]
    """

    if p is not None:
        if matrix.sparse.util.is_sparse(A):
            return matrix.sparse.permute.symmetric(A, p)
        else:
            return matrix.dense.permute.symmetric(A, p)
    else:
        return A
