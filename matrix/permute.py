import numpy as np

import matrix.constants
import matrix.dense.permute
import matrix.sparse.permute
import matrix.sparse.util


def permutation_vector(A, permutation_method=None):
    if permutation_method is not None:
        permutation_method = permutation_method.lower()
    supported_permutation_methods = matrix.constants.PERMUTATION_METHODS
    if permutation_method not in supported_permutation_methods:
        raise ValueError('Permutation method {} is unknown. Only the following permutation_methods are supported {}.'.format(permutation_method, supported_permutation_methods))

    if permutation_method in matrix.constants.NO_PERMUTATION_METHODS:
        return None
    else:
        d = A.diagonal()
        if isinstance(d, np.matrix):
            d = d.A1
        if permutation_method in (matrix.constants.INCREASING_ABSOLUTE_DIAGONAL_VALUES_PERMUTATION_METHOD, matrix.constants.DECREASING_ABSOLUTE_DIAGONAL_VALUES_PERMUTATION_METHOD):
            d = np.abs(d)
        if permutation_method in (matrix.constants.INCREASING_DIAGONAL_VALUES_PERMUTATION_METHOD, matrix.constants.DECREASING_DIAGONAL_VALUES_PERMUTATION_METHOD):
            if np.iscomplexobj(d):
                if np.all(np.isreal(d)):
                    d = d.real
                else:
                    raise matrix.errors.MatrixComplexDiagonalValueError(A)
        if permutation_method in (matrix.constants.DECREASING_DIAGONAL_VALUES_PERMUTATION_METHOD, matrix.constants.DECREASING_ABSOLUTE_DIAGONAL_VALUES_PERMUTATION_METHOD):
            d = -d
        p = np.argsort(d)
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
