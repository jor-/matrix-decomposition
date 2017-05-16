import numpy as np

import matrix.dense.permute
import matrix.sparse.permute
import matrix.sparse.util


def invert_permutation_vector(p):
    p_inverse = np.empty_like(p)
    for i in range(len(p)):
        p_inverse[p[i]] = i
    return p_inverse


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
