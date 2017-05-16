import numpy as np


def symmetric(A, p):
    """ Permute symmetrically a matrix.

    Parameters
    ----------
    A : numpy.ndarray (with shape (m, m))
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
        A = A[p[:, np.newaxis], p[np.newaxis, :]]
    return A
