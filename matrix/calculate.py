import matrix.dense.calculate

import matrix.sparse.calculate
import matrix.sparse.util


def decompose(A, permutation_method=None, check_finite=True, return_type=None):
    """
    Computes a decomposition of a matrix.

    Parameters
    ----------
    A : numpy.ndarray or scipy.sparse.spmatrix
        Matrix to be decomposed.
        It is assumed, that A is Hermitian.
        The matrix must be a squared matrix.
    permutation_method : str
        The symmetric permutation method that is applied to the matrix before
        it is decomposed. It has to be a value in
        :const:`matrix.constants.PERMUTATION_METHODS`.
        If `A` is sparse, it can also be a value in
        :const:`matrix.sparse.constants.SPARSE_PERMUTATION_METHODS`.
        optional, default: no permutation
    check_finite : bool
        Whether to check that the input matrix contains only finite numbers.
        Disabling may result in problems (crashes, non-termination) if
        the inputs do contain infinities or NaNs.
        (disabling may improve performance)
        optional, default: True
    return_type : str
        The type of the decomposition that should be calculated.
        It has to be a value in :const:`matrix.constants.DECOMPOSITION_TYPES`.
        If return_type is None the type of the returned decomposition is
        chosen by the function itself.
        optional, default: the type of the decomposition is chosen by the function itself

    Returns
    -------
    matrix.decompositions.DecompositionBase
        A decompostion of `A` of type `return_type`.

    Raises
    ------
    matrix.errors.MatrixNoDecompositionPossibleError
        If the decomposition of `A` is not possible.
    """

    if matrix.sparse.util.is_sparse(A):
        return matrix.sparse.calculate.decompose(A, permutation_method=permutation_method, check_finite=check_finite, return_type=return_type)
    else:
        return matrix.dense.calculate.decompose(A, permutation_method=permutation_method, check_finite=check_finite, return_type=return_type)


def is_positive_semi_definite(A):
    """
    Checks if the passed matrix is positive semi-definite.

    Parameters
    ----------
    A : numpy.ndarray or scipy.sparse.spmatrix
        The matrix that should be checked.
        It is assumed, that A is Hermitian.
        The matrix must be a squared matrix.

    Returns
    -------
    bool
        Whether `A` is positive semi-definite.
    """

    try:
        decomposition = decompose(A, permutation_method=matrix.constants.INCREASING_DIAGONAL_VALUES_PERMUTATION_METHOD, check_finite=True)
    except matrix.errors.MatrixNoDecompositionPossibleError:
        return False
    else:
        return decomposition.is_positive_semi_definite()


def is_positive_definite(A):
    """
    Checks if the passed matrix is positive definite.

    Parameters
    ----------
    A : numpy.ndarray or scipy.sparse.spmatrix
        The matrix that should be checked.
        It is assumed, that A is Hermitian.
        The matrix must be a squared matrix.

    Returns
    -------
    bool
        Whether `A` is positive definite.
    """

    try:
        decomposition = decompose(A, permutation_method=matrix.constants.INCREASING_DIAGONAL_VALUES_PERMUTATION_METHOD, check_finite=True)
    except matrix.errors.MatrixNoDecompositionPossibleError:
        return False
    else:
        return decomposition.is_positive_definite()
