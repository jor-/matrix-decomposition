import matrix
import matrix.constants
import matrix.errors
import matrix.permute
import matrix.util
import matrix.dense.calculate
import matrix.sparse.calculate
import matrix.sparse.util


def decompose(A, permutation=None, return_type=None, check_finite=True, overwrite_A=False):
    """
    Computes a decomposition of a matrix.

    Parameters
    ----------
    A : numpy.ndarray or scipy.sparse.spmatrix
        Matrix to be decomposed.
        `A` must be Hermitian.
    permutation : str or numpy.ndarray
        The symmetric permutation method that is applied to the matrix before
        it is decomposed. It has to be a value in
        :const:`matrix.UNIVERSAL_PERMUTATION_METHODS`.
        If `A` is sparse, it can also be a value in
        :const:`matrix.SPARSE_ONLY_PERMUTATION_METHODS`.
        It is also possible to directly pass a permutation vector.
        optional, default: no permutation
    return_type : str
        The type of the decomposition that should be calculated.
        It has to be a value in :const:`matrix.DECOMPOSITION_TYPES`.
        If return_type is None the type of the returned decomposition is
        chosen by the function itself.
        optional, default: the type of the decomposition is chosen by the function itself
    check_finite : bool
        Whether to check that `A` contains only finite numbers.
        Disabling may result in problems (crashes, non-termination)
        if the inputs do contain infinities or NaNs.
        Disabling gives a performance gain.
        optional, default: True
    overwrite_A : bool
        Whether it is allowed to overwrite `A`.
        Enabling may result in performance gain.
        optional, default: False

    Returns
    -------
    matrix.decompositions.DecompositionBase
        A decomposition of `A` of type `return_type`.

    Raises
    ------
    matrix.errors.NoDecompositionPossibleError
        If the decomposition of `A` is not possible.
    matrix.errors.MatrixNotSquareError
        If `A` is not a square matrix.
    matrix.errors.MatrixNotFiniteError
        If `A` is not a finite matrix and `check_finite` is True.
    """

    # debug logging
    matrix.logger.debug('Decomposing matrix with permutation={permutation}, return_type={return_type}, check_finite={check_finite}, overwrite_A={overwrite_A}.'.format(
        permutation=permutation,
        return_type=return_type,
        check_finite=check_finite,
        overwrite_A=overwrite_A))

    # decompose
    if matrix.sparse.util.is_sparse(A):
        decomposition = matrix.sparse.calculate.decompose(A, permutation=permutation, return_type=return_type, check_finite=check_finite, overwrite_A=overwrite_A)
    else:
        decomposition = matrix.dense.calculate.decompose(A, permutation=permutation, return_type=return_type, check_finite=check_finite, overwrite_A=overwrite_A)

    # return
    matrix.logger.debug('Decomposing matrix finished.')
    return decomposition


def is_positive_semidefinite(A, check_finite=True):
    """
    Returns whether the passed matrix is positive semi-definite.

    Parameters
    ----------
    A : numpy.ndarray or scipy.sparse.spmatrix
        The matrix that should be checked.
        `A` must be Hermitian.
    check_finite : bool
        Whether to check that `A` contain only finite numbers.
        Disabling may result in problems (crashes, non-termination)
        if they contain infinities or NaNs.
        Disabling gives a performance gain.
        optional, default: True

    Returns
    -------
    bool
        Whether `A` is positive semi-definite.

    Raises
    ------
    matrix.errors.MatrixNotFiniteError
        If `A` is not a finite matrix and `check_finite` is True.
    """

    # debug logging
    matrix.logger.debug('Checking whether matrix is positive semi-definite with check_finite={check_finite}.'.format(
        check_finite=check_finite))

    # try to decompose and check decomposition
    try:
        decomposition = decompose(A,
                                  permutation=matrix.constants.NO_PERMUTATION_METHOD,
                                  check_finite=check_finite)
    except (matrix.errors.NoDecompositionPossibleError,
            matrix.errors.MatrixComplexDiagonalValueError,
            matrix.errors.MatrixNotFiniteError,
            matrix.errors.MatrixNotSquareError):
        return False
    else:
        return decomposition.is_positive_semidefinite()


def is_positive_definite(A, check_finite=True):
    """
    Returns whether the passed matrix is positive definite.

    Parameters
    ----------
    A : numpy.ndarray or scipy.sparse.spmatrix
        The matrix that should be checked.
        `A` must be Hermitian.
    check_finite : bool
        Whether to check that `A` contain only finite numbers.
        Disabling may result in problems (crashes, non-termination)
        if they contain infinities or NaNs.
        Disabling gives a performance gain.
        optional, default: True

    Returns
    -------
    bool
        Whether `A` is positive definite.

    Raises
    ------
    matrix.errors.MatrixNotFiniteError
        If `A` is not a finite matrix and `check_finite` is True.
    """

    # debug logging
    matrix.logger.debug('Checking whether matrix is positive definite with check_finite={check_finite}.'.format(
        check_finite=check_finite))

    # try to decompose and check decomposition
    try:
        decomposition = decompose(A,
                                  permutation=matrix.constants.NO_PERMUTATION_METHOD,
                                  check_finite=check_finite)
    except (matrix.errors.NoDecompositionPossibleError,
            matrix.errors.MatrixComplexDiagonalValueError,
            matrix.errors.MatrixNotFiniteError,
            matrix.errors.MatrixNotSquareError):
        return False
    else:
        return decomposition.is_positive_definite()


def is_invertible(A, check_finite=True):
    """
    Returns whether the passed matrix is an invertible matrix.

    Parameters
    ----------
    A : numpy.ndarray or scipy.sparse.spmatrix
        The matrix that should be checked.
        `A` must be Hermitian and positive semidefinite.
    check_finite : bool
        Whether to check that `A` contain only finite numbers.
        Disabling may result in problems (crashes, non-termination)
        if they contain infinities or NaNs.
        Disabling gives a performance gain.
        optional, default: True

    Returns
    -------
    bool
        Whether `A` is invertible.

    Raises
    ------
    matrix.errors.MatrixNotFiniteError
        If `A` is not a finite matrix and `check_finite` is True.
    """

    # debug logging
    matrix.logger.debug('Checking whether matrix is invertable with check_finite={check_finite}.'.format(
        check_finite=check_finite))

    # try to decompose and check decomposition
    try:
        decomposition = decompose(A,
                                  permutation=matrix.constants.NO_PERMUTATION_METHOD,
                                  check_finite=check_finite)
    except (matrix.errors.MatrixNotFiniteError,
            matrix.errors.MatrixNotSquareError):
        return False
    else:
        return decomposition.is_invertible()


def solve(A, b, overwrite_b=False, check_finite=True):
    """
    Solves the equation `A x = b` regarding `x`.

    Parameters
    ----------
    A : numpy.ndarray or scipy.sparse.spmatrix
        The matrix that should be checked.
        `A` must be Hermitian and positive definite.
    b : numpy.ndarray
        Right-hand side vector or matrix in equation `A x = b`.
        It must hold `b.shape[0] == A.shape[0]`.
    overwrite_b : bool
        Allow overwriting data in `b`.
        Enabling gives a performance gain.
        optional, default: False
    check_finite : bool
        Whether to check that `A` and b` contain only finite numbers.
        Disabling may result in problems (crashes, non-termination)
        if they contain infinities or NaNs.
        Disabling gives a performance gain.
        optional, default: True

    Returns
    -------
    numpy.ndarray
        An `x` so that `A x = b`.
        The shape of `x` matches the shape of `b`.

    Raises
    ------
    matrix.errors.MatrixNotSquareError
        If `A` is not a square matrix.
    matrix.errors.MatrixNotFiniteError
        If `A` is not a finite matrix and `check_finite` is True.
    matrix.errors.MatrixSingularError
        If `A` is singular.
    """

    # debug logging
    matrix.logger.debug('Solving linear system with overwrite_b={overwrite_b} check_finite={check_finite}.'.format(
        overwrite_b=overwrite_b,
        check_finite=check_finite))

    # try to decompose and solve with decomposition
    decomposition = decompose(A, check_finite=check_finite)
    try:
        return decomposition.solve(b, overwrite_b=overwrite_b, check_finite=False)
    except matrix.errors.DecompositionSingularError as base_error:
        error = matrix.errors.MatrixSingularError(A)
        matrix.logger.error(error)
        raise error from base_error
