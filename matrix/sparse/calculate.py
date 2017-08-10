import warnings

import numpy as np

import matrix.sparse.constants
import matrix.sparse.permute
import matrix.sparse.util

import matrix.constants
import matrix.decompositions
import matrix.errors
import matrix.permute
import matrix.util


def _decompose(A, permutation_method=None, return_type=None, check_finite=True, overwrite_A=False, use_long=False):
    """
    Computes a decomposition of a sparse matrix.

    Parameters
    ----------
    A : scipy.sparse.spmatrix
        Matrix to be decomposed.
        It is assumed, that A is Hermitian.
        The matrix must be a squared matrix.
    permutation_method : str
        The symmetric permutation method that is applied to the matrix before
        it is decomposed. It has to be a value in
        :const:`matrix.sparse.constants.PERMUTATION_METHODS`.
        optional, default: no permutation
    return_type : str
        The type of the decomposition that should be calculated.
        It has to be a value in :const:`matrix.constants.DECOMPOSITION_TYPES`.
        If return_type is None the type of the returned decomposition is
        chosen by the function itself.
        optional, default: the type of the decomposition is chosen by the function itself
    check_finite : bool
        Whether to check that the input matrix contains only finite numbers.
        Disabling may result in problems (crashes, non-termination)
        if the inputs do contain infinities or NaNs.
        Disabling gives a performance gain.
        optional, default: True
    overwrite_A : bool
        Whether it is allowed to overwrite A.
        Enabling may result in performance gain.
        optional, default: False
    use_long: bool
        Specifies if the long type (64 bit) or the int type (32 bit)
        should be used for the indices of the sparse matrices.
        If use_long is None try to estimate if long type is needed.
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
        If `A` is not a finte matrix and `check_finite` is True.
    """
    try:
        import sksparse.cholmod
    except ImportError as e:
        raise Exception('scikit-sparse is not installed.') from e

    # check matrix A
    matrix.util.check_square_matrix(A)

    # check and apply permutation_method
    if permutation_method is not None:
        permutation_method = permutation_method.lower()
    supported_permutation_methods = matrix.sparse.constants.PERMUTATION_METHODS
    if permutation_method not in supported_permutation_methods:
        raise ValueError('Permutation method {} is unknown. Only the following methods are supported {}.'.format(permutation_method, supported_permutation_methods))

    if permutation_method in matrix.constants.PERMUTATION_METHODS:
        if permutation_method in matrix.constants.NO_PERMUTATION_METHODS:
            p = None
        else:
            p = matrix.permute.permutation_vector(A, permutation_method)
            A = matrix.sparse.permute.symmetric(A, p)
            A = A.tocsc()
        permutation_method = 'natural'
    else:
        assert permutation_method in matrix.sparse.constants.CHOLMOD_PERMUTATION_METHODS

    # check return type
    supported_return_type = matrix.sparse.constants.DECOMPOSITION_TYPES
    if return_type is not None and return_type not in supported_return_type:
        raise ValueError('Unkown decomposition type {}. Only values in {} are supported.'.format(return_type, supported_return_type))

    # convert matrix A
    A_old = A
    A = matrix.sparse.util.convert_to_csc(A, sort_indices=True, eliminate_zeros=True, overwrite_A=overwrite_A)
    overwrite_A = overwrite_A or A_old is not A
    if use_long:
        A = matrix.sparse.util.convert_index_dtype(A, np.int64, overwrite_A=overwrite_A)
    matrix.sparse.util.check_finite(A, check_finite)

    # calculate decomposition
    try:
        try:
            f = sksparse.cholmod.cholesky(A, ordering_method=permutation_method, use_long=use_long)
        except sksparse.cholmod.CholmodTooLargeError as cholmod_exception:
            raise matrix.errors.NoDecompositionPossibleTooManyEntriesError(
                A, matrix.constants.LDL_DECOMPOSITION_COMPRESSED_TYPE) from cholmod_exception
    except sksparse.cholmod.CholmodNotPositiveDefiniteError as e:
        cholmod_exception = e
        f = cholmod_exception.factor
    else:
        cholmod_exception = None

    # get correct permutation vector
    if permutation_method != 'natural':
        p = f.P()
    else:
        assert np.all(f.P() == np.arange(len(f.P())))

    # make docomposition
    decomposition = matrix.decompositions.LDL_DecompositionCompressed(f.LD(), p=p)

    # check exception
    if cholmod_exception is not None:
        bad_index = cholmod_exception.column
        decomposition.LD[bad_index, bad_index] = np.nan
        raise matrix.errors.NoDecompositionPossibleWithProblematicSubdecompositionError(
            A, matrix.constants.LDL_DECOMPOSITION_COMPRESSED_TYPE, bad_index, decomposition) from cholmod_exception

    # return
    return decomposition.as_type(return_type)


def decompose(A, permutation_method=None, return_type=None, check_finite=True, overwrite_A=False):
    """
    Computes a decomposition of a sparse matrix.

    Parameters
    ----------
    A : scipy.sparse.spmatrix
        Matrix to be decomposed.
        It is assumed, that A is Hermitian.
        The matrix must be a squared matrix.
    permutation_method : str
        The symmetric permutation method that is applied to the matrix before
        it is decomposed. It has to be a value in
        :const:`matrix.sparse.constants.PERMUTATION_METHODS`.
        optional, default: no permutation
    return_type : str
        The type of the decomposition that should be calculated.
        It has to be a value in :const:`matrix.constants.DECOMPOSITION_TYPES`.
        If return_type is None the type of the returned decomposition is
        chosen by the function itself.
        optional, default: the type of the decomposition is chosen by the function itself
    check_finite : bool
        Whether to check that the input matrix contains only finite numbers.
        Disabling may result in problems (crashes, non-termination)
        if the inputs do contain infinities or NaNs.
        Disabling gives a performance gain.
        optional, default: True
    overwrite_A : bool
        Whether it is allowed to overwrite A.
        Enabling may result in performance gain.
        optional, default: False

    Returns
    -------
    matrix.decompositions.DecompositionBase
        A decomposition of `A`. If `return_type` is not None, the decomposition
        is of this type.

    Raises
    ------
    matrix.errors.NoDecompositionPossibleError
        If the decomposition of `A` is not possible.
    matrix.errors.MatrixNotSquareError
        If `A` is not a square matrix.
    matrix.errors.MatrixNotFiniteError
        If `A` is not a finte matrix and `check_finite` is True.
    """

    try:
        return _decompose(A, permutation_method=permutation_method, return_type=return_type, check_finite=check_finite, overwrite_A=overwrite_A, use_long=False)
    except matrix.errors.NoDecompositionPossibleTooManyEntriesError as e:
        warnings.warn('Problem to large for index type {}, index type is switched to long.'.format(e.matrix_index_type))
        return _decompose(A, permutation_method=permutation_method, return_type=return_type, check_finite=False, overwrite_A=overwrite_A, use_long=True)
