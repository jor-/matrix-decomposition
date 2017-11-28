import os
import warnings

import numpy as np
import scipy.sparse

import matrix
import matrix.constants
import matrix.errors
import matrix.permute
import matrix.util
import matrix.dense.calculate
import matrix.sparse.calculate
import matrix.sparse.util


def decompose(A, permutation_method=None, return_type=None, check_finite=True, overwrite_A=False):
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
        :const:`matrix.PERMUTATION_METHODS`.
        If `A` is sparse, it can also be a value in
        :const:`matrix.SPARSE_PERMUTATION_METHODS`.
        optional, default: no permutation
    return_type : str
        The type of the decomposition that should be calculated.
        It has to be a value in :const:`matrix.DECOMPOSITION_TYPES`.
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
    matrix.logger.debug('Decomposing matrix with permutation_method={permutation_method}, return_type={return_type}, check_finite={check_finite}, overwrite_A={overwrite_A}.'.format(
        permutation_method=permutation_method,
        return_type=return_type,
        check_finite=check_finite,
        overwrite_A=overwrite_A))

    # decompose
    if matrix.sparse.util.is_sparse(A):
        decomposition = matrix.sparse.calculate.decompose(A, permutation_method=permutation_method, return_type=return_type, check_finite=check_finite, overwrite_A=overwrite_A)
    else:
        decomposition = matrix.dense.calculate.decompose(A, permutation_method=permutation_method, return_type=return_type, check_finite=check_finite, overwrite_A=overwrite_A)

    # return
    matrix.logger.debug('Decomposing matrix finished.')
    return decomposition


def _approximate_decomposition_init(A, t=None, min_abs_value=None, copy=True):
    # convert input matrix A to needed type
    is_sparse = matrix.sparse.util.is_sparse(A)
    if not is_sparse:
        A_passed = A
        A = np.asarray(A)
        copied = A is not A_passed
    A_dtype = np.result_type(A.dtype, np.float)
    if is_sparse:
        A = A.astype(A_dtype)
    else:
        A = A.astype(A_dtype, copy=not copied and copy)

    # check input matrix A
    matrix.util.check_square_matrix(A)

    # apply min_abs_value to A
    A = matrix.util.set_nearly_zero_to_zero(A, min_abs_value=min_abs_value)

    # check target vector t
    n = A.shape[0]
    if t is not None:
        t = np.asanyarray(t)
        if t.ndim != 1:
            error = ValueError('t has to be a one-dimensional array.')
            matrix.logger.error(error)
            raise error
        if len(t) != n:
            error = ValueError('The length of t {} must have the same length as the dimensions of A {}.'.format(len(t), n))
            matrix.logger.error(error)
            raise error
        if np.iscomplexobj(t):
            if np.all(np.isreal(t)):
                t = t.real
            else:
                error = ValueError('t must have real values but they are complex.')
                matrix.logger.error(error)
                raise error

    return A, t


def approximate_decomposition(A, t=None, min_abs_value=None, min_diag_value=None, max_diag_value=None, permutation_method=None, return_type=None, check_finite=True, overwrite_A=False, callback=None):
    """
    Computes an approximative decomposition of a matrix.

    If `A` is decomposable in a decomposition of type `return_type`, this decomposition is returned.
    Otherwise a decomposition of type `return_type` is retuned which represents an approximation
    of `A`.

    Parameters
    ----------
    A : numpy.ndarray or scipy.sparse.spmatrix
        The matrix that should be approximated by a decomposition.
        It is assumed, that A is Hermitian.
        The matrix must be a squared matrix.
    t : numpy.ndarray
        The targed vector used for the approximation. For each i in range(M)
        `min_diag_value <= t[i] <= max_diag_value` must hold.
        `t` and `A` must have the same length.
        optional, default : The diagonal of `A` is used as `t`.
    min_abs_value : float
        Absolute values below `min_abs_value` are considered as zero.
        optional, default : The resolution of the underlying data type is used.
    min_diag_value : float
        Each component of the diagonal of the matrix `D` in an returned `LDL` decomposition
        is forced to be greater or equal to `min_diag_value`.
        optional, default : 0.
    max_diag_value : float
        Each component of the diagonal of the matrix `D` in an returned `LDL` decomposition
        is forced to be lower or equal to `max_diag_value`.
        optional, default : No maximal value is forced.
    permutation_method : str
        The symmetric permutation method that is applied to the matrix before
        it is decomposed. It has to be a value in
        :const:`matrix.PERMUTATION_METHODS`.
        If `A` is sparse, it can also be a value in
        :const:`matrix.SPARSE_PERMUTATION_METHODS`.
        optional, default: No permutation is done.
    return_type : str
        The type of the decomposition that should be calculated.
        It has to be a value in :const:`matrix.DECOMPOSITION_TYPES`.
        optional, default : The type of the decomposition is chosen by the function itself.
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
    callback : callable
        In each iteration `callback(i, r)` is called where `i` is the index of
        the row and column where components of `A` are reduced by the factor `r`.
        optional, default : No callback function is called.

    Returns
    -------
    matrix.decompositions.DecompositionBase
        An approximative decomposition of `A` of type `return_type`.

    Raises
    ------
    matrix.errors.MatrixNotSquareError
        If `A` is not a square matrix.
    matrix.errors.MatrixNotFiniteError
        If `A` is not a finite matrix and `check_finite` is True.
    """

    # debug logging
    matrix.logger.debug('Approximating decomposition of a matrix with min_abs_value={min_abs_value}, min_diag_value={min_diag_value}, max_diag_value={max_diag_value}, permutation_method={permutation_method}, return_type={return_type}, check_finite={check_finite}, overwrite_A={overwrite_A}.'.format(
        min_abs_value=min_abs_value,
        min_diag_value=min_diag_value,
        max_diag_value=max_diag_value,
        permutation_method=permutation_method,
        return_type=return_type,
        check_finite=check_finite,
        overwrite_A=overwrite_A))

    # init
    A, t = _approximate_decomposition_init(A, t=t, min_abs_value=min_abs_value, copy=not overwrite_A)
    is_sparse = matrix.sparse.util.is_sparse(A)
    n = A.shape[0]
    matrix.util.check_finite(A, check_finite=check_finite)

    # determine max_reduction_factor
    dtype_resolution = np.finfo(A.dtype).resolution
    max_reduction_factor = 1 - dtype_resolution * 10**2
    min_diag_value_LL = dtype_resolution * 10**2

    # check min_diag_value and max_diag_value
    if min_diag_value is None:
        min_diag_value = min_diag_value_LL
    else:
        if return_type == matrix.constants.LL_DECOMPOSITION_TYPE:
            if min_diag_value < 0:
                error = ValueError('If return type is {}, min_diag_value {} has to be greater or equal zero .'.format(return_type, min_diag_value))
                matrix.logger.error(error)
                raise error
            elif min_diag_value < min_diag_value_LL:
                matrix.logger.warning('Setting min_diag_value to resolution {} of matrix data type {}.'.format(min_diag_value, A.dtype))
        else:
            if min_diag_value <= 0:
                error = ValueError('Only min_diag_values greater zero are supported.')
                matrix.logger.error(error)
                raise error
            elif min_diag_value < min_diag_value_LL:
                matrix.logger.warning('Setting min_diag_value to resolution {} of matrix data type {}.'.format(min_diag_value, A.dtype))

    if max_diag_value is None:
        max_diag_value = np.inf
    if min_diag_value > max_diag_value:
        error = ValueError('min_diag_value {} has to be lower or equal to max_diag_value {}.'.format(min_diag_value, max_diag_value))
        matrix.logger.error(error)
        raise error

    # check return type
    supported_return_types = matrix.constants.DECOMPOSITION_TYPES
    if return_type is not None and return_type not in supported_return_types:
        error = ValueError('Unkown return type {}. Only values in {} are supported.'.format(return_type, supported_return_types))
        matrix.logger.error(error)
        raise error

    # check permutation method
    if permutation_method is not None:
        permutation_method = permutation_method.lower()
    if is_sparse:
        supported_permutation_methods = matrix.sparse.constants.PERMUTATION_METHODS
    else:
        supported_permutation_methods = matrix.dense.constants.PERMUTATION_METHODS
    if permutation_method not in supported_permutation_methods:
        error = ValueError('Permutation method {} is unknown. Only the following methods are supported {}.'.format(permutation_method, supported_permutation_methods))
        matrix.logger.error(error)
        raise error

    # prepare permutation
    p_previous = None
    if permutation_method in matrix.constants.PERMUTATION_METHODS:
        permutation_method_previous = permutation_method
        permutation_method_decomposite = None
    else:
        permutation_method_previous = None
        permutation_method_decomposite = permutation_method
        assert is_sparse and permutation_method in matrix.sparse.constants.FILL_REDUCE_PERMUTATION_METHODS

    # convert input matrix
    if is_sparse:
        A = matrix.sparse.util.convert_to_csc(A, sort_indices=True, eliminate_zeros=True, overwrite_A=True)

    # calculate approximation of A
    finished = False
    while not finished:
        # apply permutation previous to decomposition
        if permutation_method_previous is not None:
            p_previous_next = matrix.permute.permutation_vector(A, permutation_method=permutation_method_previous)
            p_previous = matrix.permute.concatenate_permutation_vectors(p_previous, p_previous_next)
            A = matrix.permute.symmetric(A, p_previous_next)
            if is_sparse:
                A = A.tocsc(copy=False)
            del p_previous_next

        # try to compute decomposition
        try:
            decomposition = decompose(A, permutation_method=permutation_method_decomposite, check_finite=False, overwrite_A=False)
        except matrix.errors.NoDecompositionPossibleTooManyEntriesError as error:
            if is_sparse and (A.indices.dtype != np.int64 or A.indptr != np.int64):
                matrix.logger.warning('Problem to large for index type {}, index type is switched to long.'.format(error.matrix_index_type))
                A = matrix.sparse.util.convert_index_dtype(A, np.int64, overwrite_A=True)
                return approximate_decomposition(A, t=t, min_diag_value=min_diag_value, max_diag_value=max_diag_value, min_abs_value=min_abs_value, permutation_method=permutation_method_decomposite, return_type=return_type, overwrite_A=overwrite_A, check_finite=False, callback=callback)
            else:
                matrix.logger.error(error)
                raise
        except matrix.errors.NoDecompositionPossibleWithProblematicSubdecompositionError as e:
            decomposition = e.subdecomposition
            bad_index = e.problematic_leading_principal_submatrix_index
        else:
            bad_index = n

        # get diagonal values of current (sub-)decomposition
        decomposition = decomposition.as_any_type(matrix.constants.LDL_DECOMPOSITION_TYPE, matrix.constants.LDL_DECOMPOSITION_COMPRESSED_TYPE)
        d = decomposition.d

        # get lowest index where decomposition is not possible
        bad_indices_mask = np.logical_or(d[:bad_index] < min_diag_value, d[:bad_index] > max_diag_value)
        bad_indices = np.where(bad_indices_mask)[0]
        if len(bad_indices) > 0:
            bad_index = np.min(bad_indices)
        del bad_indices_mask, bad_indices

        # if not all diagonal entries okay, reduce
        finished = bad_index >= n
        if not finished:
            # apply permutation
            i_permuted = bad_index
            i = decomposition.p[i_permuted]
            if p_previous is not None:
                i_unpermuted = p_previous[i]
            else:
                i_unpermuted = i

            # debug information
            matrix.logger.debug('Row and column {i_permuted} of permuted matrix ({i_unpermuted} unpermuted) must be approximated.'.format(
                i_permuted=i_permuted, i_unpermuted=i_unpermuted))

            # get A[i,i]
            if is_sparse:
                A_i_start_index = A.indptr[i]
                A_i_stop_index = A.indptr[i + 1]
                assert A_i_stop_index >= A_i_start_index
                A_ii_index_mask = np.where(A.indices[A_i_start_index:A_i_stop_index] == i)[0]
                if len(A_ii_index_mask) == 1:
                    A_ii_index = A_i_start_index + A_ii_index_mask[0]
                    assert A_i_start_index <= A_ii_index and A_i_stop_index >= A_ii_index
                    A_ii = A.data[A_ii_index]
                else:
                    assert len(A_ii_index_mask) == 0
                    A_ii_index = None
                    A_ii = 0
            else:
                A_ii = A[i, i]

            if np.iscomplexobj(A_ii):
                if np.isreal(A_ii):
                    A_ii = A_ii.real
                else:
                    error = ValueError('Matrix A is not Hermitian. A[{i}, {i}] = {A_ii} is complex.'.format(i=i, A_ii=A_ii))
                    matrix.logger.error(error)
                    raise error

            # get and check t[i]
            if t is None:
                t_i = A_ii
            else:
                t_i = t[i_unpermuted]
            if t_i < min_diag_value:
                error = ValueError('Each entry in the target vector t has to be greater or equal to min_diag_value {}. But its {}-th entry is {}.'.format(min_diag_value, i_unpermuted, t_i))
                matrix.logger.error(error)
                raise error
            if t_i > max_diag_value:
                error = ValueError('Each entry in the target vector t has to be lower or equal to max_diag_value {}. But its {}-th entry is {}.'.format(max_diag_value, i_unpermuted, t_i))
                matrix.logger.error(error)
                raise error

            # get L or LD
            if decomposition.is_type(matrix.constants.LDL_DECOMPOSITION_TYPE):
                L_or_LD = decomposition.L
            elif decomposition.is_type(matrix.constants.LDL_DECOMPOSITION_COMPRESSED_TYPE):
                L_or_LD = decomposition.LD
            else:
                assert False
            L_or_LD_row_i = L_or_LD[i_permuted]
            del decomposition, L_or_LD

            # get needed part of L and d
            if is_sparse:
                L_or_LD_row_i = L_or_LD_row_i.tocsr()
                L_or_LD_row_i_columns = L_or_LD_row_i.indices
                L_or_LD_row_i_data = L_or_LD_row_i.data
                assert len(L_or_LD_row_i_data) == len(L_or_LD_row_i_columns) >= 1
                assert L_or_LD_row_i_columns[-1] == i_permuted

                L_row_i_until_column_i = L_or_LD_row_i_data[:-1]
                d_until_i = d[L_or_LD_row_i_columns[:-1]]
            else:
                L_row_i_until_column_i = L_or_LD_row_i[:i_permuted]
                d_until_i = d[:i_permuted]

            # calculate reduction factor
            d_i_unmodified = A_ii - np.sum(L_row_i_until_column_i * L_row_i_until_column_i.conj() * d_until_i)
            if np.iscomplexobj(d_i_unmodified):
                assert np.isreal(d_i_unmodified)
                d_i_unmodified = d_i_unmodified.real
            assert not np.iscomplexobj(d_i_unmodified)

            if d_i_unmodified < min_diag_value:
                reduction_factor = ((t_i - min_diag_value) / (t_i - d_i_unmodified))**(0.5)
            elif d_i_unmodified > max_diag_value:
                reduction_factor = ((max_diag_value - t_i) / (d_i_unmodified - t_i))**(0.5)
            elif np.isclose(d_i_unmodified, min_diag_value) or np.isclose(d_i_unmodified, max_diag_value):
                reduction_factor = max_reduction_factor
            elif d_i_unmodified == 0:
                if A_ii == 0 and t_i == 0:
                    reduction_factor = 0
                else:
                    reduction_factor = max_reduction_factor
            else:
                assert False

            assert np.isreal(reduction_factor)
            assert 0 <= reduction_factor <= 1

            if reduction_factor > max_reduction_factor:
                reduction_factor = max_reduction_factor
            assert 0 <= reduction_factor < 1

            # apply reduction factor
            if t is None:
                t_i = None
            if is_sparse:
                A = _approximate_decomposition_apply_reduction_factor(A, i, reduction_factor, t_i=t_i, min_abs_value=min_abs_value, is_sparse=True, A_ii=A_ii, A_i_start_index=A_i_start_index, A_i_stop_index=A_i_stop_index, A_ii_index=A_ii_index)
                A.eliminate_zeros()
            else:
                A = _approximate_decomposition_apply_reduction_factor(A, i, reduction_factor, t_i=t_i, min_abs_value=min_abs_value, is_sparse=False, A_ii=A_ii)

            # call callback
            if callback is not None:
                callback(i_unpermuted, reduction_factor)

            # do not permute A again because diagonal values of A did not changed
            if t is None:
                permutation_method_previous = None

    # apply previous permutation
    decomposition._apply_previous_permutation(p_previous)

    # return
    assert np.all(np.isreal(decomposition.d))
    assert np.all(decomposition.d >= min_diag_value)
    assert np.all(decomposition.d <= max_diag_value)

    if return_type is not None:
        decomposition = decomposition.as_type(return_type)

    matrix.logger.debug('Approximating decomposition of a matrix finished.')
    return decomposition


def _approximate_decomposition_apply_reduction_factor(A, i, reduction_factor, t_i=None, min_abs_value=None, is_sparse=None, A_ii=None, A_i_start_index=None, A_i_stop_index=None, A_ii_index=None):
    # debug information
    matrix.logger.debug('Row and column {i} matrix are reduced with factor {reduction_factor}.'.format(
        i=i, reduction_factor=reduction_factor))

    # reduce
    if reduction_factor != 1:
        # init not passed inputs
        if min_abs_value is None:
            min_abs_value = np.finfo(A.dtype).resolution
        if is_sparse is None:
            is_sparse = matrix.sparse.util.is_sparse(A)

        if is_sparse:
            A_i_start_index, A_i_stop_index, A_ii_index, A_ii = matrix.sparse.util.compressed_matrix_indices(A, i, A_i_start_index=A_i_start_index, A_i_stop_index=A_i_stop_index, A_ii_index=A_ii_index, A_ii=A_ii)
        else:
            if A_ii is None:
                A_ii = A[i, i]
                if np.iscomplexobj(A_ii):
                    if np.isreal(A_ii):
                        A_ii = A_ii.real
                    else:
                        error = ValueError('Matrix A is not Hermitian. A[{i}, {i}] = {A_ii} is complex.'.format(i=i, A_ii=A_ii))
                        matrix.logger.error(error)
                        raise error

        # apply reduction factor
        if t_i is None:
            A_ii_new = A_ii    # reduces rounding errors
        else:
            A_ii_new = (1 - reduction_factor**2) * t_i + reduction_factor**2 * A_ii
        assert np.isreal(A_ii_new)

        if is_sparse:
            # set column (or row)
            A.data[A_i_start_index:A_i_stop_index] *= reduction_factor

            # apply min_abs_value in column
            if min_abs_value > 0:
                set_to_zero_indices = np.where(np.abs(A.data[A_i_start_index:A_i_stop_index]) < min_abs_value)[0]
                set_to_zero_indices += A_i_start_index
                A.data[set_to_zero_indices] = 0
                del set_to_zero_indices

            # set row (or column)
            A_i_data = A.data[A_i_start_index:A_i_stop_index]
            A_i_rows = A.indices[A_i_start_index:A_i_stop_index]
            for j, A_ij in zip(A_i_rows, A_i_data.conj()):
                if i != j:
                    A[i, j] = A_ij
            del A_i_data, A_i_rows

            # set diagonal entry
            if A_ii_index is not None:
                A.data[A_ii_index] = A_ii_new
            elif A_ii_new != 0:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', scipy.sparse.SparseEfficiencyWarning)
                    A[i, i] = A_ii_new

        else:
            # set column
            A[:, i] *= reduction_factor

            # apply min_abs_value in column
            if min_abs_value > 0:
                A[np.abs(A[:, i]) < min_abs_value, i] = 0

            # set row
            A[i, :] = A[:, i].conj().T

            # set diagonal entry
            A[i, i] = A_ii_new

    return A


def approximate_decomposition_apply_reduction_factors(A, reduction_factors, t=None, min_abs_value=None, overwrite_A=False):
    """
    Computes an approximative of `A` using the passed reduction factors.

    Parameters
    ----------
    A : numpy.ndarray or scipy.sparse.spmatrix
        The matrix that should be approximated.
        It is assumed, that A is Hermitian.
        The matrix must be a squared matrix.
    t : numpy.ndarray
        The targed vector used for the approximation. For each i in range(M)
        `min_diag_value <= t[i] <= max_diag_value` must hold.
        `t` and `A` must have the same length.
        optional, default : The diagonal of `A` is used as `t`.
    min_abs_value : float
        Absolute values below `min_abs_value` are considered as zero.
        optional, default : The resolution of the underlying data type is used.
    overwrite_A : bool
        Whether it is allowed to overwrite A.
        Enabling may result in performance gain.
        optional, default: False

    Returns
    -------
    numpy.ndarray or scipy.sparse.spmatrix
        An approximative of `A` using the passed reduction factors.
    """

    # debug logging
    matrix.logger.debug('Applying reduction factors to a matrix with min_abs_value={min_abs_value}, overwrite_A={overwrite_A}.'.format(
        min_abs_value=min_abs_value,
        overwrite_A=overwrite_A))

    # init
    A, t = _approximate_decomposition_init(A, t=t, min_abs_value=min_abs_value, copy=not overwrite_A)
    is_sparse = matrix.sparse.util.is_sparse(A)

    # convert input matrix
    if is_sparse:
        A = matrix.sparse.util.convert_to_csc(A, sort_indices=True, eliminate_zeros=True, overwrite_A=True)

    # apply reduction factors
    for i, r in enumerate(reduction_factors):
        if r != 1:
            if t is not None:
                t_i = t[i]
            else:
                t_i = None
            A = _approximate_decomposition_apply_reduction_factor(A, i, r, t_i=t_i, min_abs_value=min_abs_value, is_sparse=is_sparse)

    # return matrix
    if is_sparse:
        A.eliminate_zeros()

    matrix.logger.debug('Applying reduction factors finished.')
    return A


def approximate_decomposition_with_reduction_factor_file(A, t=None, min_abs_value=None, min_diag_value=None, max_diag_value=None, permutation_method=None, return_type=None, check_finite=True, overwrite_A=False, reduction_factors_file=None):
    """
    Computes an approximative decomposition of a matrix.

    If `A` is decomposable in a decomposition of type `return_type`, this decomposition is returned.
    Otherwise a decomposition of type `return_type` is retuned which represents an approximation
    of `A`.

    Parameters
    ----------
    A : numpy.ndarray or scipy.sparse.spmatrix
        The matrix that should be approximated by a decomposition.
        It is assumed, that A is Hermitian.
        The matrix must be a squared matrix.
    t : numpy.ndarray
        The targed vector used for the approximation. For each i in range(M)
        `min_diag_value <= t[i] <= max_diag_value` must hold.
        `t` and `A` must have the same length.
        optional, default : The diagonal of `A` is used as `t`.
    min_abs_value : float
        Absolute values below `min_abs_value` are considered as zero.
        optional, default : The resolution of the underlying data type is used.
    min_diag_value : float
        Each component of the diagonal of the matrix `D` in an returned `LDL` decomposition
        is forced to be greater or equal to `min_diag_value`.
        optional, default : 0.
    max_diag_value : float
        Each component of the diagonal of the matrix `D` in an returned `LDL` decomposition
        is forced to be lower or equal to `max_diag_value`.
        optional, default : No maximal value is forced.
    permutation_method : str
        The symmetric permutation method that is applied to the matrix before
        it is decomposed. It has to be a value in
        :const:`matrix.PERMUTATION_METHODS`.
        If `A` is sparse, it can also be a value in
        :const:`matrix.SPARSE_PERMUTATION_METHODS`.
        optional, default: No permutation is done.
    return_type : str
        The type of the decomposition that should be calculated.
        It has to be a value in :const:`matrix.DECOMPOSITION_TYPES`.
        optional, default : The type of the decomposition is chosen by the function itself.
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
    reduction_factors_file : str
        A file where the reduction_factors used to compute the approximation of `A` are stored.
        If this file already exists the previoisly stored values are used.
        This allows the calculation of the approximation to be interrupted and resumed.

    Returns
    -------
    matrix.decompositions.DecompositionBase
        An approximative decomposition of `A` of type `return_type`.

    Raises
    ------
    matrix.errors.MatrixNotSquareError
        If `A` is not a square matrix.
    matrix.errors.MatrixNotFiniteError
        If `A` is not a finite matrix and `check_finite` is True.
    """

    # debug logging
    matrix.logger.debug('Approximating decomposition of a matrix with min_abs_value={min_abs_value}, min_diag_value={min_diag_value}, max_diag_value={max_diag_value}, permutation_method={permutation_method}, return_type={return_type}, check_finite={check_finite}, overwrite_A={overwrite_A}, reduction_factors_file={reduction_factors_file}.'.format(
        min_abs_value=min_abs_value,
        min_diag_value=min_diag_value,
        max_diag_value=max_diag_value,
        permutation_method=permutation_method,
        return_type=return_type,
        check_finite=check_finite,
        overwrite_A=overwrite_A,
        reduction_factors_file=reduction_factors_file))

    # callback function
    def approximate_callback_save_reduction_factors(reduction_factors_file, n):
        """ Returns a callback function for :const:`matrix.approximate` which saves each iteration in a :mod:`numpy` file.

        Parameters
        ----------
        reduction_factors_file : str
            The file where the reduction factors are saved.
        n : int
            The dimension of the squared matrix that is approximated.

        Returns
        -------
        callable
            A callback function for :func:`matrix.approximate` which saves each iteration in a :mod:`numpy` file.
        """

        try:
            reduction_factors = np.load(reduction_factors_file, mmap_mode='r+')
        except FileNotFoundError:
            reduction_factors = np.ones(n, dtype=np.float)
            os.makedirs(os.path.dirname(reduction_factors_file), exist_ok=True)
            np.save(reduction_factors_file, reduction_factors)
            reduction_factors = np.load(reduction_factors_file, mmap_mode='r+')
        else:
            if reduction_factors.shape != (n,):
                error = ValueError('The reduction factors file contains an array with shape {} but the expected shape is {}'.format(reduction_factors.shape, (n,)))
                matrix.logger.error(error)
                raise error

        def callback_function(i, reduction_factor):
            reduction_factors[i] = reduction_factors[i] * reduction_factor
            reduction_factors.flush()

        return callback_function

    # apply previous reduction factors
    if reduction_factors_file is not None:
        try:
            reduction_factors = np.load(reduction_factors_file)
        except FileNotFoundError:
            pass
        else:
            A = approximate_decomposition_apply_reduction_factors(A, reduction_factors, t=t, min_abs_value=min_abs_value, overwrite_A=overwrite_A)
            overwrite_A = False

        n = A.shape[0]
        callback = approximate_callback_save_reduction_factors(reduction_factors_file, n)
    else:
        callback = None

    # run approximation
    return approximate_decomposition(A, t=t, min_diag_value=min_diag_value, max_diag_value=max_diag_value, min_abs_value=min_abs_value, permutation_method=permutation_method, return_type=return_type, check_finite=check_finite, overwrite_A=overwrite_A, callback=callback)


def approximate_positive_definite_matrix(A, positive_definiteness_parameter=None, min_abs_value=None, check_finite=True, overwrite_A=False):
    """
    Computes a positive definite approximation a matrix.

    If `A` is decomposable in a decomposition of type `return_type`, this decomposition is returned.
    Otherwise a decomposition of type `return_type` is retuned which represents an approximation
    of `A`.

    Parameters
    ----------
    A : numpy.ndarray or scipy.sparse.spmatrix
        The matrix that should be approximated by a decomposition.
        It is assumed, that A is Hermitian.
        The matrix must be a squared matrix.
    positive_definiteness_parameter : float
        A positive parameter which controls how far the eigenvalues
        of the approximation are away from zero.
        optional, default : The fourth root of the resolution of the underlying data type is used.
    min_abs_value : float
        Absolute values below `min_abs_value` are considered as zero.
        optional, default : The resolution of the underlying data type is used.
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
        An approximative decomposition of `A` of type `return_type`.

    Raises
    ------
    matrix.errors.MatrixNotSquareError
        If `A` is not a square matrix.
    matrix.errors.MatrixNotFiniteError
        If `A` is not a finite matrix and `check_finite` is True.
    """

    # debug logging
    matrix.logger.debug('Approximating matrix as positive definite matrix with min_abs_value={min_abs_value}, check_finite={check_finite}, overwrite_A={overwrite_A}.'.format(
        min_abs_value=min_abs_value,
        check_finite=check_finite,
        overwrite_A=overwrite_A))

    # init min_diag_value
    dtype_resolution = np.finfo(A.dtype).resolution
    if min_abs_value is None:
        min_abs_value = dtype_resolution
    if positive_definiteness_parameter is None:
        positive_definiteness_parameter = dtype_resolution**(0.25)
    min_diag_value = max(min_abs_value, positive_definiteness_parameter)

    # init t
    t = A.diagonal()
    if np.iscomplexobj(t):
        if np.all(np.isreal(t)):
            t = t.real
        else:
            error = ValueError('A is not Hermitian. Some diagonal values are complex.')
            matrix.logger.error(error)
            raise error
    if np.any(t < min_diag_value):
        if not t.flags.writeable:
            t = t.copy()
        t[t < min_diag_value] = min_diag_value

    # calculate approximation
    approximated_decomposition = approximate_decomposition(A, t=t, min_abs_value=min_abs_value, min_diag_value=min_diag_value, permutation_method=matrix.constants.INCREASING_DIAGONAL_VALUES_PERMUTATION_METHOD, check_finite=check_finite, overwrite_A=overwrite_A)
    A_approximated = approximated_decomposition.composed_matrix
    A_approximated = matrix.util.set_diagonal_nearly_real_to_real(A_approximated, min_abs_value=min_abs_value)
    A_approximated = matrix.util.set_nearly_zero_to_zero(A_approximated, min_abs_value=min_abs_value)
    return A_approximated


def is_positive_semi_definite(A, check_finite=True):
    """
    Returns whether the passed matrix is positive semi-definite.

    Parameters
    ----------
    A : numpy.ndarray or scipy.sparse.spmatrix
        The matrix that should be checked.
        It is assumed, that A is Hermitian.
        The matrix must be a squared matrix.
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
        decomposition = decompose(A, permutation_method=matrix.constants.INCREASING_DIAGONAL_VALUES_PERMUTATION_METHOD, check_finite=check_finite)
    except (matrix.errors.NoDecompositionPossibleError,
            matrix.errors.MatrixComplexDiagonalValueError,
            matrix.errors.MatrixNotFiniteError,
            matrix.errors.MatrixNotSquareError):
        return False
    else:
        return decomposition.is_positive_semi_definite()


def is_positive_definite(A, check_finite=True):
    """
    Returns whether the passed matrix is positive definite.

    Parameters
    ----------
    A : numpy.ndarray or scipy.sparse.spmatrix
        The matrix that should be checked.
        It is assumed, that A is Hermitian.
        The matrix must be a squared matrix.
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
        decomposition = decompose(A, permutation_method=matrix.constants.INCREASING_DIAGONAL_VALUES_PERMUTATION_METHOD, check_finite=check_finite)
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
        It is assumed, that A is Hermitian.
        The matrix must be a squared matrix.
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
    matrix.logger.debug('Checking whether matrix isinvertable with check_finite={check_finite}.'.format(
        check_finite=check_finite))

    # try to decompose and check decomposition
    try:
        decomposition = decompose(A, permutation_method=matrix.constants.INCREASING_DIAGONAL_VALUES_PERMUTATION_METHOD, check_finite=check_finite)
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
        It is assumed, that A is Hermitian.
        The matrix must be a squared matrix.
    b : numpy.ndarray
        Right-hand side vector or matrix in equation `A x = b`.
        Ii must hold `b.shape[0] == A.shape[0]`.
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
