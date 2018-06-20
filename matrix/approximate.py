import numpy as np
import scipy.sparse

import matrix
import matrix.constants
import matrix.decompositions
import matrix.errors
import matrix.permute
import matrix.sparse.util


def _decomposition(
        A, min_diag_B=None, max_diag_B=None, min_diag_D=None, max_diag_D=None,
        min_abs_value_D=None, permutation=None, overwrite_A=False,
        strict_lower_triangular_only_L=False):
    """
    Computes an (approximative) :math:`LDL^H` decomposition of a matrix with the specified properties.

    Returns a :math:`LDL^H` decomposition of `A` if such a decomposition exists.
    Otherwise a :math:`LDL^H` decomposition of an approximation of `A` is retuned.

    Parameters
    ----------
    A : numpy.ndarray or scipy.sparse.spmatrix
        The matrix that should be approximated by a decomposition.
        `A` must be Hermitian.
    min_diag_B : numpy.ndarray or float
        Each component of the diagonal of the composed matrix `B` of an approximated
        :math:`LDL^H` decomposition is forced to be greater or equal to `min_diag_B`.
        optional, default : No minimal value is forced.
    max_diag_B : numpy.ndarray or float
        Each component of the diagonal of the composed matrix `B` of an approximated
        :math:`LDL^H` decomposition is forced to be lower or equal to `max_diag_B`.
        optional, default : No maximal value is forced.
    min_diag_D : float
        Each component of the diagonal of the matrix `D` in an approximated
        :math:`LDL^H` decomposition is forced to be greater or equal to `min_diag_D`.
        `min_diag_D` must be greater than 0.
        optional, default : The square root of the resolution of the underlying data type.
    max_diag_D : float
        Each component of the diagonal of the matrix `D` in an approximated
        :math:`LDL^H` decomposition is forced to be lower or equal to `max_diag_D`.
        optional, default : No maximal value is forced.
    min_abs_value_D : float
        Absolute values below `min_abs_value_D` are considered as zero
        in the matrix `D` of an approximated :math:`LDL^H` decomposition.
        `min_abs_value_D` must be greater or equal to 0.
        optional, default : The square root of the resolution of the underlying data type.
    permutation : str or numpy.ndarray
        The symmetric permutation method that is applied to the matrix before it is decomposed.
        It has to be a value in :const:`matrix.UNIVERSAL_PERMUTATION_METHODS` or
        :const:`matrix.APPROXIMATION_ONLY_PERMUTATION_METHODS`.
        If `A` is sparse, it can also be a value in
        :const:`matrix.SPARSE_ONLY_PERMUTATION_METHODS`.
        It is also possible to directly pass a permutation vector.
        optional, default: The permutation is chosen by the algorithm.
    overwrite_A : bool
        Whether it is allowed to overwrite `A`. Enabling may result in performance gain.
        optional, default: False
    strict_lower_triangular_only_L : bool
        Whether only the strict lower triangular matrix of the matrix `L` in the :math:`LDL^H`
        decomposition should be computed. If this is true, its upper triangular matrix
        may contain arbitrary values. Enabling may result in performance gain.
        optional, default : False

    Returns
    -------
    L : numpy.ndarray or scipy.sparse.spmatrix (same type as A)
        Matrix `L` of the decomposition.
    d : numpy.ndarray
        Diagonal of matrix `D` of the decomposition.
    p : numpy.ndarray
        Permutation vector.
    omega : numpy.ndarray
        Auxiliary vector for calculating B.
    delta : numpy.ndarray
        Auxiliary vector for calculating B.

    Raises
    ------
    matrix.errors.MatrixNotSquareError
        If `A` is not a square matrix.
    matrix.errors.MatrixComplexDiagonalValueError
        If `A` has complex diagonal values.
    """

    # debug info
    matrix.logger.debug(('Calculating approximated LDL decomposition with passed values: '
                         'min_diag_B={min_diag_B}, max_diag_B={max_diag_B}, '
                         'min_diag_D={min_diag_D}, max_diag_D={max_diag_D}, '
                         'min_abs_value_D={min_abs_value_D}, '
                         'permutation={permutation}, overwrite_A={overwrite_A}, '
                         'strict_lower_triangular_only_L={strict_lower_triangular_only_L}.'
                         ).format(
        min_diag_B=min_diag_B, max_diag_B=max_diag_B,
        min_diag_D=min_diag_D, max_diag_D=max_diag_D, min_abs_value_D=min_abs_value_D,
        permutation=permutation, overwrite_A=overwrite_A,
        strict_lower_triangular_only_L=strict_lower_triangular_only_L))

    # check A
    is_dense = not matrix.sparse.util.is_sparse(A)
    if is_dense:
        A = np.asarray(A)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise matrix.errors.MatrixNotSquareError(A)
    n = A.shape[0]

    # calculate gamma
    gamma = A.diagonal()
    if np.issubdtype(A.dtype, np.complexfloating):
        if not np.all(np.isreal(gamma)):
            index_with_complex_value = np.where(~np.isreal(gamma))[0][0]
            raise matrix.errors.MatrixComplexDiagonalValueError(A, i=index_with_complex_value)
        gamma = gamma.real

    # data type to use for calculations
    DTYPE = np.float64
    d_eps = np.finfo(DTYPE).eps

    # check diag values
    def check_float_scalar_or_vector(f, f_name='variable', default_value=None,
                                     minus_inf_okay=False, plus_inf_okay=False):
        if f is not None:
            try:
                f = np.asarray(f)
            except TypeError as original_error:
                error = ValueError(('{} must a scalar value or a vector but it is {}.'
                                    ).format(f_name, f))
                matrix.logger.error(error)
                raise error from original_error
            if not (f.ndim == 0 or (f.ndim == 1 and f.shape[0] == n)):
                error = ValueError(('{} must be a scalar or a vector with the same dimension as A '
                                    'but its shape is {}.').format(f_name, f.shape))
                matrix.logger.error(error)
                raise error
            if not (np.all(np.isreal(f))):
                error = ValueError(('{} must be real valued but it is {}.').format(f_name, f))
                matrix.logger.error(error)
                raise error
            if not np.all(np.logical_or(np.isfinite(f),
                                        np.logical_or(np.logical_and(plus_inf_okay,
                                                                     f == np.inf),
                                                      np.logical_and(minus_inf_okay,
                                                                     f == -np.inf)))):
                error = ValueError('{} must be finite'.format(f_name) +
                                   (' or minus infinity' if minus_inf_okay else '') +
                                   (' or plus infinity' if plus_inf_okay else '') +
                                   ' but it is {}.'.format(f))
                matrix.logger.error(error)
                raise error
        else:
            f = np.asarray(default_value)
        return f

    def check_float_scalar(f, f_name='variable', default_value=None, lower_bound=-np.inf,
                           minus_inf_okay=False, plus_inf_okay=False):
        if f is not None:
            try:
                f = float(f)
            except TypeError as original_error:
                error = ValueError('{} must a float value but it is {}.'.format(f_name, f))
                matrix.logger.error(error)
                raise error from original_error
            if not (np.isfinite(f) or (minus_inf_okay and f == -np.inf) or
                    (plus_inf_okay and f == np.inf)):
                error = ValueError('{} must be finite'.format(f_name) +
                                   (' or minus infinity' if minus_inf_okay else '') +
                                   (' or plus infinity' if plus_inf_okay else '') +
                                   ' but it is {}.'.format(f))
                matrix.logger.error(error)
                raise error
            if lower_bound is not None and f < lower_bound:
                error = ValueError(('{} must be finite and greater or equal to {} '
                                    'but it is {}.').format(f_name, lower_bound, f))
                matrix.logger.error(error)
                raise error
        else:
            f = default_value
        return f

    min_diag_B = check_float_scalar_or_vector(min_diag_B, 'min_diag_B', default_value=-np.inf,
                                              minus_inf_okay=True, plus_inf_okay=False)
    max_diag_B = check_float_scalar_or_vector(max_diag_B, 'max_diag_B', default_value=np.inf,
                                              minus_inf_okay=False, plus_inf_okay=True)

    min_diag_D = check_float_scalar(min_diag_D, f_name='min_diag_D',
                                    default_value=d_eps**0.5, lower_bound=d_eps,
                                    minus_inf_okay=False, plus_inf_okay=False)
    max_diag_D = check_float_scalar(max_diag_D, f_name='max_diag_D',
                                    default_value=np.inf, lower_bound=None,
                                    minus_inf_okay=False, plus_inf_okay=True)

    if not max(np.max(min_diag_B), min_diag_D) <= min(np.min(max_diag_B), max_diag_D):
        error = ValueError(('The values of min_diag_B and min_diag_D must be lower or equal '
                            'to the values of max_diag_B and max_diag_D.'))
        matrix.logger.error(error)
        raise error

    min_abs_value_D = check_float_scalar(min_abs_value_D, f_name='min_abs_value_D',
                                         default_value=d_eps**0.5, lower_bound=0,
                                         minus_inf_okay=False, plus_inf_okay=False)
    min_abs_value_D = max(min_abs_value_D, d_eps)

    # check overwrite_A
    if overwrite_A is None or not is_dense:
        overwrite_A = False
    if overwrite_A and np.issubdtype(A.dtype, np.integer):
        overwrite_A = False
        matrix.logger.debug(('A has an integer dtype ({}) which can not be used to store the '
                             'values of L. Thus L is not overwritten.').format(A.dtype))

    # check strict_lower_triangular_only_L
    if strict_lower_triangular_only_L is None:
        strict_lower_triangular_only_L = False

    # check permutation method
    if permutation is None:
        permutation = matrix.constants.MINIMAL_DIFFERENCE_PERMUTATION_METHOD

    # name of permutation method passed
    use_permutation_method = isinstance(permutation, str)
    if use_permutation_method:
        # check permutation method
        permutation_method = permutation.lower()
        supported_permutation_methods = (matrix.UNIVERSAL_PERMUTATION_METHODS +
                                         matrix.APPROXIMATION_ONLY_PERMUTATION_METHODS)
        if not is_dense:
            supported_permutation_methods = (supported_permutation_methods +
                                             matrix.SPARSE_ONLY_PERMUTATION_METHODS)
        if permutation_method not in supported_permutation_methods:
            error = ValueError(('Permutation method {} is unknown. Only the following methods are '
                                'supported {}.'
                                ).format(permutation_method, supported_permutation_methods))
            matrix.logger.error(error)
            raise error

        # calculate permutation vector
        matrix.logger.debug('Calculating permutation vector with method "{}".'
                            .format(permutation_method))

        use_minimal_difference_permutation_method = (
            permutation_method == matrix.constants.MINIMAL_DIFFERENCE_PERMUTATION_METHOD)

        if use_minimal_difference_permutation_method:
            if min_diag_D < 0:
                raise ValueError(('The permutation method {} is only available if min_diag_D '
                                  'greater or equal zero.'
                                  ).format(matrix.constants.MINIMAL_DIFFERENCE_PERMUTATION_METHOD))
            if is_dense:
                p = np.arange(n, dtype=np.min_scalar_type(n))
            else:
                p = matrix.permute.permutation_vector(
                    A,
                    permutation_method=matrix.sparse.constants.DEFAULT_FILL_REDUCE_PERMUTATION_METHOD)
        else:
            p = matrix.permute.permutation_vector(A, permutation_method=permutation_method)
    else:
        p = np.asanyarray(permutation)
        if p.ndim != 1 or p.shape[0] != n:
            error = ValueError(('Permutation vactor must have same length as the dimensions of A. '
                                'Its shape is {} and the shape of A is {}.'
                                ).format(p.shape, A.shape))
            matrix.logger.error(error)
            raise error
        use_minimal_difference_permutation_method = False

    # init L
    if overwrite_A:
        L = A
    else:
        L_dtype = np.promote_types(A.dtype, np.float16)
        if is_dense:
            L = np.zeros((n, n), dtype=L_dtype)
        else:
            L = scipy.sparse.lil_matrix((n, n), dtype=L_dtype)
            L_rows = L.rows
            L_data = L.data
    L_eps = np.finfo(L.dtype).eps

    # init other values
    alpha = np.zeros(n, dtype=DTYPE)
    beta = np.zeros(n, dtype=DTYPE)
    delta = np.empty(n, dtype=DTYPE)
    omega = np.empty(n, dtype=DTYPE)
    d = np.empty(n, dtype=DTYPE)

    # debug info
    matrix.logger.debug(('Using the following values: '
                         'min_diag_B={min_diag_B}, max_diag_B={max_diag_B}, '
                         'min_diag_D={min_diag_D}, max_diag_D={max_diag_D}, '
                         'min_abs_value_D={min_abs_value_D} '
                         'permutation={permutation}, overwrite_A={overwrite_A}, '
                         'strict_lower_triangular_only_L={strict_lower_triangular_only_L}.'
                         ).format(
        min_diag_B=min_diag_B, max_diag_B=max_diag_B,
        min_diag_D=min_diag_D, max_diag_D=max_diag_D, min_abs_value_D=min_abs_value_D,
        permutation=permutation, overwrite_A=overwrite_A,
        strict_lower_triangular_only_L=strict_lower_triangular_only_L))

    # calculate values iteratively
    def get_value_i(v, i):
        try:
            return v[i]
        except TypeError:
            return v
        except IndexError:
            assert v.ndim == 0
            return v

    for i in range(n):
        matrix.logger.debug('Starting iteration {} of {}.'.format(i, n - 1))

        # update p, d, omega
        if use_minimal_difference_permutation_method:
            all_minimal_changes = ((j, *_minimal_change(
                alpha[p[j]], beta[p[j]], gamma[p[j]], min_diag_D, max_diag_D=max_diag_D,
                min_diag_B=get_value_i(min_diag_B, p[j]), max_diag_B=get_value_i(max_diag_B, p[j]),
                min_abs_value_D=min_abs_value_D))
                for j in range(i, n))
            (j, d_i, omega_i, f_value_i) = min(all_minimal_changes,
                                               key=lambda x: (x[3], -x[1], x[2], x[0]))
            # swap p[i] and p[j]
            p_i = p[j]
            p[j] = p[i]
            p[i] = p_i
            # swap L[i, :] and L[j, :]
            if i > 0:
                if is_dense or L.format != 'lil':
                    L_j = L[j, :i].copy()
                    L[j, :i] = L[i, :i]
                    L[i, :i] = L_j
                else:
                    for iterable in (L_rows, L_data):
                        tmp = iterable[i]
                        iterable[i] = iterable[j]
                        iterable[j] = tmp
        else:
            p_i = p[i]
            d_i, omega_i, f_value_i = _minimal_change(
                alpha[p_i], beta[p_i], gamma[p_i], min_diag_D, max_diag_D=max_diag_D,
                min_diag_B=get_value_i(min_diag_B, p_i), max_diag_B=get_value_i(max_diag_B, p_i),
                min_abs_value_D=min_abs_value_D)

        # update d
        assert np.isfinite(d_i)
        assert d_i >= min_diag_D
        assert d_i <= max_diag_D
        assert d_i == 0 or np.abs(d_i) >= min_abs_value_D
        d[i] = d_i

        # update omega
        assert np.isfinite(omega_i)
        assert omega_i >= 0
        omega[p_i] = omega_i

        # update delta
        delta_i = d_i - gamma[p_i]
        if omega_i != 0:
            delta_i += omega_i**2 * alpha[p_i]
        assert np.isfinite(delta_i)
        delta[p_i] = delta_i

        # debug info
        matrix.logger.debug(('Using permutation index {}, omega {}, delta {} and change value {} '
                             'for iteration {} of {}. ({:.1%} done.)'
                             ).format(p_i, omega[p_i], delta[p_i], f_value_i,
                                      i, n - 1, (i + 1) / n))

        # update i-th row of L with omega
        if omega_i != 1 and i > 0:
            if is_dense:
                if omega_i != 0:
                    L[i, :i] *= omega_i
                    L[i, np.where(abs(L[i, :i]) < L_eps)[0]] = 0
                else:
                    L[i, :i] = 0
            else:
                assert L.format == 'lil'
                L_i_rows = []
                L_i_data = []
                if omega_i != 0:
                    for row, data in zip(L_rows[i], L_data[i]):
                        assert row < i
                        data_new = data * omega_i
                        if abs(data_new) >= L_eps:
                            L_i_rows.append(row)
                            L_i_data.append(data_new)
                L_rows[i] = L_i_rows
                L_data[i] = L_i_data

        # get i-th column of A
        p_after_i = p[i + 1:]
        if is_dense:
            if not overwrite_A:
                A_column_i = A[:, p_i]
            else:
                A_column_i = np.concatenate([A[:p_i, p_i], A[p_i, p_i:].conj()])
        else:
            if A.format in ('csr', 'lil'):
                A_column_i = A[p_i, :].conj().T
            else:
                A_column_i = A[:, p_i]
            A_column_i = A_column_i.toarray().reshape(-1)
        A_column_i = A_column_i[p_after_i]
        assert np.all(np.isfinite(A_column_i))

        # update beta
        beta_add = 2 * A_column_i * A_column_i.conj()
        assert np.all(np.isreal(beta_add))
        beta[p_after_i] += beta_add.real

        # update alpha and i-th column of L
        L_column_i = A_column_i
        if d_i != 0:
            # calculate i-th column of L
            if is_dense:
                if i > 0:
                    L_row_i_mul_d = L[i, :i].conj() * d[:i]
                    L_column_i -= L[i + 1:, :i] @ L_row_i_mul_d
            else:
                assert L.format == 'lil'
                L_row_i_mul_d = L[i, :].conj().multiply(d).T
                if L_row_i_mul_d.nnz > 0:
                    L_row_i_mul_d = L_row_i_mul_d.toarray().reshape(-1)
                    L_below_row_i = scipy.sparse.lil_matrix((1, 1), dtype=L.dtype)
                    L_below_row_i.rows = L_rows[i + 1:]
                    L_below_row_i.data = L_data[i + 1:]
                    L_below_row_i._shape = (n - (i + 1), n)
                    L_column_i -= L_below_row_i @ L_row_i_mul_d

            # devide by d_i
            L_column_i /= d_i
            assert np.all(np.logical_or(np.isfinite(L_column_i),
                                        np.logical_or(L_column_i == np.inf,
                                                      L_column_i == -np.inf)))

            # update i-th column of L
            if is_dense:
                L[i + 1:, i] = L_column_i
            else:
                assert L.format == 'lil'
                L_column_i_non_zero_mask = L_column_i != 0
                L_column_i = L_column_i[L_column_i_non_zero_mask]
                p_after_i = p_after_i[L_column_i_non_zero_mask]
                for k, L_column_i_k in zip(np.where(L_column_i_non_zero_mask)[0], L_column_i):
                    j = i + 1 + k
                    L_rows[j].append(i)
                    L_data[j].append(L_column_i_k)

            # update alpha
            alpha_add = L_column_i * L_column_i.conj() * d_i
            assert np.all(np.isreal(alpha_add))
            alpha_add = alpha_add.real
            assert np.all(np.logical_or(np.isfinite(alpha_add), alpha_add == np.inf))
            alpha_add[L_column_i == np.inf] = np.inf
            alpha[p_after_i] += alpha_add
            assert np.all(np.logical_or(np.isfinite(alpha), alpha == np.inf))
            assert np.all(np.logical_or(np.isfinite(L_column_i), alpha[p_after_i] == np.inf))

    # prepare diagonal and upper triangle of L if needed
    if not strict_lower_triangular_only_L:
        if is_dense:
            for i in range(n):
                L[i, i] = 1
        else:
            for i in range(n):
                L_rows[i].append(i)
                L_data[i].append(1)

    if not strict_lower_triangular_only_L and overwrite_A:
        assert is_dense
        for i in range(n):
            L[i, i + 1:] = 0

    # return values
    matrix.logger.debug('Approximation of LDL decomposition finished.')
    return L, d, p, omega, delta


def decomposition(
        A, min_diag_B=None, max_diag_B=None, min_diag_D=None, max_diag_D=None,
        min_abs_value_D=None, permutation=None, overwrite_A=False, return_type=None):
    """
    Computes an approximative decomposition of a matrix with the specified properties.

    Returns a decomposition of `A` if has such a decomposition and
    otherwise  a decomposition of an approximation of `A`.

    Parameters
    ----------
    A : numpy.ndarray or scipy.sparse.spmatrix
        The matrix that should be approximated by a decomposition.
        `A` must be Hermitian.
    min_diag_B : numpy.ndarray or float
        Each component of the diagonal of the composed matrix `B` of an approximated
        :math:`LDL^H` decomposition is forced to be greater or equal to `min_diag_B`.
        optional, default : No minimal value is forced.
    max_diag_B : numpy.ndarray or float
        Each component of the diagonal of the composed matrix `B` of an approximated
        :math:`LDL^H` decomposition is forced to be lower or equal to `max_diag_B`.
        optional, default : No maximal value is forced.
    min_diag_D : float
        Each component of the diagonal of the matrix `D` in an approximated
        :math:`LDL^H` decomposition is forced to be greater or equal to `min_diag_D`.
        `min_diag_D` must be greater than 0.
        optional, default : The square root of the resolution of the underlying data type.
    max_diag_D : float
        Each component of the diagonal of the matrix `D` in an approximated
        :math:`LDL^H` decomposition is forced to be lower or equal to `max_diag_D`.
        optional, default : No maximal value is forced.
    min_abs_value_D : float
        Absolute values below `min_abs_value_D` are considered as zero
        in the matrix `D` of an approximated :math:`LDL^H` decomposition.
        `min_abs_value_D` must be greater or equal to 0.
        optional, default : The square root of the resolution of the underlying data type.
    permutation : str or numpy.ndarray
        The symmetric permutation method that is applied to the matrix before it is decomposed.
        It has to be a value in :const:`matrix.UNIVERSAL_PERMUTATION_METHODS` or
        :const:`matrix.APPROXIMATION_ONLY_PERMUTATION_METHODS`.
        If `A` is sparse, it can also be a value in
        :const:`matrix.SPARSE_ONLY_PERMUTATION_METHODS`.
        It is also possible to directly pass a permutation vector.
        optional, default: The permutation is chosen by the algorithm.
    overwrite_A : bool
        Whether it is allowed to overwrite `A`. Enabling may result in performance gain.
        optional, default: False
    return_type : str
        The type of the decomposition that should be returned.
        It has to be a value in :const:`matrix.DECOMPOSITION_TYPES`.
        optional, default : The type of the decomposition is chosen by the function itself.

    Returns
    -------
    matrix.decompositions.DecompositionBase
        An (approximative) decomposition of `A` of type `return_type`.

    Raises
    ------
    matrix.errors.MatrixNotSquareError
        If `A` is not a square matrix.
    matrix.errors.MatrixComplexDiagonalValueError
        If `A` has complex diagonal values.
    """

    # debug info
    matrix.logger.debug(('Calculating approximated decomposition with passed values: '
                         'min_diag_B={min_diag_B}, max_diag_B={max_diag_B}, '
                         'min_diag_D={min_diag_D}, max_diag_D={max_diag_D}, '
                         'min_abs_value_D={min_abs_value_D}, '
                         'permutation={permutation}, overwrite_A={overwrite_A}, '
                         'return_type={return_type}.').format(
        min_diag_B=min_diag_B, max_diag_B=max_diag_B,
        min_diag_D=min_diag_D, max_diag_D=max_diag_D, min_abs_value_D=min_abs_value_D,
        permutation=permutation, overwrite_A=overwrite_A, return_type=return_type))

    # check return type
    supported_return_types = matrix.constants.DECOMPOSITION_TYPES
    if return_type is not None and return_type not in supported_return_types:
        error = ValueError(('Unkown return type {}. Only values in {} are supported.'
                            ).format(return_type, supported_return_types))
        matrix.logger.error(error)
        raise error

    # calculate decomposition
    L, d, p, omega, delta = _decomposition(
        A, min_diag_B=min_diag_B, max_diag_B=max_diag_B,
        min_diag_D=min_diag_D, max_diag_D=max_diag_D, min_abs_value_D=min_abs_value_D,
        permutation=permutation, overwrite_A=overwrite_A,
        strict_lower_triangular_only_L=False)

    if matrix.sparse.util.is_sparse(L):
        matrix.logger.debug('Converting L to csr format.')
        L = L.tocsr(copy=False)
    decomposition = matrix.decompositions.LDL_Decomposition(L=L, d=d, p=p).as_type(return_type)
    decomposition.omega = omega
    decomposition.delta = delta

    # return decomposition
    matrix.logger.debug('Approximation of decomposition {} finished.'.format(decomposition))
    return decomposition


def _matrix(
        A, min_diag_B=None, max_diag_B=None, min_diag_D=None, max_diag_D=None,
        min_abs_value_D=None, permutation=None, overwrite_A=False):
    """
    Computes an approximation of `A` which has a :math:`LDL^H` decomposition with the specified properties.

    Returns `A` if `A` has such a decomposition and otherwise an approximation of `A`.

    Parameters
    ----------
    A : numpy.ndarray or scipy.sparse.spmatrix
        The matrix that should be approximated.
        `A` must be Hermitian.
    min_diag_B : numpy.ndarray or float
        Each component of the diagonal of the returned matrix
        is forced to be greater or equal to `min_diag_B`.
        optional, default : No minimal value is forced.
    max_diag_B : numpy.ndarray or float
        Each component of the diagonal of the returned matrix
        is forced to be lower or equal to `max_diag_B`.
        optional, default : No maximal value is forced.
    min_diag_D : float
        Each component of the diagonal of the matrix `D` in a :math:`LDL^H` decomposition
        of the returned matrix is forced to be greater or equal to `min_diag_D`.
        `min_diag_D` must be greater than 0.
        optional, default : The square root of the resolution of the underlying data type.
    max_diag_D : float
        Each component of the diagonal of the matrix `D` in a :math:`LDL^H` decomposition
        of the returned matrix is forced to be lower or equal to `max_diag_D`.
        optional, default : No maximal value is forced.
    min_abs_value_D : float
        Absolute values below `min_abs_value_D` are considered as zero
        in the matrix `D` in a :math:`LDL^H` decomposition of the returned matrix.
        `min_abs_value_D` must be greater or equal to 0.
        optional, default : The square root of the resolution of the underlying data type.
    permutation : str or numpy.ndarray
        The symmetric permutation method that is applied to the matrix before it is decomposed.
        It has to be a value in :const:`matrix.UNIVERSAL_PERMUTATION_METHODS` or
        :const:`matrix.APPROXIMATION_ONLY_PERMUTATION_METHODS`.
        If `A` is sparse, it can also be a value in
        :const:`matrix.SPARSE_ONLY_PERMUTATION_METHODS`.
        It is also possible to directly pass a permutation vector.
        optional, default: The permutation is chosen by the algorithm.
    overwrite_A : bool
        Whether it is allowed to overwrite `A`. Enabling may result in performance gain.
        optional, default: False

    Returns
    -------
    B : numpy.ndarray or scipy.sparse.spmatrix (same type as `A`)
        An approximation of `A` which has a :math:`LDL^H` decomposition.

    Raises
    ------
    matrix.errors.MatrixNotSquareError
        If `A` is not a square matrix.
    matrix.errors.MatrixComplexDiagonalValueError
        If `A` has complex diagonal values.
    """

    # debug info
    matrix.logger.debug(('Calcualting approximation of a matrix with passed values: '
                         'min_diag_B={min_diag_B}, max_diag_B={max_diag_B}, '
                         'min_diag_D={min_diag_D}, max_diag_D={max_diag_D}, '
                         'min_abs_value_D={min_abs_value_D}, '
                         'permutation={permutation}, overwrite_A={overwrite_A}.'
                         ).format(
        min_diag_B=min_diag_B, max_diag_B=max_diag_B,
        min_diag_D=min_diag_D, max_diag_D=max_diag_D, min_abs_value_D=min_abs_value_D,
        permutation=permutation, overwrite_A=overwrite_A))

    # calculate decomposition
    L, d, p, omega, delta = _decomposition(
        A, min_diag_B=min_diag_B, max_diag_B=max_diag_B,
        min_diag_D=min_diag_D, max_diag_D=max_diag_D, min_abs_value_D=min_abs_value_D,
        permutation=permutation, overwrite_A=overwrite_A,
        strict_lower_triangular_only_L=True)

    # init B
    matrix.logger.debug('Calculating approximation matrix B.')
    n = L.shape[0]
    B_dtype = L.dtype
    if np.all(d != 0):
        del L
    is_dense = not matrix.sparse.util.is_sparse(A)
    if overwrite_A:
        B = A
    else:
        if is_dense:
            B = np.empty((n, n), dtype=B_dtype)
        else:
            B = scipy.sparse.lil_matrix((n, n), dtype=B_dtype)

    # populate B
    q = matrix.permute.invert_permutation_vector(p)

    def get_value_i(v, i):
        try:
            return v[i]
        except TypeError:
            return v
        except IndexError:
            assert v.ndim == 0
            return v

    for i in range(n):
        # set diagonal entries
        B_i_i = A[i, i] + delta[i]
        min_B_i_i = get_value_i(min_diag_B, i)
        if min_B_i_i is not None and B_i_i < min_B_i_i:
            assert np.isclose(B_i_i, min_B_i_i)
            B_i_i = min_B_i_i
        max_B_i_i = get_value_i(max_diag_B, i)
        if max_B_i_i is not None and B_i_i > max_B_i_i:
            assert np.isclose(B_i_i, max_B_i_i)
            B_i_i = max_B_i_i
        assert np.isfinite(B_i_i)
        assert np.isreal(B_i_i)
        B[i, i] = B_i_i

        # set upper diagonal entries
        for j in range(i + 1, n):
            if q[i] > q[j]:
                a = j
                b = i
            else:
                a = i
                b = j
            assert q[a] < q[b]
            if d[q[a]] != 0:
                B_i_j = A[i, j] * omega[b]
            elif omega[b] == 0:
                B_i_j = 0
            else:
                if q[a] > 0:
                    if is_dense:
                        r_i = L[q[i], :q[a]] * d[:q[a]]
                    else:
                        r_i = L[q[i], :q[a]].multiply(d[:q[a]])
                    B_i_j = r_i @ L[q[j], :q[a]].T.conj()
                    if not is_dense:
                        assert B_i_j.shape == (1, 1)
                        B_i_j = B_i_j[0, 0]
                else:
                    B_i_j = 0
            assert np.isfinite(B_i_j)
            B[i, j] = B_i_j

    # set lower diagonal entries
    for i in range(n):
        for j in range(i + 1, n):
            B[j, i] = np.conj(B[i, j])

    # return B
    if not is_dense:
        B = B.asformat(A.format)
    matrix.logger.debug('Approximation matrix B calculated.')
    return B


def positive_definite_matrix(
        A, min_diag_B=None, max_diag_B=None, min_diag_D=None, max_diag_D=None,
        permutation=None, overwrite_A=False):
    """
    Computes a positive definite approximation of `A`.

    Returns `A` if `A` is positive definite and otherwise an approximation of `A`.

    Parameters
    ----------
    A : numpy.ndarray or scipy.sparse.spmatrix
        The matrix that should be approximated.
        `A` must be Hermitian.
    min_diag_B : numpy.ndarray or float
        Each component of the diagonal of the returned matrix
        is forced to be greater or equal to `min_diag_B`.
        optional, default : No minimal value is forced.
    max_diag_B : numpy.ndarray or float
        Each component of the diagonal of the returned matrix
        is forced to be lower or equal to `max_diag_B`.
        optional, default : No maximal value is forced.
    min_diag_D : float
        Each component of the diagonal of the matrix `D` in a :math:`LDL^H` decomposition
        of the returned matrix is forced to be greater or equal to `min_diag_D`.
        `min_diag_D` must be greater than 0.
        optional, default : The square root of the resolution of the underlying data type.
    max_diag_D : float
        Each component of the diagonal of the matrix `D` in a :math:`LDL^H` decomposition
        of the returned matrix is forced to be lower or equal to `max_diag_D`.
        optional, default : No maximal value is forced.
    permutation : str or numpy.ndarray
        The symmetric permutation method that is applied to the matrix before it is decomposed.
        It has to be a value in :const:`matrix.UNIVERSAL_PERMUTATION_METHODS` or
        :const:`matrix.APPROXIMATION_ONLY_PERMUTATION_METHODS`.
        If `A` is sparse, it can also be a value in
        :const:`matrix.SPARSE_ONLY_PERMUTATION_METHODS`.
        It is also possible to directly pass a permutation vector.
        optional, default: The permutation is chosen by the algorithm.
    overwrite_A : bool
        Whether it is allowed to overwrite `A`. Enabling may result in performance gain.
        optional, default: False

    Returns
    -------
    B : numpy.ndarray or scipy.sparse.spmatrix (same type as `A`)
        An approximation of `A` which is positive definite.

    Raises
    ------
    matrix.errors.MatrixNotSquareError
        If `A` is not a square matrix.
    matrix.errors.MatrixComplexDiagonalValueError
        If `A` has complex diagonal values.
    """

    if min_diag_D is None:
        min_diag_D = np.finfo(np.float64).eps**(0.5)
    elif min_diag_D <= 0:
        raise ValueError('min_diag_D must be greater than zero.')

    return _matrix(
        A, min_diag_B=min_diag_B, max_diag_B=max_diag_B,
        min_diag_D=min_diag_D, max_diag_D=max_diag_D,
        permutation=permutation, overwrite_A=overwrite_A)


def _minimal_change(alpha, beta, gamma, min_diag_D, max_diag_D=np.inf,
                    min_diag_B=-np.inf, max_diag_B=np.inf, min_abs_value_D=0):

    # debug info
    matrix.logger.debug(('Calculating best d and omega for alpha {}, beta {}, gamma {}, '
                         'min_diag_D {}, max_diag_D {}, min_abs_value_D {} and '
                         'min_diag_B {}, max_diag_B {}.'
                         ).format(alpha, beta, gamma, min_diag_D, max_diag_D, min_abs_value_D,
                                  min_diag_B, max_diag_B))

    # check input
    assert np.isfinite(alpha) or alpha == np.inf
    assert np.isfinite(beta) or beta == np.inf
    assert np.isfinite(gamma)
    assert np.isfinite(min_diag_D)
    assert np.isfinite(min_abs_value_D)
    assert alpha >= 0
    assert beta >= 0
    assert beta != 0 or alpha == 0
    assert min_diag_D > 0
    assert min_abs_value_D > 0
    assert max(min_diag_D, min_abs_value_D, min_diag_B) <= min(max_diag_D, max_diag_B)

    # define difference function
    def f(d, omega):
        if omega == 0:
            f_value = (d - gamma)**2 + beta
        elif omega == 1:
            f_value = (d + alpha - gamma)**2
        else:
            f_value = (d + omega**2 * alpha - gamma)**2 + (omega - 1)**2 * beta
        assert f_value >= 0
        return f_value

    # global solution
    d = gamma - alpha
    if max(min_diag_D, min_abs_value_D, min_diag_B - alpha) <= d <= min(max_diag_D, max_diag_B - alpha):
        assert alpha != np.inf
        omega = 1
        f_value = 0
        assert np.isclose(f(d, omega), f_value)

    # solution at bounds
    else:
        # alpha == 0 or alpha == inf or beta == inf
        if alpha == 0 or alpha == np.inf or beta == np.inf:
            d = max(min_diag_D, min_abs_value_D, min_diag_B, min(gamma, max_diag_D, max_diag_B))
            if alpha == np.inf:
                matrix.logger.warning(('Alpha is infinity so omega is forced to be zero. '
                                       'Maybe the datatype should be changed to a more accurate one.'))
                omega = 0
            elif beta == np.inf:
                matrix.logger.warning(('Beta is infinity so omega is forced to be zero. '
                                       'Maybe the datatype should be changed to a more accurate one.'))
                omega = 0
            else:
                omega = 1
            f_value = f(d, omega)

        # 0 < alpha < inf and 0 < beta < inf
        else:
            # prepare candidate set
            C = []

            # omega on bound
            assert alpha > 0
            assert beta > 0
            a = max(min_diag_D, min_abs_value_D)
            b = min(max_diag_D, max_diag_B)
            for d in (min_diag_B - alpha, max_diag_B - alpha):
                if np.isfinite(d) and a <= d <= b:
                    C.append((d, 1))
            if max(a, min_diag_B) <= gamma <= b:
                C.append((gamma, 0))

            # d on bound
            d_values = [a]
            if np.isfinite(b):
                d_values.append(b)
            for d in d_values:
                assert np.isfinite(d)
                # get roots
                omegas = np.roots([2 * alpha**2, 0, 2 * alpha * (d - gamma) + beta, - beta])
                assert len(omegas) == 3
                # use only real roots
                omegas = tuple(omega.real for omega in omegas if np.isreal(omega))
                assert len(omegas) in (1, 3)
                # apply bounds and add to candidate list
                omega_lower = (max(min_diag_B - d, 0) / alpha)**0.5
                assert omega_lower >= 0
                omega_upper = ((max_diag_B - d) / alpha)**0.5
                assert omega_upper >= omega_lower
                for omega in omegas:
                    omega = min(max(omega, omega_lower), omega_upper)
                    C.append((d, omega))

            # calculate function values for candidates
            C_with_f_value = ((d, omega, f(d, omega)) for (d, omega) in C)

            # return best values
            (d, omega, f_value) = min(C_with_f_value, key=lambda x: (x[2], -x[0], x[1]))

    # return value
    assert min_diag_D <= d <= max_diag_D
    assert d >= min_abs_value_D or (d == 0 and min_diag_D <= 0 and max_diag_D >= 0)
    assert omega >= 0
    assert f_value >= 0
    assert np.isfinite(alpha) or alpha == np.inf
    assert alpha != np.inf or omega == 0
    assert alpha == np.inf or (min_diag_B <= d + omega**2 * alpha or np.isclose(min_diag_B, d + omega**2 * alpha))
    assert alpha == np.inf or (max_diag_B >= d + omega**2 * alpha or np.isclose(max_diag_B, d + omega**2 * alpha))
    assert alpha != np.inf or (min_diag_B <= d or np.isclose(min_diag_B, d))
    assert alpha != np.inf or (max_diag_B >= d or np.isclose(max_diag_B, d))
    return (d, omega, f_value)
