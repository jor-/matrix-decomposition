import numpy as np

import matrix.decompositions


def _modified_LDLt(A, choose_d, choose_d_state=None, overwrite_A=False):
    choose_d_state = choose_d_state if choose_d_state is not None else {}
    A = A if overwrite_A else A.copy()
    # decompose
    n = len(A)
    p = np.arange(n)
    L = np.eye(n)
    d = np.empty(n)
    delta = np.empty(n)
    for k in range(n):
        # pivot diagonal element of maximum magnitude
        j = np.argmax(np.abs(np.diag(A)))
        j = 0
        p[k], p[k + j] = p[k + j], p[k]
        A[0, :], A[j, :] = A[j, :], A[0, :].copy()
        A[:, 0], A[:, j] = A[:, j], A[:, 0].copy()
        # update L and d
        a = A[0, 0]
        c = A[1:, 0]
        A = A[1:, 1:]
        d[k], choose_d_state = choose_d(a, c, A, choose_d_state)
        assert d[k] >= 0
        delta[p[k]] = d[k] - a
        L[k + 1:, k] = c / d[k]
        A -= np.outer(c, c) / d[k]
    # return
    decomposition = matrix.decompositions.LDL_Decomposition(L=L, d=d, p=p)
    assert decomposition.is_positive_definite()
    decomposition.delta = delta
    return decomposition


def _modified_A(A, choose_d, choose_d_state=None, min_diag_B=None, max_diag_B=None, overwrite_A=False):
    # calculate modified LDL decomposition
    decomposition = _modified_LDLt(A, choose_d, choose_d_state=choose_d_state, overwrite_A=False)
    delta = decomposition.delta
    # calculate B
    B = A if overwrite_A else A.copy()
    n = len(A)
    for i in range(n):
        B[i, i] += delta[i]
    assert np.allclose(decomposition.composed_matrix, B)
    # set diagonal diagonal values
    if min_diag_B is not None or max_diag_B is not None:
        min_diag_B = -np.inf if min_diag_B is None else min_diag_B
        max_diag_B = np.inf if max_diag_B is None else max_diag_B
        diag_B_old = B.diagonal()
        assert np.all(diag_B_old > 0)
        diag_B_new = np.minimum(np.maximum(diag_B_old, min_diag_B), max_diag_B)
        S = np.diag((diag_B_new / diag_B_old)**0.5)
        B = S @ B @ S
        for i in range(n):
            B[i, i] = diag_B_new[i]
        assert np.all(min_diag_B <= B.diagonal())
        assert np.all(max_diag_B >= B.diagonal())
    # return
    assert matrix.calculate.is_positive_definite(B)
    return B


def _approximate(A, choose_d, init_choose_d_state, min_diag_B=None, max_diag_B=None, min_diag_D=None, overwrite_A=False, use_two_phases=True, nondecreasing_startegy=True, use_abs=False, revisited_version_mu=None):
    # check input
    A = np.asarray(A)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise matrix.errors.MatrixNotSquareError(A)
    assert np.all(A == A.T)

    if min_diag_D is None:
        dtype = np.float64
        eps = np.finfo(dtype).eps
        min_diag_D = eps**0.5
    assert min_diag_D > 0

    # init phase
    phase = 1 if use_two_phases else 1.5
    if phase == 1:
        n = len(A)
        if revisited_version_mu is not None:
            max_abs_diag_A = np.abs(A.diagonal()).max()
            phase_1_min_d_next = -revisited_version_mu * max_abs_diag_A
            phase_1_min_d_factor = -revisited_version_mu
        else:
            phase_1_min_d_next = min_diag_D
            phase_1_min_d_factor = -np.inf

    # init choose_d_state
    choose_d_state = {'phase': phase, 'previous_delta': 0}

    # define how to choose d
    def choose_d_both_phases(a, c, A, choose_d_state):
        if choose_d_state['phase'] == 1:
            if a >= min_diag_D and np.all(A.diagonal() - c**2 / a >= phase_1_min_d_next) and np.all(A.diagonal() >= phase_1_min_d_factor * a):
                d = a
            else:
                choose_d_state['phase'] = 1.5

        if choose_d_state['phase'] == 1.5:
            choose_d_state = init_choose_d_state(A, choose_d_state)
            choose_d_state['phase'] = 2

        if choose_d_state['phase'] == 2:
            d, choose_d_state = choose_d(a, c, A, choose_d_state)
            d = max(d, min_diag_D)
            a_changed = np.abs(a) if use_abs else a
            if nondecreasing_startegy:
                a_changed += choose_d_state['previous_delta']
            d = max(d, a_changed)
            choose_d_state['previous_delta'] = d - a

        return d, choose_d_state

    # calculate B
    B = _modified_A(A, choose_d_both_phases, choose_d_state=choose_d_state, min_diag_B=min_diag_B, max_diag_B=max_diag_B, overwrite_A=overwrite_A)
    return B


def _GMW(A, min_diag_B=None, max_diag_B=None, min_diag_D=None, overwrite_A=False, use_two_phases=True, nondecreasing_startegy=True, use_abs=False, revisited_version_mu=0.75):
    # calculate beta squared
    def init_choose_d_state(A, choose_d_state):
        dtype = np.float64
        eps = np.finfo(dtype).eps
        m = len(A)
        if m > 1:
            max_abs_diag_A = 0
            max_abs_off_diag_A = 0
            for i in range(len(A)):
                max_abs_diag_A = max(max_abs_diag_A, np.abs(A[i, i]))
                for j in range(i):
                    max_abs_off_diag_A = max(max_abs_off_diag_A, np.abs(A[i, j]))
            if use_abs:
                x = m**2 - 1
            else:
                x = m**2 - m
            beta_squared = max(eps, max_abs_diag_A, max_abs_off_diag_A / x**0.5)
        elif m == 1:
            beta_squared = max(eps, np.abs(A))
        else:
            beta_squared = eps
        choose_d_state['beta_squared'] = beta_squared
        return choose_d_state

    # define how to choose d
    def choose_d(a, c, A, choose_d_state):
        beta_squared = choose_d_state['beta_squared']
        d = np.linalg.norm(c, ord=np.inf)**2 / beta_squared if len(c) > 0 else 0
        return d, choose_d_state

    B = _approximate(A, choose_d, init_choose_d_state, min_diag_B=min_diag_B, max_diag_B=max_diag_B, min_diag_D=min_diag_D, overwrite_A=overwrite_A, use_two_phases=use_two_phases, nondecreasing_startegy=nondecreasing_startegy, use_abs=use_abs, revisited_version_mu=revisited_version_mu)
    return B


def _SE(A, min_diag_B=None, max_diag_B=None, min_diag_D=None, overwrite_A=False, use_two_phases=True, nondecreasing_startegy=True, use_abs=False, revisited_version_mu=0.1):
    dtype = np.float64
    eps = np.finfo(dtype).eps
    if revisited_version_mu is not None:
        tau = eps**(2 / 3)
    else:
        tau = eps**(1 / 3)

    def init_choose_d_state(A, choose_d_state):
        if len(A) > 0:
            max_abs_diag_A = np.abs(A.diagonal()).max()
        else:
            max_abs_diag_A = 0
        choose_d_state['tau_max_abs_diag_A'] = tau * max_abs_diag_A
        return choose_d_state

    def choose_d(a, c, A, choose_d_state):
        m = len(c)
        # more than two iterations left
        if m > 1:
            d = max(np.linalg.norm(c, ord=1), choose_d_state['tau_max_abs_diag_A'])
        # two iterations left
        elif m == 1:
            lambda_1, lambda_2 = np.sort(np.linalg.eigvalsh(np.array([[a, c], [c, A]], dtype=dtype)))
            d = a - lambda_1 + max(tau * (lambda_2 - lambda_1) / (1 - tau), choose_d_state['tau_max_abs_diag_A'])
            if use_abs:
                d = max(d, a - 2 * lambda_1)
            choose_d_state['d_at_last_iteration'] = d
        # one iteration left
        else:
            try:
                d = choose_d_state['d_at_last_iteration']
            except KeyError:
                d = max((- eps**(1 / 3) * a) / (1 - eps**(1 / 3)), choose_d_state['tau_max_abs_diag_A'])
        return d, choose_d_state

    B = _approximate(A, choose_d, init_choose_d_state, min_diag_B=min_diag_B, max_diag_B=max_diag_B, min_diag_D=min_diag_D, overwrite_A=overwrite_A, use_two_phases=use_two_phases, nondecreasing_startegy=nondecreasing_startegy, use_abs=use_abs, revisited_version_mu=revisited_version_mu)
    return B


def GMW_81(A, min_diag_B=None, max_diag_B=None, min_diag_D=None, overwrite_A=False):
    """
    Computes a positive definite approximation of `A`.

    Returns `A` if `A` is positive definite and meets the constrains and otherwise a positive
    definite approximation of `A`.

    Parameters
    ----------
    A : numpy.ndarray
        The matrix that should be approximated.
        `A` must be symmetric.
    min_diag_B : numpy.ndarray or float
        Each component of the diagonal of the returned matrix
        is forced to be greater or equal to `min_diag_B`.
        optional, default : No minimal value is forced.
    max_diag_B : numpy.ndarray or float
        Each component of the diagonal of the returned matrix
        is forced to be lower or equal to `max_diag_B`.
        optional, default : No maximal value is forced.
    min_diag_D : float
        Each component of the diagonal of the matrix `D` in a :math:`LDL^T` decomposition
        of the returned matrix is forced to be greater or equal to `min_diag_D`.
        `min_diag_D` must be positive.
        optional, default : Is chosen by the algorithm.
    overwrite_A : bool
        Whether it is allowed to overwrite `A`. Enabling may result in performance gain.
        optional, default: False

    Returns
    -------
    B : numpy.ndarray
        An approximation of `A` which is positive definite.

    Raises
    ------
    matrix.errors.MatrixNotSquareError
        If `A` is not a square matrix.

    Notes
    -----
    The algorithm is introduced in [1].
    Is is also described in [2].
    This discription has been used for this implementation.
    The implementation has been extended to allow restrictions on the diagonal values.

    References
    ----------
    [1] Gill, P. E.; Murray, W. & Wright, M. H.,
        Practical optimization,
        Academic press, 1981
    [2] Fang, H.-r. & O'Leary, D. P.,
        Modified Cholesky algorithms: a catalog with new approaches,
        Mathematical Programming, 2008, 115, 319-349
    """

    return _GMW(A, min_diag_B=min_diag_B, max_diag_B=max_diag_B, min_diag_D=min_diag_D, overwrite_A=overwrite_A, use_two_phases=False, nondecreasing_startegy=False, use_abs=True, revisited_version_mu=None)


def GMW_T1(A, min_diag_B=None, max_diag_B=None, min_diag_D=None, overwrite_A=False):
    """
    Computes a positive definite approximation of `A`.

    Returns `A` if `A` is positive definite and meets the constrains and otherwise a positive
    definite approximation of `A`.

    Parameters
    ----------
    A : numpy.ndarray
        The matrix that should be approximated.
        `A` must be symmetric.
    min_diag_B : numpy.ndarray or float
        Each component of the diagonal of the returned matrix
        is forced to be greater or equal to `min_diag_B`.
        optional, default : No minimal value is forced.
    max_diag_B : numpy.ndarray or float
        Each component of the diagonal of the returned matrix
        is forced to be lower or equal to `max_diag_B`.
        optional, default : No maximal value is forced.
    min_diag_D : float
        Each component of the diagonal of the matrix `D` in a :math:`LDL^T` decomposition
        of the returned matrix is forced to be greater or equal to `min_diag_D`.
        `min_diag_D` must be positive.
        optional, default : Is chosen by the algorithm.
    overwrite_A : bool
        Whether it is allowed to overwrite `A`. Enabling may result in performance gain.
        optional, default: False

    Returns
    -------
    B : numpy.ndarray
        An approximation of `A` which is positive definite.

    Raises
    ------
    matrix.errors.MatrixNotSquareError
        If `A` is not a square matrix.

    Notes
    -----
    The algorithm is introduced in [1].
    This discription has been used for this implementation.
    The algorithm is based on [2].
    The implementation has been extended to allow restrictions on the diagonal values.

    References
    ----------
    [1] Fang, H.-r. & O'Leary, D. P.,
        Modified Cholesky algorithms: a catalog with new approaches,
        Mathematical Programming, 2008, 115, 319-349
    [2] Gill, P. E.; Murray, W. & Wright, M. H.,
        Practical optimization,
        Academic press, 1981
    """

    return _GMW(A, min_diag_B=min_diag_B, max_diag_B=max_diag_B, min_diag_D=min_diag_D, overwrite_A=overwrite_A, use_two_phases=True, nondecreasing_startegy=True, use_abs=True, revisited_version_mu=0.75)


def GMW_T2(A, min_diag_B=None, max_diag_B=None, min_diag_D=None, overwrite_A=False):
    """
    Computes a positive definite approximation of `A`.

    Returns `A` if `A` is positive definite and meets the constrains and otherwise a positive
    definite approximation of `A`.

    Parameters
    ----------
    A : numpy.ndarray
        The matrix that should be approximated.
        `A` must be symmetric.
    min_diag_B : numpy.ndarray or float
        Each component of the diagonal of the returned matrix
        is forced to be greater or equal to `min_diag_B`.
        optional, default : No minimal value is forced.
    max_diag_B : numpy.ndarray or float
        Each component of the diagonal of the returned matrix
        is forced to be lower or equal to `max_diag_B`.
        optional, default : No maximal value is forced.
    min_diag_D : float
        Each component of the diagonal of the matrix `D` in a :math:`LDL^T` decomposition
        of the returned matrix is forced to be greater or equal to `min_diag_D`.
        `min_diag_D` must be positive.
        optional, default : Is chosen by the algorithm.
    overwrite_A : bool
        Whether it is allowed to overwrite `A`. Enabling may result in performance gain.
        optional, default: False

    Returns
    -------
    B : numpy.ndarray
        An approximation of `A` which is positive definite.

    Raises
    ------
    matrix.errors.MatrixNotSquareError
        If `A` is not a square matrix.

    Notes
    -----
    The algorithm is introduced in [1].
    This discription has been used for this implementation.
    The algorithm is based on [2].
    The implementation has been extended to allow restrictions on the diagonal values.

    References
    ----------
    [1] Fang, H.-r. & O'Leary, D. P.,
        Modified Cholesky algorithms: a catalog with new approaches,
        Mathematical Programming, 2008, 115, 319-349
    [2] Gill, P. E.; Murray, W. & Wright, M. H.,
        Practical optimization,
        Academic press, 1981
    """

    return _GMW(A, min_diag_B=min_diag_B, max_diag_B=max_diag_B, min_diag_D=min_diag_D, overwrite_A=overwrite_A, use_two_phases=True, nondecreasing_startegy=True, use_abs=False, revisited_version_mu=0.75)


def SE_90(A, min_diag_B=None, max_diag_B=None, min_diag_D=None, overwrite_A=False):
    """
    Computes a positive definite approximation of `A`.

    Returns `A` if `A` is positive definite and meets the constrains and otherwise a positive
    definite approximation of `A`.

    Parameters
    ----------
    A : numpy.ndarray
        The matrix that should be approximated.
        `A` must be symmetric.
    min_diag_B : numpy.ndarray or float
        Each component of the diagonal of the returned matrix
        is forced to be greater or equal to `min_diag_B`.
        optional, default : No minimal value is forced.
    max_diag_B : numpy.ndarray or float
        Each component of the diagonal of the returned matrix
        is forced to be lower or equal to `max_diag_B`.
        optional, default : No maximal value is forced.
    min_diag_D : float
        Each component of the diagonal of the matrix `D` in a :math:`LDL^T` decomposition
        of the returned matrix is forced to be greater or equal to `min_diag_D`.
        `min_diag_D` must be positive.
        optional, default : Is chosen by the algorithm.
    overwrite_A : bool
        Whether it is allowed to overwrite `A`. Enabling may result in performance gain.
        optional, default: False

    Returns
    -------
    B : numpy.ndarray
        An approximation of `A` which is positive definite.

    Raises
    ------
    matrix.errors.MatrixNotSquareError
        If `A` is not a square matrix.

    Notes
    -----
    The algorithm is introduced in [1].
    Is is also described in [2].
    This discription has been used for this implementation.
    The implementation has been extended to allow restrictions on the diagonal values.

    References
    ----------
    [1] Schnabel, R. & Eskow, E.,
        A New Modified Cholesky Factorization,
        SIAM Journal on Scientific and Statistical Computing, 1990, 11, 1136-1158
    [2] Fang, H.-r. & O'Leary, D. P.,
        Modified Cholesky algorithms: a catalog with new approaches,
        Mathematical Programming, 2008, 115, 319-349
    """

    return _SE(A, min_diag_B=min_diag_B, max_diag_B=max_diag_B, min_diag_D=min_diag_D, overwrite_A=overwrite_A, use_two_phases=True, nondecreasing_startegy=True, use_abs=False, revisited_version_mu=None)


def SE_99(A, min_diag_B=None, max_diag_B=None, min_diag_D=None, overwrite_A=False):
    """
    Computes a positive definite approximation of `A`.

    Returns `A` if `A` is positive definite and meets the constrains and otherwise a positive
    definite approximation of `A`.

    Parameters
    ----------
    A : numpy.ndarray
        The matrix that should be approximated.
        `A` must be symmetric.
    min_diag_B : numpy.ndarray or float
        Each component of the diagonal of the returned matrix
        is forced to be greater or equal to `min_diag_B`.
        optional, default : No minimal value is forced.
    max_diag_B : numpy.ndarray or float
        Each component of the diagonal of the returned matrix
        is forced to be lower or equal to `max_diag_B`.
        optional, default : No maximal value is forced.
    min_diag_D : float
        Each component of the diagonal of the matrix `D` in a :math:`LDL^T` decomposition
        of the returned matrix is forced to be greater or equal to `min_diag_D`.
        `min_diag_D` must be positive.
        optional, default : Is chosen by the algorithm.
    overwrite_A : bool
        Whether it is allowed to overwrite `A`. Enabling may result in performance gain.
        optional, default: False

    Returns
    -------
    B : numpy.ndarray
        An approximation of `A` which is positive definite.

    Raises
    ------
    matrix.errors.MatrixNotSquareError
        If `A` is not a square matrix.

    Notes
    -----
    The algorithm is introduced in [1].
    Is is also described in [2].
    This discription has been used for this implementation.
    The algorithm is based on [3].
    The implementation has been extended to allow restrictions on the diagonal values.

    References
    ----------
    [1] Schnabel, R. & Eskow, E.,
        A Revised Modified Cholesky Factorization Algorithm,
        SIAM Journal on Optimization, 1999, 9, 1135-1148
    [2] Fang, H.-r. & O'Leary, D. P.,
        Modified Cholesky algorithms: a catalog with new approaches,
        Mathematical Programming, 2008, 115, 319-349
    [3] Schnabel, R. & Eskow, E.,
        A New Modified Cholesky Factorization,
        SIAM Journal on Scientific and Statistical Computing, 1990, 11, 1136-1158
    """

    return _SE(A, min_diag_B=min_diag_B, max_diag_B=max_diag_B, min_diag_D=min_diag_D, overwrite_A=overwrite_A, use_two_phases=True, nondecreasing_startegy=True, use_abs=False, revisited_version_mu=0.1)


def SE_T1(A, min_diag_B=None, max_diag_B=None, min_diag_D=None, overwrite_A=False):
    """
    Computes a positive definite approximation of `A`.

    Returns `A` if `A` is positive definite and meets the constrains and otherwise a positive
    definite approximation of `A`.

    Parameters
    ----------
    A : numpy.ndarray
        The matrix that should be approximated.
        `A` must be symmetric.
    min_diag_B : numpy.ndarray or float
        Each component of the diagonal of the returned matrix
        is forced to be greater or equal to `min_diag_B`.
        optional, default : No minimal value is forced.
    max_diag_B : numpy.ndarray or float
        Each component of the diagonal of the returned matrix
        is forced to be lower or equal to `max_diag_B`.
        optional, default : No maximal value is forced.
    min_diag_D : float
        Each component of the diagonal of the matrix `D` in a :math:`LDL^T` decomposition
        of the returned matrix is forced to be greater or equal to `min_diag_D`.
        `min_diag_D` must be positive.
        optional, default : Is chosen by the algorithm.
    overwrite_A : bool
        Whether it is allowed to overwrite `A`. Enabling may result in performance gain.
        optional, default: False

    Returns
    -------
    B : numpy.ndarray
        An approximation of `A` which is positive definite.

    Raises
    ------
    matrix.errors.MatrixNotSquareError
        If `A` is not a square matrix.

    Notes
    -----
    The algorithm is introduced in [1].
    This discription has been used for this implementation.
    The algorithm is based on [2].
    The implementation has been extended to allow restrictions on the diagonal values.

    References
    ----------
    [1] Fang, H.-r. & O'Leary, D. P.,
        Modified Cholesky algorithms: a catalog with new approaches,
        Mathematical Programming, 2008, 115, 319-349
    [2] Schnabel, R. & Eskow, E.,
        A Revised Modified Cholesky Factorization Algorithm,
        SIAM Journal on Optimization, 1999, 9, 1135-1148
    [3] Schnabel, R. & Eskow, E.,
        A New Modified Cholesky Factorization,
        SIAM Journal on Scientific and Statistical Computing, 1990, 11, 1136-1158
    """

    return _SE(A, min_diag_B=min_diag_B, max_diag_B=max_diag_B, min_diag_D=min_diag_D, overwrite_A=overwrite_A, use_two_phases=True, nondecreasing_startegy=False, use_abs=True, revisited_version_mu=0.1)


SE_T2 = SE_99
