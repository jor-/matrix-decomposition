import warnings

import numpy as np
import scipy.sparse
import pytest

import matrix
import matrix.approximate
import matrix.constants
import matrix.tests.random
import matrix.util


# *** approximate *** #


def supported_permutation_methods(dense):
    if dense:
        return matrix.APPROXIMATION_PERMUTATION_METHODS
    else:
        return matrix.APPROXIMATION_PERMUTATION_METHODS + matrix.SPARSE_ONLY_PERMUTATION_METHODS


test_approximate_dense_sparse_same_setups = [
    (n, complex_values, min_diag_B, max_diag_B, min_diag_D, max_diag_D, min_abs_value_D)
    for n in (10,)
    for complex_values in (True, False)
    for min_diag_B in (None, 1, np.arange(n) / n)
    for max_diag_B in (None, 1, np.arange(n) + 1)
    for min_diag_D in (None, 1)
    for max_diag_D in (None, 1)
    for min_abs_value_D in (None, 0.01)
]


@pytest.mark.parametrize(('n, complex_values, min_diag_B, max_diag_B,'
                          'min_diag_D, max_diag_D, min_abs_value_D'),
                         test_approximate_dense_sparse_same_setups)
def test_approximate_dense_sparse_same(n, complex_values, min_diag_B, max_diag_B,
                                       min_diag_D, max_diag_D, min_abs_value_D):

    # create random hermitian matrix
    A_sparse = matrix.tests.random.hermitian_matrix(n, dense=False, complex_values=complex_values) * 10
    A_sparse = A_sparse.tocsc(copy=False)
    A_dense = A_sparse.todense()

    # approximate decompositions
    permutation = 'none'
    overwrite_A = False

    decomposition_sparse = matrix.approximate.decomposition(
        A_sparse, permutation=permutation,
        min_diag_B=min_diag_B, max_diag_B=max_diag_B,
        min_diag_D=min_diag_D, max_diag_D=max_diag_D, min_abs_value_D=min_abs_value_D,
        overwrite_A=overwrite_A)

    decomposition_dense = matrix.approximate.decomposition(
        A_dense, permutation=permutation,
        min_diag_B=min_diag_B, max_diag_B=max_diag_B,
        min_diag_D=min_diag_D, max_diag_D=max_diag_D, min_abs_value_D=min_abs_value_D,
        overwrite_A=overwrite_A)

    assert decomposition_sparse.is_almost_equal(decomposition_dense)


test_approximate_setups = [
    (n, dense, complex_values, permutation, min_diag_B, max_diag_B, min_diag_D, max_diag_D,
     min_abs_value_D, overwrite_A)
    for n in (10,)
    for dense in (True, False)
    for complex_values in (True, False)
    for permutation in supported_permutation_methods(dense) + (matrix.tests.random.permutation_vector(n),)
    for min_diag_B in (None, 1, np.arange(n) / n)
    for max_diag_B in (None, 1, np.arange(n) + 1)
    for min_diag_D in (None, 1)
    for max_diag_D in (None, 1)
    for min_abs_value_D in (None, 0.01)
    for overwrite_A in (False, True)
]


@pytest.mark.parametrize(('n, dense, complex_values, permutation, min_diag_B, max_diag_B,'
                          'min_diag_D, max_diag_D, min_abs_value_D, overwrite_A'),
                         test_approximate_setups)
def test_approximate(n, dense, complex_values, permutation, min_diag_B, max_diag_B,
                     min_diag_D, max_diag_D, min_abs_value_D, overwrite_A):
    # create random hermitian matrix
    A = matrix.tests.random.hermitian_matrix(n, dense=dense, complex_values=complex_values) * 10
    if not dense:
        A = A.tocsc(copy=False)
    if overwrite_A:
        A_copy = A.copy()
    else:
        A_copy = A

    # suppress sparse efficiency warning at overwrite
    if overwrite_A:
        sparse_efficiency_warning_action = 'ignore'
    else:
        sparse_efficiency_warning_action = 'default'

    # create approximated decomposition
    with warnings.catch_warnings():
        warnings.simplefilter(sparse_efficiency_warning_action,
                              scipy.sparse.SparseEfficiencyWarning)
        decomposition = matrix.approximate.decomposition(
            A, permutation=permutation,
            min_diag_B=min_diag_B, max_diag_B=max_diag_B,
            min_diag_D=min_diag_D, max_diag_D=max_diag_D, min_abs_value_D=min_abs_value_D,
            overwrite_A=overwrite_A)

    # check overwrite
    assert overwrite_A or matrix.util.is_equal(A, A_copy)

    # check values in decomposition
    d = decomposition.d
    assert np.all(np.isfinite(d))
    assert np.all(np.isreal(d))
    assert min_diag_D is None or np.all(d >= min_diag_D)
    assert max_diag_D is None or np.all(d <= max_diag_D)
    assert min_abs_value_D is None or np.all(np.logical_or(np.abs(d) >= min_abs_value_D, d == 0))

    L = decomposition.L
    if not dense:
        L = L.tocsr(copy=False).data
    assert np.all(np.isfinite(L))
    assert complex_values or np.all(np.isreal(L))

    # reset A after overwrite
    A = A_copy.copy() if overwrite_A else A

    # create approximated positive semidefinite matrix
    with warnings.catch_warnings():
        warnings.simplefilter(sparse_efficiency_warning_action,
                              scipy.sparse.SparseEfficiencyWarning)
        B = matrix.approximate.positive_semidefinite_matrix(
            A, permutation=permutation,
            min_diag_B=min_diag_B, max_diag_B=max_diag_B,
            min_diag_D=min_diag_D, max_diag_D=max_diag_D, min_abs_value_D=min_abs_value_D,
            overwrite_A=overwrite_A)

    # check overwrite
    assert overwrite_A or matrix.util.is_equal(A, A_copy)

    # check values in approximation matrix
    if dense:
        B_data = B
    else:
        B_data = B.data
    assert np.all(np.isfinite(B_data))
    assert complex_values or np.all(np.isreal(B_data))
    B_diagonal = B.diagonal()
    assert np.all(np.isreal(B_diagonal))
    assert min_diag_B is None or np.all(B_diagonal >= min_diag_B)
    assert max_diag_B is None or np.all(B_diagonal <= max_diag_B)

    # check if approximation are the same
    assert matrix.util.is_almost_equal(decomposition.composed_matrix, B)

    # checks for positive definite matrix
    do_positive_defintie_checks = min_diag_D is not None and min_diag_D > 0
    if do_positive_defintie_checks:
        # reset A after overwrite
        A = A_copy.copy() if overwrite_A else A

        # create approximated positive definite matrix
        with warnings.catch_warnings():
            warnings.simplefilter(sparse_efficiency_warning_action,
                                  scipy.sparse.SparseEfficiencyWarning)
            C = matrix.approximate.positive_definite_matrix(
                A, permutation=permutation,
                min_diag_B=min_diag_B, max_diag_B=max_diag_B,
                min_diag_D=min_diag_D, max_diag_D=max_diag_D, min_abs_value_D=min_abs_value_D,
                overwrite_A=overwrite_A)

        # check overwrite
        assert overwrite_A or matrix.util.is_equal(A, A_copy)

        # check if approximation are the same
        assert matrix.util.is_equal(B, C)


test_approximate_invariance_setups = [
    (n, dense, complex_values, min_diag_B, max_diag_B, min_diag_D, max_diag_D, min_abs_value_D)
    for n in (10,)
    for dense in (True, False)
    for complex_values in (True, False)
    for min_diag_B in (None, 1, np.arange(n) / n)
    for max_diag_B in (None, 1, np.arange(n) + 1)
    for min_diag_D in (None, 1)
    for max_diag_D in (None, 1)
    for min_abs_value_D in (None, 0.01)
]


@pytest.mark.parametrize(('n, dense, complex_values, min_diag_B, max_diag_B,'
                          'min_diag_D, max_diag_D, min_abs_value_D'),
                         test_approximate_invariance_setups)
def test_approximate_invariance(n, dense, complex_values, min_diag_B, max_diag_B,
                                min_diag_D, max_diag_D, min_abs_value_D):
    # create random hermitian matrix
    A = matrix.tests.random.hermitian_matrix(n, dense=dense, complex_values=complex_values) * 10
    if not dense:
        A = A.tocsc(copy=False)

    # approximate matrix
    permutation = 'none'
    overwrite_A = False
    B = matrix.approximate.positive_semidefinite_matrix(
        A, permutation=permutation,
        min_diag_B=min_diag_B, max_diag_B=max_diag_B,
        min_diag_D=min_diag_D, max_diag_D=max_diag_D, min_abs_value_D=min_abs_value_D,
        overwrite_A=overwrite_A)

    # approximate again matrix
    B_invariant = matrix.approximate.positive_semidefinite_matrix(
        B, permutation=permutation,
        min_diag_B=min_diag_B, max_diag_B=max_diag_B,
        min_diag_D=min_diag_D, max_diag_D=max_diag_D, min_abs_value_D=min_abs_value_D,
        overwrite_A=overwrite_A)
    assert matrix.util.is_almost_equal(B, B_invariant, rtol=1e-04, atol=1e-06)

    # checks for positive definite matrix
    do_positive_defintie_checks = min_diag_D is not None and min_diag_D > 0
    if do_positive_defintie_checks:
        B_invariant = matrix.approximate.positive_definite_matrix(
            B, permutation=permutation,
            min_diag_B=min_diag_B, max_diag_B=max_diag_B,
            min_diag_D=min_diag_D, max_diag_D=max_diag_D, min_abs_value_D=min_abs_value_D,
            overwrite_A=overwrite_A)
        assert matrix.util.is_almost_equal(B, B_invariant, rtol=1e-04, atol=1e-06)
