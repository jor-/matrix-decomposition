import os.path
import tempfile

import numpy as np
import pytest

import matrix
import matrix.calculate
import matrix.constants
import matrix.decompositions
import matrix.permute
import matrix.tests.random


# *** approximate ***

def supported_permutation_methods(dense):
    if dense:
        return matrix.UNIVERSAL_PERMUTATION_METHODS
    else:
        return matrix.UNIVERSAL_PERMUTATION_METHODS + matrix.SPARSE_ONLY_PERMUTATION_METHODS


test_approximate_setups = [
    (n, dense, complex_values, permutation_method, check_finite, return_type, t, min_diag_value, max_diag_value_shift, min_abs_value, overwrite_A)
    for n in (10,)
    for dense in (True, False)
    for complex_values in (True, False)
    for permutation_method in supported_permutation_methods(dense)
    for check_finite in (True, False)
    for return_type in matrix.DECOMPOSITION_TYPES
    for min_abs_value in (None, 10**-8)
    for min_diag_value in (None, 10**-4)
    for max_diag_value_shift in (None, 1 + np.random.rand(1) * 10)
    for t in (None, matrix.tests.random.vector(n) + 10**-4)
    for overwrite_A in (True, False)
]


@pytest.mark.parametrize('n, dense, complex_values, permutation_method, check_finite, return_type, t, min_diag_value, max_diag_value_shift, min_abs_value, overwrite_A', test_approximate_setups)
def test_approximate_decomposition(n, dense, complex_values, permutation_method, check_finite, return_type, t, min_diag_value, max_diag_value_shift, min_abs_value, overwrite_A):
    # init values
    if t is None:
        if min_diag_value is None:
            A_min_diag_value = 10**-6
        else:
            A_min_diag_value = min_diag_value
    else:
        A_min_diag_value = None
        assert min_diag_value is None or min_diag_value <= t.min()

    A = matrix.tests.random.hermitian_matrix(n, dense=dense, complex_values=complex_values, min_diag_value=A_min_diag_value)

    if t is None:
        t_used = A.diagonal()
    else:
        t_used = t
    if max_diag_value_shift is not None:
        max_diag_value = t_used.max()
        max_diag_value = max_diag_value + max_diag_value_shift
    else:
        max_diag_value = None

    # calculate approximated decomposition
    if overwrite_A:
        A_copied = A.copy()
    else:
        A_copied = A

    decomposition = matrix.approximate_decomposition(A_copied, t=t, min_diag_value=min_diag_value, max_diag_value=max_diag_value, min_abs_value=min_abs_value, permutation_method=permutation_method, return_type=return_type, check_finite=check_finite, overwrite_A=overwrite_A)

    # check overwrite_A
    if not overwrite_A:
        assert matrix.util.is_equal(A, A_copied)

    # check d values
    if min_diag_value is not None or max_diag_value is not None:
        decomposition = decomposition.as_type(matrix.LDL_DECOMPOSITION_TYPE)
        assert np.all(np.isreal(decomposition.d))
        if min_diag_value is not None:
            assert np.all(decomposition.d >= min_diag_value)
        if max_diag_value is not None:
            assert np.all(decomposition.d <= max_diag_value)

    # calculate reduction factors
    if overwrite_A:
        A_copied = A.copy()

    with tempfile.TemporaryDirectory() as tmp_dir:
        reduction_factors_file = os.path.join(tmp_dir, 'reduction_factors_file.npy')
        decomposition = matrix.calculate.approximate_decomposition_with_reduction_factor_file(A_copied, t=t, min_diag_value=min_diag_value, max_diag_value=max_diag_value, min_abs_value=min_abs_value, permutation_method=permutation_method, check_finite=check_finite, return_type=return_type, reduction_factors_file=reduction_factors_file, overwrite_A=overwrite_A)
        reduction_factors = np.load(reduction_factors_file)

    # check overwrite_A
    if not overwrite_A:
        assert matrix.util.is_equal(A, A_copied)

    # calculate approximation with reduction factors
    if overwrite_A:
        A_copied = A.copy()

    A_approximated = matrix.calculate.approximate_decomposition_apply_reduction_factors(A_copied, reduction_factors, t=t, min_abs_value=min_abs_value, overwrite_A=overwrite_A)

    # check overwrite_A
    if not overwrite_A:
        assert matrix.util.is_equal(A, A_copied)

    # check approximation with reduction factors
    assert matrix.util.is_almost_equal(decomposition.composed_matrix, A_approximated, atol=1e-06)


test_approximate_positive_definite_matrix_setups = [
    (n, dense, complex_values, positive_definiteness_parameter, check_finite, min_abs_value, overwrite_A)
    for n in (10,)
    for dense in (True, False)
    for positive_definiteness_parameter in (None, 10**-4)
    for complex_values in (True, False)
    for check_finite in (True, False)
    for min_abs_value in (10**-7,)
    for overwrite_A in (True, False)
]


@pytest.mark.parametrize('n, dense, complex_values, positive_definiteness_parameter, check_finite, min_abs_value, overwrite_A', test_approximate_positive_definite_matrix_setups)
def test_approximate_positive_definite_matrix(n, dense, complex_values, positive_definiteness_parameter, check_finite, min_abs_value, overwrite_A):
    # init
    A = matrix.tests.random.hermitian_matrix(n, dense=dense, complex_values=complex_values)
    if overwrite_A:
        A_copied = A.copy()
    else:
        A_copied = A

    # calculate positive definite approximation
    A_positive_definite = matrix.approximate_positive_definite_matrix(A_copied, positive_definiteness_parameter=positive_definiteness_parameter, min_abs_value=min_abs_value, check_finite=check_finite, overwrite_A=overwrite_A)
    assert matrix.is_positive_definite(A_positive_definite, check_finite=True)

    # check overwrite_A
    if not overwrite_A:
        assert matrix.util.is_equal(A, A_copied)
