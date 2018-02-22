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


# *** convert *** #

test_convert_setups = [
    (n, dense, complex_values, type_str, copy)
    for n in (10,)
    for dense in (True, False)
    for complex_values in (True, False)
    for type_str in matrix.constants.DECOMPOSITION_TYPES
    for copy in (True, False)
]


@pytest.mark.parametrize('n, dense, complex_values, type_str, copy', test_convert_setups)
def test_convert(n, dense, complex_values, type_str, copy):
    decomposition = matrix.tests.random.decomposition(n, type_str=type_str, dense=dense, complex_values=complex_values, positive_semi_definite=True, invertible=True)
    for convert_type_str in matrix.constants.DECOMPOSITION_TYPES:
        converted_decomposition = decomposition.as_type(convert_type_str, copy=copy)
        equal = type_str == convert_type_str
        equal_calculated = decomposition == converted_decomposition
        assert equal == equal_calculated


# *** decompose *** #

def supported_permutation_methods(dense):
    if dense:
        return matrix.UNIVERSAL_PERMUTATION_METHODS
    else:
        return matrix.UNIVERSAL_PERMUTATION_METHODS + matrix.SPARSE_ONLY_PERMUTATION_METHODS


test_decompose_setups = [
    (n, dense, complex_values, permutation_method, check_finite, return_type, overwrite_A)
    for n in (10,)
    for dense in (True, False)
    for complex_values in (True, False)
    for permutation_method in supported_permutation_methods(dense)
    for check_finite in (True, False)
    for return_type in matrix.DECOMPOSITION_TYPES
    for overwrite_A in (True, False)
]


@pytest.mark.parametrize('n, dense, complex_values, permutation_method, check_finite, return_type, overwrite_A', test_decompose_setups)
def test_decompose(n, dense, complex_values, permutation_method, check_finite, return_type, overwrite_A):
    A = matrix.tests.random.hermitian_matrix(n, dense=dense, complex_values=complex_values, positive_semi_definite=True, invertible=True)
    if not overwrite_A:
        A_copied = A.copy()
    # decompose
    decomposition = matrix.decompose(A, permutation_method=permutation_method, return_type=return_type, check_finite=check_finite, overwrite_A=overwrite_A)
    # check if decomposition correct
    assert matrix.util.is_almost_equal(decomposition.composed_matrix, A, atol=1e-06)
    # check if A overwritten
    if not overwrite_A:
        assert matrix.util.is_equal(A, A_copied)
    # check if real valued d in LDL decomposition
    decomposition = decomposition.as_type(matrix.LDL_DECOMPOSITION_TYPE)
    assert np.all(np.isreal(decomposition.d))


# *** positive definite *** #

test_positive_definite_setups = [
    (n, dense, complex_values)
    for n in (10,)
    for dense in (True, False)
    for complex_values in (True, False)
]


@pytest.mark.parametrize('n, dense, complex_values', test_positive_definite_setups)
def test_positive_definite(n, dense, complex_values):
    A = matrix.tests.random.hermitian_matrix(n, dense=dense, complex_values=complex_values, positive_semi_definite=True, invertible=True)
    assert matrix.is_positive_semi_definite(A)
    assert not matrix.is_positive_semi_definite(-A)
    assert matrix.is_positive_definite(A)
    assert not matrix.is_positive_definite(-A)


# *** approximate *** #

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


# *** save and load *** #

test_save_and_load_setups = [
    (n, dense, complex_values, type_str)
    for n in (10,)
    for dense in (True, False)
    for complex_values in (True, False)
    for type_str in matrix.constants.DECOMPOSITION_TYPES
]


@pytest.mark.parametrize('n, dense, complex_values, type_str', test_save_and_load_setups)
def test_save_and_load(n, dense, complex_values, type_str):
    decomposition = matrix.tests.random.decomposition(n, type_str=type_str, dense=dense, complex_values=complex_values)
    # test own IO methods
    decomposition_other = type(decomposition)()
    with tempfile.TemporaryDirectory() as tmp_dir:
        file = os.path.join(tmp_dir, 'decomposition')
        decomposition.save(file)
        decomposition_other.load(file)
    assert decomposition == decomposition_other
    # test general IO methods
    with tempfile.TemporaryDirectory() as tmp_dir:
        file = os.path.join(tmp_dir, 'decomposition')
        matrix.decompositions.save(file, decomposition)
        decomposition_other = matrix.decompositions.load(file)
    assert decomposition == decomposition_other


# *** is finite *** #

test_is_finite_setups = [
    (n, dense, complex_values, type_str, finite)
    for n in (10,)
    for dense in (True, False)
    for complex_values in (True, False)
    for type_str in matrix.constants.DECOMPOSITION_TYPES
    for finite in (True, False)
]


@pytest.mark.parametrize('n, dense, complex_values, type_str, finite', test_is_finite_setups)
def test_is_finite(n, dense, complex_values, type_str, finite):
    # make random decomposition
    decomposition = matrix.tests.random.decomposition(n, type_str=type_str, dense=dense, complex_values=complex_values, finite=finite)
    # test
    assert decomposition.is_finite() == finite
    if not finite:
        with np.testing.assert_raises(matrix.errors.DecompositionNotFiniteError):
            decomposition.check_finite()
    else:
        decomposition.check_finite()


# *** is invertible *** #

test_is_invertible_setups = [
    (n, dense, complex_values, type_str, invertible)
    for n in (10,)
    for dense in (True, False)
    for complex_values in (True, False)
    for type_str in matrix.constants.DECOMPOSITION_TYPES
    for invertible in (True, False)
]


@pytest.mark.parametrize('n, dense, complex_values, type_str, invertible', test_is_finite_setups)
def test_is_invertible(n, dense, complex_values, type_str, invertible):
    # make random decomposition
    decomposition = matrix.tests.random.decomposition(n, type_str=type_str, dense=dense, complex_values=complex_values, invertible=invertible)
    # test
    assert decomposition.is_invertible() == invertible
    if not invertible:
        with np.testing.assert_raises(matrix.errors.DecompositionSingularError):
            decomposition.check_invertible()
    else:
        decomposition.check_invertible()


# *** multiply *** #

test_multiply_setups = [
    (n, dense, complex_values, type_str, invertible, x, y)
    for n in (10,)
    for dense in (True, False)
    for complex_values in (True, False)
    for type_str in matrix.constants.DECOMPOSITION_TYPES
    for invertible in (True, False)
    for x in (np.zeros(n), np.arange(n), matrix.tests.random.vector(n), matrix.tests.random.vector(n, complex_values=True), matrix.tests.random.universal_matrix(n, 3), matrix.tests.random.universal_matrix(n, 3, complex_values=True))
    for y in (None, np.zeros(n), np.arange(n), matrix.tests.random.vector(n), matrix.tests.random.vector(n, complex_values=True), matrix.tests.random.universal_matrix(n, 3), matrix.tests.random.universal_matrix(n, 3, complex_values=True))
]


@pytest.mark.parametrize('n, dense, complex_values, type_str, invertible, x, y', test_multiply_setups)
def test_multiply(n, dense, complex_values, type_str, invertible, x, y):
    # prepare decomposition and values
    decomposition = matrix.tests.random.decomposition(n, type_str=type_str, dense=dense, complex_values=complex_values, invertible=invertible)
    A = decomposition.composed_matrix
    if not dense:
        A = A.todense().getA()
    if y is None:
        y_H = x.transpose().conj()
    else:
        y_H = y.transpose().conj()
    # test matrix multiplication
    res = decomposition.matrix_right_side_multiplication(x)
    np.testing.assert_array_almost_equal(res, A @ x)
    res = decomposition.matrix_both_sides_multiplication(x, y)
    np.testing.assert_array_almost_equal(res, y_H @ A @ x)
    # test inverse matrix multiplication
    if invertible:
        B = np.linalg.inv(A)
        res = decomposition.inverse_matrix_right_side_multiplication(x)
        np.testing.assert_array_almost_equal(res, B @ x)
        res = decomposition.inverse_matrix_both_sides_multiplication(x, y)
        np.testing.assert_array_almost_equal(res, y_H @ B @ x)
    else:
        with np.testing.assert_raises(matrix.errors.DecompositionSingularError):
            decomposition.inverse_matrix_right_side_multiplication(x)
        with np.testing.assert_raises(matrix.errors.DecompositionSingularError):
            decomposition.inverse_matrix_both_sides_multiplication(x, y)


# *** solve *** #

test_solve_setups = [
    (n, dense, complex_values, type_str, invertible, b)
    for n in (10,)
    for dense in (True, False)
    for complex_values in (True, False)
    for type_str in matrix.constants.DECOMPOSITION_TYPES
    for invertible in (True, False)
    for b in (matrix.tests.random.vector(n), np.zeros(n), np.arange(n), matrix.tests.random.universal_matrix(n, 2))
]


@pytest.mark.parametrize('n, dense, complex_values, type_str, invertible, b', test_solve_setups)
def test_solve(n, dense, complex_values, type_str, invertible, b):
    # make random decomposition
    decomposition = matrix.tests.random.decomposition(n, type_str=type_str, dense=dense, complex_values=complex_values, finite=True, invertible=invertible)
    if invertible:
        # calculate solution
        x = decomposition.solve(b)
        # verify solution
        A = decomposition.composed_matrix
        y = A @ x
        assert np.all(np.isclose(b, y))
    else:
        with np.testing.assert_raises(matrix.errors.DecompositionSingularError):
            decomposition.solve(b)
