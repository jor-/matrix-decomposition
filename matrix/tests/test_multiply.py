import numpy as np
import pytest

import matrix.constants
import matrix.errors
import matrix.tests.random


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
