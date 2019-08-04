import pytest

import matrix.constants
import matrix.tests.random


# *** convert *** #

test_convert_setups = [
    (n_1, n_2, dense_1, dense_2, complex_values_1, complex_values_2, type_str_1, type_str_2)
    for n_1 in (3, 10)
    for n_2 in (3, 10)
    for dense_1 in (True, False)
    for dense_2 in (True, False)
    for complex_values_1 in (True, False)
    for complex_values_2 in (True, False)
    for type_str_1 in matrix.constants.DECOMPOSITION_TYPES
    for type_str_2 in matrix.constants.DECOMPOSITION_TYPES
]


@pytest.mark.parametrize('n_1, n_2, dense_1, dense_2, complex_values_1, complex_values_2, type_str_1, type_str_2', test_convert_setups)
def test_convert(n_1, n_2, dense_1, dense_2, complex_values_1, complex_values_2, type_str_1, type_str_2):
    dec_1 = matrix.tests.random.decomposition(n_1, type_str=type_str_1, dense=dense_1, complex_values=complex_values_1, positive_semidefinite=True, invertible=True)
    dec_2 = matrix.tests.random.decomposition(n_2, type_str=type_str_2, dense=dense_2, complex_values=complex_values_2, positive_semidefinite=True, invertible=True)
    A = dec_1.composed_matrix
    B = dec_2.composed_matrix
    E = matrix.util.block_diag(A, B)
    dec = dec_1.append_block_decomposition(dec_2)
    D = dec.composed_matrix
    assert D.shape == (n_1 + n_2, n_1 + n_2)
    assert matrix.util.is_almost_equal(D, E, atol=1e-06)
