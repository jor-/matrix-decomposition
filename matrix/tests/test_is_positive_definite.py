import pytest

import matrix
import matrix.tests.random


# *** is positive definite *** #

test_is_positive_definite_setups = [
    (n, dense, complex_values)
    for n in (10,)
    for dense in (True, False)
    for complex_values in (True, False)
]


@pytest.mark.parametrize('n, dense, complex_values', test_is_positive_definite_setups)
def test_is_positive_definite(n, dense, complex_values):
    A = matrix.tests.random.hermitian_matrix(n, dense=dense, complex_values=complex_values, positive_semidefinite=True, invertible=True)
    assert matrix.is_positive_semidefinite(A)
    assert not matrix.is_positive_semidefinite(-A)
    assert matrix.is_positive_definite(A)
    assert not matrix.is_positive_definite(-A)
