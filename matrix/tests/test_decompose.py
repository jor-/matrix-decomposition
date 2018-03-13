import numpy as np
import pytest

import matrix
import matrix.constants
import matrix.tests.random
import matrix.util


# *** decompose *** #

def supported_permutation_methods(dense):
    if dense:
        return matrix.UNIVERSAL_PERMUTATION_METHODS
    else:
        return matrix.UNIVERSAL_PERMUTATION_METHODS + matrix.SPARSE_ONLY_PERMUTATION_METHODS


test_decompose_setups = [
    (n, dense, complex_values, permutation, check_finite, return_type, overwrite_A)
    for n in (10,)
    for dense in (True, False)
    for complex_values in (True, False)
    for permutation in supported_permutation_methods(dense) + (matrix.tests.random.permutation_vector(n),)
    for check_finite in (True, False)
    for return_type in matrix.DECOMPOSITION_TYPES
    for overwrite_A in (True, False)
]


@pytest.mark.parametrize('n, dense, complex_values, permutation, check_finite, return_type, overwrite_A', test_decompose_setups)
def test_decompose(n, dense, complex_values, permutation, check_finite, return_type, overwrite_A):
    # create Hermitian matrix
    A = matrix.tests.random.hermitian_matrix(n, dense=dense, complex_values=complex_values, positive_semidefinite=True, invertible=True)
    if not overwrite_A:
        A_copy = A.copy()
    # decompose
    decomposition = matrix.decompose(A, permutation=permutation, return_type=return_type, check_finite=check_finite, overwrite_A=overwrite_A)
    # check if decomposition correct
    assert matrix.util.is_almost_equal(decomposition.composed_matrix, A, atol=1e-06)
    # check if A overwritten
    assert overwrite_A or matrix.util.is_equal(A, A_copy)
    # check values in LDL decomposition
    decomposition = decomposition.as_type(matrix.LDL_DECOMPOSITION_TYPE)
    d = decomposition.d
    assert np.all(np.isfinite(d))
    assert np.all(np.isreal(d))
    L = decomposition.L
    if not dense:
        L = L.data
    assert np.all(np.isfinite(L))
    assert complex_values or np.all(np.isreal(L))
