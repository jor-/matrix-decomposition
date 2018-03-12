import numpy as np
import pytest

import matrix
import matrix.constants
import matrix.permute
import matrix.tests.random


# *** permutation vector *** #

test_permutation_vector_setups = [
    (n, dense, permutation_method)
    for n in (10,)
    for dense in (True, False)
    for permutation_method in (matrix.UNIVERSAL_PERMUTATION_METHODS +
                               matrix.SPARSE_ONLY_PERMUTATION_METHODS + ('DUMMY_METHOD',))
]


@pytest.mark.parametrize('n, dense, permutation_method', test_permutation_vector_setups)
def test_permutation_vector(n, dense, permutation_method):
    A = matrix.tests.random.hermitian_matrix(n, dense=dense)
    if (permutation_method in matrix.constants.UNIVERSAL_PERMUTATION_METHODS or
            (permutation_method in matrix.constants.SPARSE_ONLY_PERMUTATION_METHODS and
             not dense)):
        p = matrix.permute.permutation_vector(A, permutation_method=permutation_method)
        assert p is not None
        assert p.ndim == 1
        p.sort()
        np.testing.assert_array_equal(p, np.arange(n))
    else:
        with np.testing.assert_raises(ValueError):
            matrix.permute.permutation_vector(A, permutation_method=permutation_method)


# *** permute matrix *** #

test_permute_matrix_setups = [
    (n, dense, complex_values)
    for n in (10,)
    for dense in (True, False)
    for complex_values in (True, False)
]


@pytest.mark.parametrize('n, dense, complex_values', test_permute_matrix_setups)
def test_permute_matrix(n, dense, complex_values):
    p = matrix.tests.random.permutation_vector(n)
    A = matrix.tests.random.hermitian_matrix(n, dense=dense, complex_values=complex_values)
    if dense:
        permuted_matrices = (matrix.permute.symmetric(A, p), )
    else:
        permuted_matrices = (matrix.permute.symmetric(A_converted, p).tocsc(copy=False) for A_converted in (A.tocoo(copy=False), A.tocsr(copy=False), A.tocsc(copy=False)))

    for A_permuted in permuted_matrices:
        for i in range(n):
            for j in range(n):
                assert A[p[i], p[j]] == A_permuted[i, j]
        p_inverse = matrix.permute.invert_permutation_vector(p)
        np.testing.assert_array_equal(p[p_inverse], np.arange(n))
        np.testing.assert_array_equal(p_inverse[p], np.arange(n))
