import numpy as np
import scipy.sparse
import pytest

import matrix
import matrix.dense
import matrix.sparse
import matrix.permute


# *** random values *** #

def random_square_matrix(n, dense=True, positive_semi_definite=False):
    random_state = 1234
    if dense:
        np.random.seed(random_state)
        A = np.random.rand(n, n)
        A = np.asmatrix(A)
    else:
        density = 0.1
        A = scipy.sparse.rand(n, n, density=density, random_state=random_state)
    A = A + A.H
    if positive_semi_definite:
        A = A @ A
    return A


def random_permutation_vector(n):
    random_state = 1234
    np.random.seed(random_state)
    p = np.arange(n)
    np.random.shuffle(p)
    return p


# *** permute *** #

test_permute_setups = [
    (n, dense)
    for n in (100,)
    for dense in (True, False)
]


@pytest.mark.parametrize('n, dense', test_permute_setups)
def test_permute(n, dense):
    p = random_permutation_vector(n)
    A = random_square_matrix(n, dense=dense, positive_semi_definite=True)
    A_permuted = matrix.permute.symmetric(A, p)
    for i in range(n):
        for j in range(n):
            assert A[p[i], p[j]] == A_permuted[i, j]
    p_inverse = matrix.permute.invert_permutation_vector(p)
    np.testing.assert_array_equal(p[p_inverse], np.arange(n))
    np.testing.assert_array_equal(p_inverse[p], np.arange(n))
