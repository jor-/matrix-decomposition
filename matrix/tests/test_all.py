import numpy as np
import scipy.sparse
import pytest

import matrix
import matrix.constants
import matrix.decompositions
import matrix.permute


# *** random values *** #

def random_matrix(n, dense=True):
    random_state = 1234
    if dense:
        np.random.seed(random_state)
        A = np.random.rand(n, n)
        A = np.asmatrix(A)
    else:
        density = 0.1
        A = scipy.sparse.rand(n, n, density=density, random_state=random_state)
    return A


def random_square_matrix(n, dense=True, positive_semi_definite=False, positive_definite=False):
    A = random_matrix(n, dense=dense)
    A = A + A.H
    if positive_semi_definite or positive_definite:
        A = A @ A
    if positive_definite:
        min_diag_value = 1
        for i in range(n):
            A[i, i] += min_diag_value
    return A


def random_lower_triangle_matrix(n, dense=True):
    A = random_matrix(n, dense=dense)
    if dense:
        A = np.tril(A)
    else:
        A = scipy.sparse.tril(A).tocsc()
    return A


def random_vector(n):
    random_state = 1234
    np.random.seed(random_state)
    v = np.random.rand(n)
    return v


def random_permutation_vector(n):
    random_state = 1234
    np.random.seed(random_state)
    p = np.arange(n)
    np.random.shuffle(p)
    return p


def random_decomposition(decomposition_type, n, dense=True):
    LD = random_lower_triangle_matrix(n, dense=dense)
    p = random_permutation_vector(n)
    decomposition = matrix.decompositions.LDL_DecompositionCompressed(LD, p)
    decomposition = decomposition.to(decomposition_type)
    return decomposition


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


# *** equal *** #

test_equal_setups = [
    (n, dense, decomposition_type)
    for n in (100,)
    for dense in (True, False)
    for decomposition_type in matrix.constants.DECOMPOSITION_TYPES
]


@pytest.mark.parametrize('n, dense, decomposition_type', test_equal_setups)
def test_equal(n, dense, decomposition_type):
    decomposition = random_decomposition(decomposition_type, n, dense=dense)
    for (n_other, dense_other, decomposition_type_other) in test_equal_setups:
        decomposition_other = random_decomposition(decomposition_type_other, n_other, dense=dense_other)
        equal = n == n_other and dense == dense_other and decomposition_type == decomposition_type_other
        equal_calculated = decomposition == decomposition_other
        assert equal == equal_calculated


# *** convert *** #

test_convert_setups = [
    (n, dense, decomposition_type, copy)
    for n in (100,)
    for dense in (True, False)
    for decomposition_type in matrix.constants.DECOMPOSITION_TYPES
    for copy in (True, False)
]


@pytest.mark.parametrize('n, dense, decomposition_type, copy', test_convert_setups)
def test_convert(n, dense, decomposition_type, copy):
    decomposition = random_decomposition(decomposition_type, n, dense=dense)
    for convert_decomposition_type in matrix.constants.DECOMPOSITION_TYPES:
        converted_decomposition = decomposition.to(convert_decomposition_type, copy=copy)
        equal = decomposition_type == convert_decomposition_type
        equal_calculated = decomposition == converted_decomposition
        assert equal == equal_calculated


# *** decompose *** #

def supported_permutation_methods(dense):
    if dense:
        return matrix.PERMUTATION_METHODS
    else:
        return matrix.SPARSE_PERMUTATION_METHODS


test_decompose_setups = [
    (n, dense, permutation_method, check_finite, return_type)
    for n in (100,)
    for dense in (True, False)
    for permutation_method in supported_permutation_methods(dense)
    for check_finite in (True, False)
    for return_type in matrix.DECOMPOSITION_TYPES
]


@pytest.mark.parametrize('n, dense, permutation_method, check_finite, return_type', test_decompose_setups)
def test_decompose(n, dense, permutation_method, check_finite, return_type):
    A = random_square_matrix(n, dense=dense, positive_semi_definite=True)
    if dense:
        A_dense = A
    else:
        A_dense = A.todense()
    decomposition = matrix.decompose(A, permutation_method=permutation_method, check_finite=check_finite, return_type=return_type)
    A_composed = decomposition.composed_matrix
    if dense:
        A_composed_dense = A_composed
    else:
        A_composed_dense = A_composed.todense()
    np.testing.assert_array_almost_equal(A_dense, A_composed_dense)


# *** positive definite *** #

test_positive_definite_setups = [
    (n, dense)
    for n in (100,)
    for dense in (True, False)
]


@pytest.mark.parametrize('n, dense', test_positive_definite_setups)
def test_positive_definite(n, dense):
    A = random_square_matrix(n, dense=dense, positive_semi_definite=True)
    assert matrix.is_positive_semi_definite(A)
    assert not matrix.is_positive_semi_definite(-A)
    A = random_square_matrix(n, dense=dense, positive_definite=True)
    assert matrix.is_positive_semi_definite(A)
    assert not matrix.is_positive_semi_definite(-A)
    assert matrix.is_positive_definite(A)
    assert not matrix.is_positive_definite(-A)
