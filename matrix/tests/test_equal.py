import warnings

import numpy as np
import scipy.sparse
import pytest

import matrix.constants
import matrix.tests.random


# *** equal *** #

test_equal_setups = [
    (n, dense, complex_values, type_str)
    for n in (10,)
    for dense in (True, False)
    for complex_values in (True, False)
    for type_str in matrix.constants.DECOMPOSITION_TYPES
]


@pytest.mark.parametrize('n, dense, complex_values, type_str', test_equal_setups)
def test_equal(n, dense, complex_values, type_str):
    decomposition = matrix.tests.random.decomposition(n, type_str=type_str, dense=dense, complex_values=complex_values)

    # compare with self
    assert decomposition == decomposition
    assert decomposition.is_equal(decomposition)
    assert decomposition.is_almost_equal(decomposition)

    # compare with other format
    for (n_other, dense_other, complex_values_other, type_str_other) in test_equal_setups:
        if n != n_other or dense != dense_other or complex_values != complex_values_other or type_str != type_str_other:
            decomposition_other = matrix.tests.random.decomposition(n_other, type_str=type_str_other, dense=dense_other, complex_values=complex_values_other)
            assert decomposition != decomposition_other
            assert not decomposition.is_equal(decomposition_other)
            assert not decomposition.is_almost_equal(decomposition_other)

    # compare with almost equal and not almost equal
    decomposition_almost_equal = decomposition.copy()
    decomposition_not_almost_equal = decomposition.copy()
    i = np.random.randint(n)
    j = np.random.randint(i + 1)
    eps = np.finfo(np.float64).eps * 10**2
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', scipy.sparse.SparseEfficiencyWarning)
        if decomposition.is_type(matrix.constants.LL_DECOMPOSITION_TYPE) or decomposition.is_type(matrix.constants.LDL_DECOMPOSITION_TYPE):
            decomposition_almost_equal.L[i, j] += eps
            decomposition_not_almost_equal.L[i, j] += 1
        else:
            assert decomposition.is_type(matrix.constants.LDL_DECOMPOSITION_COMPRESSED_TYPE)
            decomposition_almost_equal.LD[i, j] += eps
            decomposition_not_almost_equal.LD[i, j] += 1

    assert decomposition != decomposition_almost_equal
    assert not decomposition.is_equal(decomposition_almost_equal)
    assert decomposition.is_almost_equal(decomposition_almost_equal)
    assert decomposition != decomposition_almost_equal
    assert not decomposition.is_equal(decomposition_not_almost_equal)
    assert not decomposition.is_almost_equal(decomposition_not_almost_equal)
