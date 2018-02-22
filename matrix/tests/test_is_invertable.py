import numpy as np
import pytest

import matrix.constants
import matrix.errors
import matrix.tests.random


# *** is invertible *** #

test_is_invertible_setups = [
    (n, dense, complex_values, type_str, invertible)
    for n in (10,)
    for dense in (True, False)
    for complex_values in (True, False)
    for type_str in matrix.constants.DECOMPOSITION_TYPES
    for invertible in (True, False)
]


@pytest.mark.parametrize('n, dense, complex_values, type_str, invertible', test_is_invertible_setups)
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
