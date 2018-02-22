import numpy as np
import pytest

import matrix.constants
import matrix.errors
import matrix.tests.random


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
