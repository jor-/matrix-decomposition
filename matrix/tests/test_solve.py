import numpy as np
import pytest

import matrix.constants
import matrix.errors
import matrix.tests.random


# *** solve *** #

test_solve_setups = [
    (n, dense, complex_values, type_str, invertible, b)
    for n in (10,)
    for dense in (True, False)
    for complex_values in (True, False)
    for type_str in matrix.constants.DECOMPOSITION_TYPES
    for invertible in (True, False)
    for b in (matrix.tests.random.vector(n), np.zeros(n), np.arange(n), matrix.tests.random.universal_matrix(n, 2))
]


@pytest.mark.parametrize('n, dense, complex_values, type_str, invertible, b', test_solve_setups)
def test_solve(n, dense, complex_values, type_str, invertible, b):
    # make random decomposition
    decomposition = matrix.tests.random.decomposition(n, type_str=type_str, dense=dense, complex_values=complex_values, finite=True, invertible=invertible)
    if invertible:
        # calculate solution
        x = decomposition.solve(b)
        # verify solution
        A = decomposition.composed_matrix
        y = A @ x
        assert np.allclose(b, y)
    else:
        with np.testing.assert_raises(matrix.errors.DecompositionSingularError):
            decomposition.solve(b)
