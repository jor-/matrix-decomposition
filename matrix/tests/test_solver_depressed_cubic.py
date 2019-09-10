import cmath

import numpy as np
import pytest

import matrix._util.roots


# *** solver_depressed_cubic *** #

test_solver_depressed_cubic_setups = [
    (1, 1),   # D > 0
    (-3, 1),  # D < 0
    (0, 0),   # D == 0 and p == 0
    (-3, 2),  # D == 0 and p != 0
]
test_solver_depressed_cubic_setups.extend((p, q) for (p, q) in np.random.uniform(low=-10, high=10, size=(6, 2)))


@pytest.mark.parametrize('p, q', test_solver_depressed_cubic_setups)
def test_solver_depressed_cubic(p, q):
    roots = matrix._util.roots.solver_depressed_cubic(p, q, include_complex_values=True)
    assert 1 <= len(roots) <= 3
    assert all(cmath.isclose(x**3 + p * x + q, 0, abs_tol=1e-08) for x in roots)
    roots = matrix._util.roots.solver_depressed_cubic(p, q, include_complex_values=False)
    assert 1 <= len(roots) <= 3
    assert all(x.imag == 0 for x in roots)
