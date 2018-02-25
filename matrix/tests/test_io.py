import os.path
import tempfile

import numpy as np
import pytest

import matrix.constants
import matrix.decompositions
import matrix.errors
import matrix.tests.random


# *** save and load *** #

test_save_and_load_setups = [
    (n, dense, complex_values, type_str)
    for n in (10,)
    for dense in (True, False)
    for complex_values in (True, False)
    for type_str in matrix.constants.DECOMPOSITION_TYPES
]


@pytest.mark.parametrize('n, dense, complex_values, type_str', test_save_and_load_setups)
def test_save_and_load(n, dense, complex_values, type_str):
    # create decompositions
    decomposition = matrix.tests.random.decomposition(n, type_str=type_str, dense=dense, complex_values=complex_values)
    decomposition_other_same_type = type(decomposition)()
    other_type_str = tuple(other_type_str for other_type_str in matrix.constants.DECOMPOSITION_TYPES if other_type_str != type_str)[0]
    decomposition_other_different_type = matrix.tests.random.decomposition(n, type_str=other_type_str, dense=dense, complex_values=complex_values)
    # test decomposition IO methods
    with tempfile.TemporaryDirectory() as tmp_dir:
        file = os.path.join(tmp_dir, 'decomposition')
        decomposition.save(file)
        decomposition_other_same_type.load(file)
        with np.testing.assert_raises(matrix.errors.DecompositionInvalidDecompositionTypeFile):
            decomposition_other_different_type.load(file)
    assert decomposition == decomposition_other_same_type
    assert decomposition != decomposition_other_different_type
    # test general IO methods
    with tempfile.TemporaryDirectory() as tmp_dir:
        file = os.path.join(tmp_dir, 'decomposition')
        matrix.decompositions.save(file, decomposition)
        decomposition_other_same_type = matrix.decompositions.load(file)
    assert decomposition == decomposition_other_same_type
