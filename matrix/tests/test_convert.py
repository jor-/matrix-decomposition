import pytest

import matrix.constants
import matrix.tests.random


# *** convert *** #

test_convert_setups = [
    (n, dense, complex_values, type_str, copy)
    for n in (10,)
    for dense in (True, False)
    for complex_values in (True, False)
    for type_str in matrix.constants.DECOMPOSITION_TYPES
    for copy in (True, False)
]


@pytest.mark.parametrize('n, dense, complex_values, type_str, copy', test_convert_setups)
def test_convert(n, dense, complex_values, type_str, copy):
    decomposition = matrix.tests.random.decomposition(n, type_str=type_str, dense=dense, complex_values=complex_values, positive_semidefinite=True, invertible=True)
    for convert_type_str in matrix.constants.DECOMPOSITION_TYPES:
        converted_decomposition = decomposition.as_type(convert_type_str, copy=copy)
        equal = type_str == convert_type_str
        equal_calculated = decomposition == converted_decomposition
        assert equal == equal_calculated
