"""
This module is deprecated and will be removed in a future release.
Please use matrix.approximation.positive_semidefinite instead.
"""

import warnings

warnings.warn("This module is deprecated and will be removed in a future release. Please use matrix.approximation.positive_semidefinite instead.", DeprecationWarning, stacklevel=2)

from matrix.approximation.positive_semidefinite import decomposition, APPROXIMATION_ONLY_PERMUTATION_METHODS
from matrix.approximation.positive_semidefinite import positive_semidefinite_matrix as positive_definite_matrix


def positive_semidefinite_matrix(*args, **kargs):
    return positive_definite_matrix(*args, min_diag_D=0, **kargs)
