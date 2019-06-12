# *** submodules *** #

from matrix import constants, decompositions, errors

# *** functions *** #

from matrix.calculate import (is_positive_semidefinite, is_positive_definite, is_invertible,
                              decompose, solve)
from matrix.approximate import decomposition, positive_definite_matrix

# *** constants *** #

from matrix.constants import (
    DECOMPOSITION_TYPES,
    LDL_DECOMPOSITION_TYPE, LDL_DECOMPOSITION_COMPRESSED_TYPE, LL_DECOMPOSITION_TYPE,
    UNIVERSAL_PERMUTATION_METHODS, SPARSE_ONLY_PERMUTATION_METHODS, APPROXIMATION_ONLY_PERMUTATION_METHODS,
    NO_PERMUTATION_METHOD,
    DECREASING_DIAGONAL_VALUES_PERMUTATION_METHOD, INCREASING_DIAGONAL_VALUES_PERMUTATION_METHOD,
    DECREASING_ABSOLUTE_DIAGONAL_VALUES_PERMUTATION_METHOD,
    INCREASING_ABSOLUTE_DIAGONAL_VALUES_PERMUTATION_METHOD)

DECOMPOSITION_TYPES = DECOMPOSITION_TYPES
""" Supported types of decompositions. """
UNIVERSAL_PERMUTATION_METHODS = UNIVERSAL_PERMUTATION_METHODS
""" Supported permutation methods for decompose dense and sparse matrices. """
SPARSE_ONLY_PERMUTATION_METHODS = SPARSE_ONLY_PERMUTATION_METHODS
""" Supported permutation methods only for sparse matrices. """
APPROXIMATION_ONLY_PERMUTATION_METHODS = APPROXIMATION_ONLY_PERMUTATION_METHODS
""" Supported permutation methods only for approximate dense and sparse matrices. """


# *** version *** #

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions


# *** logging *** #

import logging
logger = logging.getLogger(__name__)
del logging
