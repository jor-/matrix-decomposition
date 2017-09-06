# *** submodules *** #

import matrix.constants
import matrix.decompositions
import matrix.errors


# *** functions *** #

from matrix.calculate import decompose, approximate, approximate_positive_definite, approximate_with_reduction_factor_file, is_positive_semi_definite, is_positive_definite, is_invertible, solve


# *** constants *** #

from matrix.constants import LDL_DECOMPOSITION_TYPE, LDL_DECOMPOSITION_COMPRESSED_TYPE, LL_DECOMPOSITION_TYPE

DECOMPOSITION_TYPES = matrix.constants.DECOMPOSITION_TYPES
""" Supported types of decompositions. """

from matrix.constants import DECREASING_DIAGONAL_VALUES_PERMUTATION_METHOD, INCREASING_DIAGONAL_VALUES_PERMUTATION_METHOD, DECREASING_ABSOLUTE_DIAGONAL_VALUES_PERMUTATION_METHOD, INCREASING_ABSOLUTE_DIAGONAL_VALUES_PERMUTATION_METHOD
PERMUTATION_METHODS = matrix.constants.PERMUTATION_METHODS
""" Supported permutation methods for dense and sparse matrices. """
SPARSE_PERMUTATION_METHODS = matrix.constants.SPARSE_PERMUTATION_METHODS
""" Supported permutation methods only for sparse matrices. """


# *** version *** #

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions


# *** logging *** #

import logging

logger = logging.getLogger(__name__)
