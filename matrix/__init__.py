# *** submodules *** #

from matrix import constants, decompositions, errors, approximation, nearest

# *** functions *** #

from matrix.calculate import (is_positive_semidefinite, is_positive_definite, is_invertible,
                              decompose, solve)

# *** constants *** #

from matrix.constants import (
    DECOMPOSITION_TYPES,
    LDL_DECOMPOSITION_TYPE, LDL_DECOMPOSITION_COMPRESSED_TYPE, LL_DECOMPOSITION_TYPE,
    UNIVERSAL_PERMUTATION_METHODS, SPARSE_ONLY_PERMUTATION_METHODS,
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


# *** version *** #

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions


# *** logging *** #

import logging
logger = logging.getLogger(__name__)
del logging


# *** deprecated *** #

def __getattr__(name):
    deprecated_names = ['decomposition', 'positive_definite_matrix',
                        'positive_semidefinite_matrix', 'APPROXIMATION_ONLY_PERMUTATION_METHODS']
    if name in deprecated_names:
        import warnings
        warnings.warn(f'"matrix.{name}" is deprecated. Take a look at'
                      ' "matrix.approximation.positive_semidefinite" instead.',
                      DeprecationWarning, stacklevel=2)
        import matrix.approximate
        return matrix.approximate.__getattribute__(name)
    raise AttributeError(f'Module {__name__} has no attribute {name}.')
