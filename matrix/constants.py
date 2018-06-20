# *** decomposition types *** #

BASE_DECOMPOSITION_TYPE = 'base'
LDL_DECOMPOSITION_TYPE = 'LDL'
LDL_DECOMPOSITION_COMPRESSED_TYPE = 'LDL_compressed'
LL_DECOMPOSITION_TYPE = 'LL'

DECOMPOSITION_TYPES = (LDL_DECOMPOSITION_TYPE, LDL_DECOMPOSITION_COMPRESSED_TYPE, LL_DECOMPOSITION_TYPE)
""" Supported types of decompositions. """

# *** permutation methods *** #

NO_PERMUTATION_METHOD = 'none'
DECREASING_DIAGONAL_VALUES_PERMUTATION_METHOD = 'decreasing_diagonal_values'
INCREASING_DIAGONAL_VALUES_PERMUTATION_METHOD = 'increasing_diagonal_values'
DECREASING_ABSOLUTE_DIAGONAL_VALUES_PERMUTATION_METHOD = 'decreasing_absolute_diagonal_values'
INCREASING_ABSOLUTE_DIAGONAL_VALUES_PERMUTATION_METHOD = 'increasing_absolute_diagonal_values'

DIAGONAL_VALUES_PERMUTATION_METHODS = (
    DECREASING_DIAGONAL_VALUES_PERMUTATION_METHOD,
    INCREASING_DIAGONAL_VALUES_PERMUTATION_METHOD,
    DECREASING_ABSOLUTE_DIAGONAL_VALUES_PERMUTATION_METHOD,
    INCREASING_ABSOLUTE_DIAGONAL_VALUES_PERMUTATION_METHOD)
UNIVERSAL_PERMUTATION_METHODS = (NO_PERMUTATION_METHOD,) + DIAGONAL_VALUES_PERMUTATION_METHODS
""" Supported permutation methods for decompose dense and sparse matrices. """

MINIMAL_DIFFERENCE_PERMUTATION_METHOD = 'minimal_difference'
APPROXIMATION_ONLY_PERMUTATION_METHODS = (MINIMAL_DIFFERENCE_PERMUTATION_METHOD,)
""" Supported permutation methods only for approximate dense and sparse matrices. """

from matrix.sparse.constants import SPARSE_ONLY_PERMUTATION_METHODS
""" Supported permutation methods only for sparse matrices. """

from matrix.sparse.constants import BEST_FILL_REDUCE_PERMUTATION_METHOD
""" Supported permutation method to best reduce fill of decomposed sparse matrices. """
from matrix.sparse.constants import DEFAULT_FILL_REDUCE_PERMUTATION_METHOD
""" Supported permutation method to default reduce fill of decomposed sparse matrices. """

# *** save and load *** #

DECOMPOSITION_ATTRIBUTE_FILENAME = 'attribute_{attribute_name}.{file_extension}'
DECOMPOSITION_ATTRIBUTE_DENSE_FILE_EXTENSION = 'dense.npz'
DECOMPOSITION_ATTRIBUTE_SPARSE_FILE_EXTENSION = 'sparse.npz'
DECOMPOSITION_TYPE_FILENAME = 'type.txt'
DECOMPOSITION_FILENAME_EXTENSION = 'dec'
