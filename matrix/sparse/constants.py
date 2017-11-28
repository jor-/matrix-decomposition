from matrix.constants import PERMUTATION_METHODS, DECOMPOSITION_TYPES

CHOLMOD_NO_PERMUTATION_METHOD = 'natural'
try:
    import sksparse.cholmod
except ImportError:
    CHOLMOD_PERMUTATION_METHODS = ()
else:
    CHOLMOD_PERMUTATION_METHODS = tuple(sksparse.cholmod._ordering_methods.keys())

FILL_REDUCE_PERMUTATION_METHOD_PREFIX = 'fill_reduce_'
FILL_REDUCE_PERMUTATION_METHODS = tuple(FILL_REDUCE_PERMUTATION_METHOD_PREFIX + method for method in sksparse.cholmod._ordering_methods.keys() if method != CHOLMOD_NO_PERMUTATION_METHOD) + (CHOLMOD_NO_PERMUTATION_METHOD,)
""" Supported permutation methods only for sparse matrices. """

PERMUTATION_METHODS = PERMUTATION_METHODS + FILL_REDUCE_PERMUTATION_METHODS
