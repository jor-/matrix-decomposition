import sksparse.cholmod

from matrix.constants import PERMUTATION_METHODS, DECOMPOSITION_TYPES

CHOLMOD_PERMUTATION_METHODS = tuple(sksparse.cholmod._ordering_methods.keys())
SPARSE_PERMUTATION_METHODS = CHOLMOD_PERMUTATION_METHODS
""" Supported permutation methods only for sparse matrices. """

PERMUTATION_METHODS = PERMUTATION_METHODS + SPARSE_PERMUTATION_METHODS
