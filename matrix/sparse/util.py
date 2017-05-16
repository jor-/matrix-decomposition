import scipy.sparse


def is_sparse(A):
    return scipy.sparse.issparse(A)
