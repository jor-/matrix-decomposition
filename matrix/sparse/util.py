import scipy.sparse


def is_sparse(A):
    return scipy.sparse.issparse(A)


def equal(A, B):
    return A.shape == B.shape and (A != B).nnz == 0
