import matrix.dense.util
import matrix.sparse.util

import matrix.errors


def equal(A, B):
    A_is_sparse = matrix.sparse.util.is_sparse(A)
    B_is_sparse = matrix.sparse.util.is_sparse(B)
    if A_is_sparse != B_is_sparse:
        return False
    if A_is_sparse:
        assert B_is_sparse
        return matrix.sparse.util.equal(A, B)
    else:
        assert not B_is_sparse
        return matrix.dense.util.equal(A, B)


def check_square_matrix(A):
    if len(A.shape) != 2 or A.shape[0] != A.shape[1]:
        raise matrix.errors.MatrixNotSquareError(matrix=A)
