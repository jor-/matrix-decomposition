import warnings

import numpy as np
import scipy.sparse

import matrix
import matrix.calculate
import matrix.constants
import matrix.decompositions
import matrix.permute


RANDOM_STATE = 1234


# *** random values *** #

def universal_matrix(n, m, dense=True, complex_values=False):
    np.random.seed(RANDOM_STATE)
    if dense:
        # generate random matrix
        A = np.random.rand(n, m)
        # apply complex values
        if complex_values:
            A = A + np.random.rand(n, m) * 1j
    else:
        # generate random matrix
        density = 0.1
        A = scipy.sparse.rand(n, m, density=density, random_state=RANDOM_STATE)
        # apply complex values
        if complex_values:
            B = A.tocsr(copy=True)
            B.data = np.random.rand(len(B.data)) * 1j
            A = A + B
    return A


def lower_triangle_matrix(n, dense=True, complex_values=False, real_values_diagonal=False, finite=True, positive_semidefinite=False, invertible=False):
    # create random triangle matrix
    A = universal_matrix(n, n, dense=dense, complex_values=complex_values)
    if dense:
        A = np.tril(A)
    else:
        A = scipy.sparse.tril(A).tocsc()

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', scipy.sparse.SparseEfficiencyWarning)

        # apply real_values_diagonal and positive_semidefinite
        if real_values_diagonal or positive_semidefinite:
            for i in range(n):
                A_ii = A[i, i]
                if A_ii != 0:
                    A[i, i] = A_ii.real
        else:
            for i in range(n):
                A_ii = A[i, i]
                if A_ii != 0:
                    A[i, i] = A_ii * np.sign(np.random.random() - 0.5)

        # apply invertible
        if invertible:
            A = A + scipy.sparse.eye(n)
        else:
            i = np.random.randint(n)
            A[i, i] = 0

        # apply finite
        if not finite:
            i = np.random.randint(n)
            j = np.random.randint(i + 1)
            A[i, j] = np.nan
            A[i, i] = np.nan

    return A


def hermitian_matrix(n, dense=True, complex_values=False, positive_semidefinite=False, invertible=False, min_diag_value=None):
    # generate hermitian and maybe positive (semi-)definite matrix
    A = universal_matrix(n, n, dense=dense, complex_values=complex_values)
    if positive_semidefinite or invertible:
        d = np.random.rand(n)
        if invertible:
            d = d + 1
        else:
            d[np.random.randint(n)] = 0
        if dense:
            D = np.diag(d)
            L = np.tril(A, -1)
        else:
            D = scipy.sparse.diags(d)
            L = scipy.sparse.tril(A, -1).tocsr()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', scipy.sparse.SparseEfficiencyWarning)
            for i in range(n):
                L[i, i] = 1
        A = L @ D @ L.transpose().conj()
    else:
        A = A + A.transpose().conj()

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', scipy.sparse.SparseEfficiencyWarning)
        # set real diagonal values
        if complex_values:
            for i in range(n):
                A[i, i] = A[i, i].real

        # set min diag value
        if min_diag_value is not None:
            for i in range(n):
                A[i, i] = max(min_diag_value, A[i, i])

    # return
    assert np.all(np.isreal(A.diagonal()))
    return A


def vector(n, complex_values=False):
    np.random.seed(RANDOM_STATE)
    v = np.random.rand(n)
    if complex_values:
        v = v + np.random.rand(n) * 1j
    return v


def permutation_vector(n):
    np.random.seed(RANDOM_STATE)
    return np.random.permutation(n)


def decomposition(n, type_str='LL', dense=True, complex_values=False, finite=True, positive_semidefinite=False, invertible=False):
    # make random parts of decomposition
    L = lower_triangle_matrix(n, dense=dense, complex_values=complex_values, real_values_diagonal=True, finite=finite, positive_semidefinite=positive_semidefinite, invertible=invertible)
    p = permutation_vector(n)
    # make decomposition of correct type
    if type_str == 'LL':
        decomposition = matrix.decompositions.LL_Decomposition(L, p)
    elif type_str == 'LDL_compressed':
        decomposition = matrix.decompositions.LDL_DecompositionCompressed(L, p)
    elif type_str == 'LDL':
        d = np.asarray(L.diagonal().real).flatten() * (np.random.rand(n) + 1)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', scipy.sparse.SparseEfficiencyWarning)
            for i in range(n):
                L[i, i] = 1
        decomposition = matrix.decompositions.LDL_Decomposition(L, d, p)
    else:
        raise ValueError('Unknown decomposition type {}.'.format(type_str))
    return decomposition
