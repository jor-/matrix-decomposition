import abc

import numpy as np
import scipy.sparse

import matrix.permute


class DecompositionBase(metaclass=abc.ABCMeta):
    """ A matrix decomposition.

    This class is a base class for matrix decompositions.
    """

    def __init__(self, p=None):
        """
        Parameters
        ----------
        p : numpy.ndarray
            The permutation vector used for the decomposition.
            This decomposition is of A[p[:, np.newaxis], p[np.newaxis, :]] where A is a matrix.
            optional, default: no permutation
        """

        if p is not None:
            self._p = p

    # *** permutation *** #

    @property
    def is_permuted(self):
        """ :class:`bool`: Whether this is a decompositon with permutation."""

        try:
            p = self._p
        except AttributeError:
            return False
        else:
            return np.any(p != np.arange(len(p)))

    @property
    def p(self):
        """ :class:`numpy.ndarray`: The permutation vector.
        A[p[:, np.newaxis], p[np.newaxis, :]] is the matrix A permuted by the permutation of the decomposition"""

        try:
            return self._p
        except AttributeError:
            return np.arange(self.n)

    @property
    def p_inverse(self):
        """ :class:`numpy.ndarray`: The permutation vector that undos the permutation."""

        return matrix.permute.invert_permutation_vector(self.p)

    def _apply_previous_permutation(self, p_first):
        """ Applies a previous permutation to the current permutation.

        Parameters
        ----------
        p_first : numpy.ndarray
            The previous permutation vector.
        """

        if p_first is not None:
            try:
                p_after = self._p
            except AttributeError:
                self._p = p_first
            else:
                self._p = p_after[p_first]

    def _apply_succeeding_permutation(self, p_after):
        """ Applies a succeeding permutation to the current permutation.

        Parameters
        ----------
        p_after : numpy.ndarray
            The succeeding permutation vector.
        """

        if p_after is not None:
            try:
                p_first = self._p
            except AttributeError:
                self._p = p_after
            else:
                self._p = p_after[p_first]

    @property
    def P(self):
        """ :class:`scipy.sparse.dok_matrix`: The permutation matrix.
        P @ A @ P.H is the matrix A permuted by the permutation of the decomposition"""

        p = self.p
        n = len(p)
        P = scipy.sparse.dok_matrix((n, n), dtype=np.int8)
        for i in range(n):
            P[i, p[i]] = 1
        return P

    def permute_matrix(self, A):
        """ Permute a matrix by the permutation of the decomposition.

        Parameters
        ----------
        A : numpy.ndarray or scipy.sparse.spmatrix
            The matrix that should be permuted.

        Returns
        -------
        numpy.ndarray or scipy.sparse.spmatrix
            The matrix `A` permuted by the permutation of the decomposition.
        """

        if self.is_permuted:
            return matrix.permute.symmetric(A, self.p)
        else:
            return A

    def unpermute_matrix(self, A):
        """ Unpermute a matrix permuted by the permutation of the decomposition.

        Parameters
        ----------
        A : numpy.ndarray or scipy.sparse.spmatrix
            The matrix that should be unpermuted.

        Returns
        -------
        numpy.ndarray or scipy.sparse.spmatrix
            The matrix `A` unpermuted by the permutation of the decomposition.
        """

        if self.is_permuted:
            return matrix.permute.symmetric(A, self.p_inverse)
        else:
            return A

    # *** basic properties *** #

    @property
    @abc.abstractmethod
    def n(self):
        """:class:`int`: The dimension of the squared decomposed matrix."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def is_sparse(self):
        """:class:`bool`: Whether this is a sparse decompositon."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def composed_matrix(self):
        """:class:`numpy.matrix` or :class:`scipy.sparse.spmatrix`: The composed matrix represented by this decomposition."""
        raise NotImplementedError


class LDL_Decomposition(DecompositionBase):
    """ A matrix decomposition where :math:`LDL^H` is the decomposed (permuted) matrix.

    `L` is a lower triangle matrix with ones on the diagonal. `D` is a diagonal matrix.
    Only the diagonal values of `D` are stored.
    """

    def __init__(self, L, d, p=None):
        """
        Parameters
        ----------
        L : numpy.ndarray or scipy.sparse.spmatrix
            The matrix `L` of the decomposition.
        d : numpy.ndarray
            The vector of the diagonal components of `D` of the decompositon.
        p : numpy.ndarray
            The permutation vector used for the decomposition.
            This decomposition is of A[p[:, np.newaxis], p[np.newaxis, :]] where A is a matrix.
            optional, default: no permutation
        """

        self.L = L
        self.d = d
        super().__init__(p=p)

    # *** base properties *** #

    @property
    def n(self):
        return self.L.shape[0]

    @property
    def is_sparse(self):
        return scipy.sparse.issparse(self.L)

    @property
    def composed_matrix(self):
        A = self.L @ self.D @ self.L.H
        A = self.unpermute_matrix(A)
        return A

    # *** decomposition specific properties *** #

    @property
    def L(self):
        """:class:`numpy.matrix` or :class:`scipy.sparse.spmatrix`: The matrix `L` of the decomposition."""
        return self._L

    @L.setter
    def L(self, L):
        self._L = L
        if not self.is_sparse:
            L = np.asmatrix(L)
        self._L = L

    @property
    def d(self):
        """:class:`numpy.ndarray`: The diagonal vector of the matrix `D` of the decomposition."""
        return self._d

    @d.setter
    def d(self, d):
        self._d = np.asarray(d)

    @property
    def D(self):
        """ :class:`scipy.sparse.dia_matrix`: The permutation matrix."""
        return scipy.sparse.diags(self.d)

    @property
    def LD(self):
        """:class:`numpy.matrix` or :class:`scipy.sparse.spmatrix`: A matrix whose diagonal values are the diagonal values of `D` and whose off-diagonal values are those of `L`."""
        LD = self.L.copy()
        d = self.d
        for i in range(self.n):
            LD[i, i] = d[i]
        return LD


class LDL_DecompositionCompressed(DecompositionBase):
    """ A matrix decomposition where :math:`LDL^H` is the decomposed (permuted) matrix.

    `L` is a lower triangle matrix with ones on the diagonal. `D` is a diagonal matrix.
    `L` and `D` are stored in one matrix whose diagonal values are the diagonal values of `D`
    and whose off-diagonal values are those of `L`.
    """

    def __init__(self, LD, p=None):
        """
        Parameters
        ----------
        LD : numpy.ndarray or scipy.sparse.spmatrix
            A matrix whose diagonal values are the diagonal values of `D` and whose off-diagonal values are those of `L`.
        p : numpy.ndarray
            The permutation vector used for the decomposition.
            This decomposition is of A[p[:, np.newaxis], p[np.newaxis, :]] where A is a matrix.
            optional, default: no permutation
        """
        self.LD = LD
        super().__init__(p=p)

    # *** base properties *** #

    @property
    def n(self):
        return self.LD.shape[0]

    @property
    def is_sparse(self):
        return scipy.sparse.issparse(self.LD)

    @property
    def composed_matrix(self):
        return self.to_LDL_Decomposition().composed_matrix

    # *** decomposition specific properties *** #

    @property
    def LD(self):
        """:class:`numpy.matrix` or :class:`scipy.sparse.spmatrix`: A matrix whose diagonal values are the diagonal values of `D` and whose off-diagonal values are those of `L`."""
        return self._LD

    @LD.setter
    def LD(self, LD):
        self._LD = LD
        if not self.is_sparse:
            LD = np.asmatrix(LD)
        self._LD = LD

    @property
    def d(self):
        """:class:`numpy.ndarray`: The diagonal vector of the matrix `D` of the decomposition."""
        d = self.LD.diagonal()
        if not self.is_sparse:
            d = d.A1
        return d

    @property
    def D(self):
        """ :class:`scipy.sparse.dia_matrix`: The permutation matrix."""
        return scipy.sparse.diags(self.d)

    @property
    def L(self):
        """:class:`numpy.matrix` or :class:`scipy.sparse.spmatrix`: The matrix `L` of the decomposition."""

        L = self.LD.copy()
        for i in range(self.n):
            L[i, i] = 1
        return L


class LL_Decomposition(DecompositionBase):
    """ A matrix decomposition where :math:`LL^H` is the decomposed (permuted) matrix.

    `L` is a lower triangle matrix with ones on the diagonal.
    This decomposition is also called Cholesky decomposition.
    """

    def __init__(self, L, p=None):
        """
        Parameters
        ----------
        L : numpy.ndarray or scipy.sparse.spmatrix
            The matrix `L` of the decomposition.
        p : numpy.ndarray
            The permutation vector used for the decomposition.
            This decomposition is of A[p[:, np.newaxis], p[np.newaxis, :]] where A is a matrix.
            optional, default: no permutation
        """
        self.L = L
        super().__init__(p=p)

    # *** base properties *** #

    @property
    def n(self):
        return self.L.shape[0]

    @property
    def is_sparse(self):
        return scipy.sparse.issparse(self.L)

    @property
    def composed_matrix(self):
        A = self.L @ self.L.H
        A = self.unpermute_matrix(A)
        return A

    # *** decomposition specific properties *** #

    @property
    def L(self):
        """:class:`numpy.matrix` or :class:`scipy.sparse.spmatrix`: The matrix `L` of the decomposition."""
        return self._L

    @L.setter
    def L(self, L):
        self._L = L
        if not self.is_sparse:
            L = np.asmatrix(L)
        self._L = L
