import abc
import copy
import warnings

import numpy as np
import scipy.sparse

import matrix.constants
import matrix.errors
import matrix.permute
import matrix.util


class DecompositionBase(metaclass=abc.ABCMeta):
    """ A matrix decomposition.

    This class is a base class for matrix decompositions.
    """

    _decomposition_type = matrix.constants.BASE_DECOMPOSITION_TYPE
    """ :class:`str`: The type of this decomposition represented as string. """

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

    @property
    def decomposition_type(self):
        """:class:`str`: The type of this decompositon."""
        return self._decomposition_type

    # *** compare methods *** #

    def __eq__(self, other):
        if not isinstance(other, DecompositionBase):
            return False
        if not self.decomposition_type == other.decomposition_type:
            return False
        if self.is_permuted and other.is_permuted:
            return np.all(self.p == other.p)
        else:
            return not self.is_permuted and not other.is_permuted

    # *** convert type *** #

    def copy(self):
        """ Copy this decomposition.

        Returns
        -------
        matrix.decompositions.DecompositionBase
            A copy of this decomposition.
        """
        return copy.deepcopy(self)

    def is_type(self, decomposition_type):
        """ Whether this is a decomposition of the passed type.

        Parameters
        ----------
        decomposition_type : str
            The decomposition type according to which is checked.

        Returns
        -------
        bool
            Whether this is a decomposition of the passed type.
        """

        if decomposition_type is None:
            return True
        else:
            try:
                return self._decomposition_type == decomposition_type
            except AttributeError:
                False

    def to(self, decomposition_type, copy=False):
        """ Convert decomposition to passed type.

        Parameters
        ----------
        decomposition_type : str
            The decomposition type to which this decomposition is converted.
        copy : bool
            Whether the data of this decomposition should always be copied or only if needed.

        Returns
        -------
        matrix.decompositions.DecompositionBase
            If the type of this decomposition is not `decomposition_type`, a decompostion of type
            `decomposition_type` is returned which represents the same decomposed matrix as this
            decomposition. Otherwise this decomposition or a copy of it is returned, depending on
            `copy`.
        """

        if self.is_type(decomposition_type):
            if copy:
                return self.copy()
            else:
                return self
        else:
            raise matrix.errors.MatrixDecompositionNoConversionImplementedError(
                original_decomposition=self, desired_decomposition_type=decomposition_type)

    def to_any(self, *decomposition_types, copy=False):
        """ Convert decomposition to any of the passed types.

        Parameters
        ----------
        *decomposition_types : str
            The decomposition types to any of them this this decomposition is converted.
        copy : bool
            Whether the data of this decomposition should always be copied or only if needed.

        Returns
        -------
        matrix.decompositions.DecompositionBase
            If the type of this decomposition is not in `decomposition_types`, a decompostion of
            type `decomposition_type[0]` is returned which represents the same decomposed matrix
            as this decomposition. Otherwise this decomposition or a copy of it is returned,
            depending on `copy`.
        """

        if len(decomposition_types) == 0 or any(map(self.is_type, decomposition_types)):
            if copy:
                return self.copy()
            else:
                return self
        else:
            return self.to(decomposition_types[0])

    # *** positive definite checks *** #

    @property
    @abc.abstractmethod
    def is_positive_semi_definite(self):
        """:class:`bool`: Whether the matrix represented by this decomposition is positive semi-definite."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def is_positive_definite(self):
        """:class:`bool`: Whether the matrix represented by this decomposition is positive definite."""
        raise NotImplementedError


class LDL_Decomposition(DecompositionBase):
    """ A matrix decomposition where :math:`LDL^H` is the decomposed (permuted) matrix.

    `L` is a lower triangle matrix with ones on the diagonal. `D` is a diagonal matrix.
    Only the diagonal values of `D` are stored.
    """

    _decomposition_type = matrix.constants.LDL_DECOMPOSITION_TYPE
    """ :class:`str`: The type of this decomposition represented as string. """

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

    # *** compare methods *** #

    def __eq__(self, other):
        if not super().__eq__(other):
            return False
        return np.all(self.d == other.d) and matrix.util.equal(self.L, other.L)

    # *** convert type *** #

    def to_LL_Decomposition(self):
        L = self.L
        d = self.d
        p = self.p

        # check d for negative entries
        d_negative_mask = d < 0
        if np.any(d_negative_mask):
            i = np.where(d_negative_mask)[0][0]
            p_i = p[i]
            raise matrix.errors.MatrixNoLLDecompositionPossibleError(
                problematic_leading_principal_submatrix_index=p_i)
        del d_negative_mask

        # compute new d
        d = np.sqrt(d)

        # compute new L
        D = scipy.sparse.diags(d)
        L = L @ D

        # construct new decompostion
        return LL_Decomposition(L, p=p)

    def to_LDL_DecompositionCompressed(self):
        return LDL_DecompositionCompressed(self.LD, p=self.p)

    def to(self, decomposition_type, copy=False):
        try:
            return super().to(decomposition_type, copy=copy)
        except matrix.errors.MatrixDecompositionNoConversionImplementedError:
            if decomposition_type == matrix.constants.LL_DECOMPOSITION_TYPE:
                return self.to_LL_Decomposition()
            elif decomposition_type == matrix.constants.LDL_DECOMPOSITION_COMPRESSED_TYPE:
                return self.to_LDL_DecompositionCompressed()
            else:
                raise

    # *** positive definite checks *** #

    def is_positive_semi_definite(self):
        return np.all(self.d >= 0)

    def is_positive_definite(self):
        eps = np.finfo(self.d.dtype).resolution
        return np.all(self.d > eps)


class LDL_DecompositionCompressed(DecompositionBase):
    """ A matrix decomposition where :math:`LDL^H` is the decomposed (permuted) matrix.

    `L` is a lower triangle matrix with ones on the diagonal. `D` is a diagonal matrix.
    `L` and `D` are stored in one matrix whose diagonal values are the diagonal values of `D`
    and whose off-diagonal values are those of `L`.
    """

    _decomposition_type = matrix.constants.LDL_DECOMPOSITION_COMPRESSED_TYPE
    """ :class:`str`: The type of this decomposition represented as string. """

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
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', scipy.sparse.SparseEfficiencyWarning)
            for i in range(self.n):
                L[i, i] = 1
        return L

    # *** compare methods *** #

    def __eq__(self, other):
        if not super().__eq__(other):
            return False
        return matrix.util.equal(self.LD, other.LD)

    # *** convert type *** #

    def to_LDL_Decomposition(self):
        return LDL_Decomposition(self.L, self.d, p=self.p)

    def to(self, decomposition_type, copy=False):
        try:
            return super().to(decomposition_type, copy=copy)
        except matrix.errors.MatrixDecompositionNoConversionImplementedError:
            if decomposition_type == matrix.constants.LDL_DECOMPOSITION_TYPE:
                return self.to_LDL_Decomposition()
            elif decomposition_type == matrix.constants.LL_DECOMPOSITION_TYPE:
                return self.to_LDL_Decomposition().to_LL_Decomposition()
            else:
                raise

    # *** positive definite checks *** #

    def is_positive_semi_definite(self):
        return np.all(self.d >= 0)

    def is_positive_definite(self):
        d = self.d
        eps = np.finfo(self.d.dtype).resolution
        return np.all(d > eps)


class LL_Decomposition(DecompositionBase):
    """ A matrix decomposition where :math:`LL^H` is the decomposed (permuted) matrix.

    `L` is a lower triangle matrix with ones on the diagonal.
    This decomposition is also called Cholesky decomposition.
    """

    _decomposition_type = matrix.constants.LL_DECOMPOSITION_TYPE
    """ :class:`str`: The type of this decomposition represented as string. """

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

    # *** compare methods *** #

    def __eq__(self, other):
        if not super().__eq__(other):
            return False
        return matrix.util.equal(self.L, other.L)

    # *** convert type *** #

    @property
    def _d(self):
        """:class:`numpy.ndarray`: The diagonal vector of `L`."""
        d = self.L.diagonal()
        if not self.is_sparse:
            d = d.A1
        return d

    def to_LDL_Decomposition(self):
        L = self.L
        p = self.p

        # d inverse
        d = self._d
        d_zero_mask = d == 0
        d_inverse = np.empty(d.shape)
        d_inverse[d_zero_mask] = 0
        d_inverse[~d_zero_mask] = 1 / d[~d_zero_mask]
        assert np.all(np.isfinite(d_inverse[np.isfinite(d)]))

        # check entries where diagonal is zero
        n = self.n
        if np.any(d_zero_mask):
            for i in np.where(d_zero_mask)[0]:
                for j in range(i + 1, n):
                    if not np.isclose(L[j, i], 0):
                        p_i = i[i]
                        raise matrix.errors.MatrixNoLLDecompositionPossibleError(
                            problematic_leading_principal_submatrix_index=p_i)

        # compute new L
        D_inverse = scipy.sparse.diags(d_inverse)
        L = L @ D_inverse

        # set all diagonal elements to one (due to rounding errors)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', scipy.sparse.SparseEfficiencyWarning)
            for i in range(n):
                assert np.isclose(L[i, i], 1) or d_zero_mask[i] or not np.isfinite(L[i, i])
                L[i, i] = 1

        # compute new d
        d = d**2

        # construct new decompostion
        return LDL_Decomposition(L, d, p=p)

    def to(self, decomposition_type, copy=False):
        try:
            return super().to(decomposition_type, copy=copy)
        except matrix.errors.MatrixDecompositionNoConversionImplementedError:
            if decomposition_type == matrix.constants.LDL_DECOMPOSITION_TYPE:
                return self.to_LDL_Decomposition()
            elif decomposition_type == matrix.constants.LDL_DECOMPOSITION_COMPRESSED_TYPE:
                return self.to_LDL_Decomposition().to_LDL_DecompositionCompressed()
            else:
                raise

    # *** positive definite checks *** #

    def is_positive_semi_definite(self):
        return True

    def is_positive_definite(self):
        d = self._d
        eps = np.finfo(d.dtype).resolution
        return np.all(d > eps)
