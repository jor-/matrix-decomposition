import abc
import copy
import os
import tarfile
import tempfile
import warnings

import numpy as np
import scipy.sparse

import matrix
import matrix.constants
import matrix.errors
import matrix.permute
import matrix.util


class DecompositionBase(metaclass=abc.ABCMeta):
    """ A matrix decomposition.

    This class is a base class for matrix decompositions.
    """

    type_str = matrix.constants.BASE_DECOMPOSITION_TYPE
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

        self.p = p

    # *** str *** #

    def __str__(self):
        return '{type_str} decomposition of matrix with shape ({n}, {n})'.format(
            type_str=self.type_str, n=self.n)

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

    @p.setter
    def p(self, p):
        if p is not None:
            p = matrix.util.as_vector(p)
            self._p = p
        else:
            del self.p

    @p.deleter
    def p(self):
        try:
            del self._p
        except AttributeError:
            pass

    @property
    def p_inverse(self):
        """ :class:`numpy.ndarray`: The permutation vector that undos the permutation."""

        return matrix.permute.invert_permutation_vector(self.p)

    def _apply_previous_permutation(self, p_previous):
        """ Applies a previous permutation to the current permutation.

        Parameters
        ----------
        p_previous : numpy.ndarray
            The previous permutation vector.
        """

        try:
            p_next = self._p
        except AttributeError:
            p_next = None
        self._p = matrix.permute.concatenate_permutation_vectors(p_previous, p_next)

    def _apply_succeeding_permutation(self, p_next):
        """ Applies a succeeding permutation to the current permutation.

        Parameters
        ----------
        p_next : numpy.ndarray
            The succeeding permutation vector.
        """

        try:
            p_previous = self._p
        except AttributeError:
            p_previous = None
        self._p = matrix.permute.concatenate_permutation_vectors(p_previous, p_next)

    @property
    def P(self):
        """ :class:`scipy.sparse.dok_matrix`: The permutation matrix.
        P @ A @ P.T is the matrix A permuted by the permutation of the decomposition"""

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
    def composed_matrix(self):
        """:class:`numpy.matrix` or :class:`scipy.sparse.spmatrix`: The composed matrix represented by this decomposition."""
        raise NotImplementedError

    # *** compare methods *** #
    def _is_same_type_and_permutation(self, other):
        return (isinstance(other, DecompositionBase) and
                self.type_str == other.type_str and
                self.is_permuted == other.is_permuted and
                (not self.is_permuted or
                 (self.n == other.n and np.all(self.p == other.p))))

    def is_equal(self, other):
        """ Whether this decomposition is equal to passed decomposition.

        Parameters
        ----------
        other : str
            The decomposition which to compare to this decomposition.

        Returns
        -------
        bool
            Whether this decomposition is equal to passed decomposition.
        """
        return self._is_same_type_and_permutation(other)

    def __eq__(self, other):
        return self.is_equal(other)

    def is_almost_equal(self, other, rtol=1e-04, atol=1e-06):
        """ Whether this decomposition is close to passed decomposition.

        Parameters
        ----------
        other : str
            The decomposition which to compare to this decomposition.
        rtol : float
            The relative tolerance parameter.
        atol : float
            The absolute tolerance parameter.

        Returns
        -------
        bool
            Whether this decomposition is close to passed decomposition.
        """
        return self._is_same_type_and_permutation(other)

    # *** convert type *** #

    def copy(self):
        """ Copy this decomposition.

        Returns
        -------
        matrix.decompositions.DecompositionBase
            A copy of this decomposition.
        """

        # debug info
        matrix.logging.debug('Copying {}.'.format(self))
        # make copy
        return copy.deepcopy(self)

    def is_type(self, type_str):
        """ Whether this is a decomposition of the passed type.

        Parameters
        ----------
        type_str : str
            The decomposition type according to which is checked.

        Returns
        -------
        bool
            Whether this is a decomposition of the passed type.
        """

        if type_str is None:
            return True
        else:
            try:
                return self.type_str == type_str
            except AttributeError:
                return False

    def as_type(self, type_str, copy=False):
        """ Convert decomposition to passed type.

        Parameters
        ----------
        type_str : str
            The decomposition type to which this decomposition is converted.
        copy : bool
            Whether the data of this decomposition should always be copied or only if needed.

        Returns
        -------
        matrix.decompositions.DecompositionBase
            If the type of this decomposition is not `type_str`, a decomposition of type
            `type_str` is returned which represents the same decomposed matrix as this
            decomposition. Otherwise this decomposition or a copy of it is returned, depending on
            `copy`.
        """

        # debug info
        matrix.logging.debug('Converting {} to type {type_str} with copy={copy}.'.format(self, type_str=type_str, copy=copy))

        # convert
        if self.is_type(type_str):
            if copy:
                return self.copy()
            else:
                return self
        else:
            raise matrix.errors.NoDecompositionConversionImplementedError(self, type_str)

    def as_any_type(self, *type_strs, copy=False):
        """ Convert decomposition to any of the passed types.

        Parameters
        ----------
        *type_strs : str
            The decomposition types to any of them this this decomposition is converted.
        copy : bool
            Whether the data of this decomposition should always be copied or only if needed.

        Returns
        -------
        matrix.decompositions.DecompositionBase
            If the type of this decomposition is not in `type_strs`, a decomposition of
            type `type_str[0]` is returned which represents the same decomposed matrix
            as this decomposition. Otherwise this decomposition or a copy of it is returned,
            depending on `copy`.
        """

        # debug info
        matrix.logging.debug('Converting {} to any type of {type_strs} with copy={copy}.'.format(self, type_strs=type_strs, copy=copy))

        # convert
        if len(type_strs) == 0 or any(map(self.is_type, type_strs)):
            if copy:
                return self.copy()
            else:
                return self
        else:
            return self.as_type(type_strs[0])

    # *** features of decomposition *** #

    @abc.abstractmethod
    def is_sparse(self):
        """
        Returns whether this is a decomposition of a sparse matrix.

        Returns
        -------
        bool
            Whether this is a decomposition of a sparse matrix.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def is_positive_semidefinite(self):
        """
        Returns whether this is a decomposition of a positive semi-definite matrix.

        Returns
        -------
        bool
            Whether this is a decomposition of a positive semi-definite matrix.
        """
        raise NotImplementedError

    def is_positive_definite(self):

        """
        Returns whether this is a decomposition of a positive definite matrix.

        Returns
        -------
        bool
            Whether this is a decomposition of a positive definite matrix.
        """
        return self.is_positive_semidefinite() and not self.is_singular()

    @abc.abstractmethod
    def is_finite(self):
        """
        Returns whether this is a decomposition representing a finite matrix.

        Returns
        -------
        bool
            Whether this is a decomposition representing a finite matrix.
        """
        raise NotImplementedError

    def check_finite(self, check_finite=True):
        """
        Check if this is a decomposition representing a finite matrix.

        Parameters
        ----------
        check_finite : bool
            Whether to perform this check.
            default: True

        Raises
        -------
        matrix.errors.DecompositionNotFiniteError
            If this is a decomposition representing a non-finite matrix.
        """

        if check_finite and not self.is_finite():
            raise matrix.errors.DecompositionNotFiniteError(self)

    @abc.abstractmethod
    def is_singular(self):
        """
        Returns whether this is a decomposition representing a singular matrix.

        Returns
        -------
        bool
            Whether this is a decomposition representing a singular matrix.
        """
        raise NotImplementedError

    def is_invertible(self):
        """
        Returns whether this is a decomposition representing an invertible matrix.

        Returns
        -------
        bool
            Whether this is a decomposition representing an invertible matrix.
        """

        return not self.is_singular()

    def check_invertible(self):
        """
        Check if this is a decomposition representing an invertible matrix.

        Raises
        -------
        matrix.errors.DecompositionSingularError
            If this is a decomposition representing a singular matrix.
        """

        if self.is_singular():
            raise matrix.errors.DecompositionSingularError(self)

    # *** save and load *** #

    @property
    @abc.abstractmethod
    def _attribute_names(self):
        raise NotImplementedError

    @staticmethod
    def _check_decomposition_filename(filename):
        filename_suffix = '.' + matrix.constants.DECOMPOSITION_FILENAME_EXTENSION
        if not filename.endswith(filename_suffix):
            filename = filename + filename_suffix
        return filename

    def _attribute_filename(self, attribute_name, is_sparse):
        if is_sparse:
            file_extension = matrix.constants.DECOMPOSITION_ATTRIBUTE_SPARSE_FILE_EXTENSION
        else:
            file_extension = matrix.constants.DECOMPOSITION_ATTRIBUTE_DENSE_FILE_EXTENSION
        filename = matrix.constants.DECOMPOSITION_ATTRIBUTE_FILENAME.format(
            attribute_name=attribute_name,
            file_extension=file_extension)
        return filename

    # *** save *** #

    def _save_type(self, dirname):
        os.makedirs(dirname, exist_ok=True)
        type_file = os.path.join(dirname, matrix.constants.DECOMPOSITION_TYPE_FILENAME)
        with open(type_file, 'w') as f:
            f.write(self.type_str)

    def _save_attribute(self, dirname, attribute_name):
        value = getattr(self, attribute_name)
        is_sparse = scipy.sparse.issparse(value)
        filename = self._attribute_filename(attribute_name, is_sparse)
        file = os.path.join(dirname, filename)
        if is_sparse:
            scipy.sparse.save_npz(file, value)
        else:
            np.savez_compressed(file, **{attribute_name: value})

    def save(self, filename):
        """ Saves this decomposition.

        Parameters
        ----------
        filename : str
            Where this decomposition should be saved.
        """

        # debug info
        matrix.logging.debug('Saving {} to {}.'.format(self, filename))

        # check filename
        filename = DecompositionBase._check_decomposition_filename(filename)

        # create directory for file
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        # save
        with tempfile.TemporaryDirectory() as untared_dir:
            # save all files to temporary directory
            self._save_type(untared_dir)
            for attribute_name in self._attribute_names:
                self._save_attribute(untared_dir, attribute_name)

            # make temporary directory to tar file
            with tarfile.open(filename, mode='w') as tarfile_object:
                for file in os.listdir(untared_dir):
                    tarfile_object.add(os.path.join(untared_dir, file), arcname=file)

    # *** load *** #

    @staticmethod
    def _load_from_tar_file(filename, tar_attribute_name, file_load_function):
        try:
            with tarfile.open(filename, mode='r') as tar_file_object:
                with tar_file_object.extractfile(tar_attribute_name) as attribut_file_object:
                    return file_load_function(attribut_file_object)
        except KeyError as e:
            raise matrix.errors.DecompositionInvalidFile(filename) from e

    @staticmethod
    def _load_type(filename):
        # check filename
        filename = DecompositionBase._check_decomposition_filename(filename)

        # define loader
        def file_load_function(buffered_reader):
            return buffered_reader.read().decode()

        # load
        return DecompositionBase._load_from_tar_file(
            filename,
            matrix.constants.DECOMPOSITION_TYPE_FILENAME,
            file_load_function)

    def _load_attribute(self, filename, attribute_name):
        # check filename
        filename = DecompositionBase._check_decomposition_filename(filename)

        # define loaders
        def file_load_function_sparse(buffered_reader):
            return scipy.sparse.load_npz(buffered_reader)

        def file_load_function_nonsparse(buffered_reader):
            with np.load(buffered_reader, allow_pickle=False) as npz_file_object:
                keys = npz_file_object.keys()
                if len(keys) != 1 or keys[0] != attribute_name:
                    raise matrix.errors.DecompositionInvalidFile(filename)
                array = npz_file_object[attribute_name]
                return array

        # load
        try:
            attribute = DecompositionBase._load_from_tar_file(
                filename,
                self._attribute_filename(attribute_name, is_sparse=True),
                file_load_function_sparse)
        except matrix.errors.DecompositionInvalidFile:
            attribute = DecompositionBase._load_from_tar_file(
                filename,
                self._attribute_filename(attribute_name, is_sparse=False),
                file_load_function_nonsparse)

        setattr(self, attribute_name, attribute)

    def load(self, filename):
        """ Loads a decomposition of this type.

        Parameters
        ----------
        filename : str
            Where the decomposition is saved.

        Raises
        ----------
        FileNotFoundError
            If the files are not found in the passed directory.
        """

        # debug info
        matrix.logging.debug('Loading decomposition of type {} from {}.'.format(self.type_str, filename))

        # check filename
        filename = DecompositionBase._check_decomposition_filename(filename)

        # check type
        type_str = self._load_type(filename)
        if type_str != self.type_str:
            raise matrix.errors.DecompositionInvalidDecompositionTypeFile(filename, type_str, self.type_str)

        # load attributes
        for attribute_name in self._attribute_names:
            self._load_attribute(filename, attribute_name)

    # *** multiply *** #

    def matrix_right_side_multiplication(self, x):
        """
        Calculates the right side (matrix-matrix or matrix-vector) product `A @ x`, where `A` is the composed matrix represented by this decomposition.

        Parameters
        ----------
        x : numpy.ndarray or scipy.sparse.spmatrix
            Vector or matrix in the product in the
            matrix-matrix or matrix-vector `A @ x`.
            It must hold `self.n == x.shape[0]`.

        Returns
        -------
        numpy.ndarray or scipy.sparse.spmatrix
            The result of `A @ x`.
        """

        x = matrix.util.as_matrix_or_vector(x)
        return self.composed_matrix @ x

    def matrix_both_sides_multiplication(self, x, y=None):
        """
        Calculates the both sides (matrix-matrix or matrix-vector) product `y.H @ A @ x`, where `A` is the composed matrix represented by this decomposition.

        Parameters
        ----------
        x : numpy.ndarray or scipy.sparse.spmatrix
            Vector or matrix in the product `y.H @ A @ x`.
            It must hold `self.n == x.shape[0]`.
        y : numpy.ndarray or scipy.sparse.spmatrix
            Vector or matrix in the product `y.H @ A @ x`.
            It must hold `self.n == y.shape[0]`.
            optional, default: If y is not passed, x is used as y.

        Returns
        -------
        numpy.ndarray or scipy.sparse.spmatrix
            The result of `x.H @ A @ y`.
        """

        x = matrix.util.as_matrix_or_vector(x)
        if y is None:
            y = x
        else:
            y = matrix.util.as_matrix_or_vector(y)
        y = y.transpose().conj()
        return y @ self.matrix_right_side_multiplication(x)

    @abc.abstractmethod
    def inverse_matrix_right_side_multiplication(self, x):
        """
        Calculates the right side (matrix-matrix or matrix-vector) product `B @ x`, where `B` is the matrix inverse of the composed matrix represented by this decomposition.

        Parameters
        ----------
        x : numpy.ndarray or scipy.sparse.spmatrix
            Vector or matrix in the product in the
            matrix-matrix or matrix-vector `B @ x`.
            It must hold `self.n == x.shape[0]`.

        Returns
        -------
        numpy.ndarray or scipy.sparse.spmatrix
            The result of `B @ x`.

        Raises
        ------
        matrix.errors.DecompositionSingularError
            If this is a decomposition representing a singular matrix.
        """
        raise NotImplementedError

    def inverse_matrix_both_sides_multiplication(self, x, y=None):
        """
        Calculates the both sides (matrix-matrix or matrix-vector) product `y.H @ B @ x`, where `B` is the mattrix inverse of the composed matrix represented by this decomposition.

        Parameters
        ----------
        x : numpy.ndarray or scipy.sparse.spmatrix
            Vector or matrix in the product `y.H @ B @ x`.
            It must hold `self.n == x.shape[0]`.
        y : numpy.ndarray or scipy.sparse.spmatrix
            Vector or matrix in the product `y.H @ B @ x`.
            It must hold `self.n == y.shape[0]`.
            optional, default: If y is not passed, x is used as y.

        Returns
        -------
        numpy.ndarray or scipy.sparse.spmatrix
            The result of `x.H @ A @ y`.

        Raises
        ------
        matrix.errors.DecompositionSingularError
            If this is a decomposition representing a singular matrix.
        """

        x = matrix.util.as_matrix_or_vector(x)
        if y is None:
            y = x
        else:
            y = matrix.util.as_matrix_or_vector(y)
        y = y.transpose().conj()
        return y @ self.inverse_matrix_right_side_multiplication(x)

    # *** solve systems of linear equations *** #

    def solve(self, b):
        """
        Solves the equation `A x = b` regarding `x`, where `A` is the composed matrix represented by this decomposition.

        Parameters
        ----------
        b : numpy.ndarray or scipy.sparse.spmatrix
            Right-hand side vector or matrix in equation `A x = b`.
            It must hold `self.n == b.shape[0]`.

        Returns
        -------
        numpy.ndarray or scipy.sparse.spmatrix
            An `x` so that `A x = b`.
            The shape of `x` matches the shape of `b`.

        Raises
        ------
        matrix.errors.DecompositionSingularError
            If this is a decomposition representing a singular matrix.
        """

        # debug info
        matrix.logging.debug('Solving linear system with {}.'.format(self))
        # solve
        return self.inverse_matrix_right_side_multiplication(b)


class LDL_Decomposition(DecompositionBase):
    """ A matrix decomposition where :math:`LDL^H` is the decomposed (permuted) matrix.

    `L` is a lower triangle matrix with ones on the diagonal. `D` is a diagonal matrix.
    Only the diagonal values of `D` are stored.
    """

    type_str = matrix.constants.LDL_DECOMPOSITION_TYPE
    """ :class:`str`: The type of this decomposition represented as string. """

    def __init__(self, L=None, d=None, p=None):
        """
        Parameters
        ----------
        L : numpy.ndarray or scipy.sparse.spmatrix
            The matrix `L` of the decomposition.
            optional, If it is not set yet, it must be set later.
        d : numpy.ndarray
            The vector of the diagonal components of `D` of the decompositon.
            optional, If it is not set yet, it must be set later.
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
    def composed_matrix(self):
        A = self.L @ self.D @ self.L.transpose().conj()
        A = self.unpermute_matrix(A)
        return A

    # *** decomposition specific properties *** #

    @property
    def L(self):
        """:class:`numpy.matrix` or :class:`scipy.sparse.spmatrix`: The matrix `L` of the decomposition."""
        return self._L

    @L.setter
    def L(self, L):
        if L is not None:
            L = matrix.util.as_matrix(L)
            if np.any(L.diagonal() != 1):
                raise ValueError('The diagonal values of the lower triangle matrix L must all be equal to 1.')
            self._L = L
        else:
            try:
                del self._L
            except AttributeError:
                pass

    @property
    def d(self):
        """:class:`numpy.ndarray`: The diagonal vector of the matrix `D` of the decomposition."""
        return self._d

    @d.setter
    def d(self, d):
        if d is not None:
            d = matrix.util.as_vector(d)
            if np.iscomplexobj(d) and np.all(np.isreal(d)):
                d = d.real
            self._d = d
        else:
            try:
                del self._d
            except AttributeError:
                pass

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

    def is_equal(self, other):
        return (super().is_equal(other) and
                np.all(self.d == other.d) and
                matrix.util.is_equal(self.L, other.L))

    def is_almost_equal(self, other, rtol=1e-04, atol=1e-06):
        return (super().is_almost_equal(other) and
                np.allclose(self.d, other.d, rtol=rtol, atol=atol) and
                matrix.util.is_almost_equal(self.L, other.L, rtol=rtol, atol=atol))

    # *** convert type *** #

    def as_LL_Decomposition(self):
        L = self.L
        d = self.d
        p = self.p

        # check d for negative entries
        for i, d_i in enumerate(d):
            if d_i < 0:
                p_i = p[i]
                raise matrix.errors.NoDecompositionPossibleWithProblematicSubdecompositionError(
                    self, matrix.constants.LL_DECOMPOSITION_TYPE, p_i)

        # compute new d
        d = np.sqrt(d)

        # compute new L
        D = scipy.sparse.diags(d)
        L = L @ D

        # construct new decomposition
        return LL_Decomposition(L, p=p)

    def as_LDL_DecompositionCompressed(self):
        return LDL_DecompositionCompressed(self.LD, p=self.p)

    def as_type(self, type_str, copy=False):
        try:
            return super().as_type(type_str, copy=copy)
        except matrix.errors.NoDecompositionConversionImplementedError:
            if type_str == matrix.constants.LL_DECOMPOSITION_TYPE:
                return self.as_LL_Decomposition()
            elif type_str == matrix.constants.LDL_DECOMPOSITION_COMPRESSED_TYPE:
                return self.as_LDL_DecompositionCompressed()
            else:
                raise

    # *** features of decomposition *** #

    def is_sparse(self):
        return scipy.sparse.issparse(self.L)

    def is_finite(self):
        return matrix.util.is_finite(self.L) and matrix.util.is_finite(self.d)

    def is_positive_semidefinite(self):
        return np.all(self.d >= 0)

    def is_positive_definite(self):
        eps = np.finfo(self.d.dtype).resolution
        return np.all(self.d >= eps)

    def is_singular(self):
        eps = np.finfo(self.d.dtype).resolution
        return np.any(np.abs(self.d) < eps)

    # *** save and load *** #

    @property
    def _attribute_names(self):
        return ('L', 'd', 'p')

    # *** multiply *** #

    def matrix_right_side_multiplication(self, x):
        x = matrix.util.as_matrix_or_vector(x)
        x = x[self.p]
        x = self.L.transpose().conj() @ x
        x = self.D @ x
        x = self.L @ x
        x = x[self.p_inverse]
        return x

    def matrix_both_sides_multiplication(self, x, y=None):
        x = matrix.util.as_matrix_or_vector(x)
        x = x[self.p]
        x = self.L.transpose().conj() @ x
        if y is not None:
            y = matrix.util.as_matrix_or_vector(y)
            y = y[self.p]
            y = y.transpose().conj()
            y = y @ self.L
        else:
            y = x.transpose().conj()
        return y @ self.D @ x

    def inverse_matrix_right_side_multiplication(self, x):
        x = matrix.util.as_matrix_or_vector(x)
        self.check_invertible()
        x = x[self.p]
        x = matrix.util.solve_triangular(self.L, x, lower=True, unit_diagonal=True, overwrite_b=True)
        x = scipy.sparse.diags(1 / self.d) @ x
        x = matrix.util.solve_triangular(self.L.transpose().conj(), x, lower=False, unit_diagonal=True, overwrite_b=True)
        x = x[self.p_inverse]
        return x

    def inverse_matrix_both_sides_multiplication(self, x, y=None):
        x = matrix.util.as_matrix_or_vector(x)
        self.check_invertible()
        x = x[self.p]
        x = matrix.util.solve_triangular(self.L, x, lower=True, unit_diagonal=True, overwrite_b=True)
        if y is not None:
            y = matrix.util.as_matrix_or_vector(y)
            y = y[self.p]
            y = matrix.util.solve_triangular(self.L, y, lower=True, unit_diagonal=True, overwrite_b=True)
            y = y.transpose().conj()
        else:
            y = x.transpose().conj()
        return y @ scipy.sparse.diags(1 / self.d) @ x


class LDL_DecompositionCompressed(DecompositionBase):
    """ A matrix decomposition where :math:`LDL^H` is the decomposed (permuted) matrix.

    `L` is a lower triangle matrix with ones on the diagonal. `D` is a diagonal matrix.
    `L` and `D` are stored in one matrix whose diagonal values are the diagonal values of `D`
    and whose off-diagonal values are those of `L`.
    """

    type_str = matrix.constants.LDL_DECOMPOSITION_COMPRESSED_TYPE
    """ :class:`str`: The type of this decomposition represented as string. """

    def __init__(self, LD=None, p=None):
        """
        Parameters
        ----------
        LD : numpy.ndarray or scipy.sparse.spmatrix
            A matrix whose diagonal values are the diagonal values of `D` and whose off-diagonal values are those of `L`.
            optional, If it is not set yet, it must be set later.
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
    def composed_matrix(self):
        return self.as_LDL_Decomposition().composed_matrix

    # *** decomposition specific properties *** #

    @property
    def LD(self):
        """:class:`numpy.matrix` or :class:`scipy.sparse.spmatrix`: A matrix whose diagonal values are the diagonal values of `D` and whose off-diagonal values are those of `L`."""
        return self._LD

    @LD.setter
    def LD(self, LD):
        if LD is not None:
            LD = matrix.util.as_matrix(LD)
            self._LD = LD
        else:
            try:
                del self._LD
            except AttributeError:
                pass

    @property
    def d(self):
        """:class:`numpy.ndarray`: The diagonal vector of the matrix `D` of the decomposition."""
        LD = self.LD
        if not self.is_sparse():
            LD = np.asarray(LD)
        d = LD.diagonal()
        if np.iscomplexobj(d) and np.all(np.isreal(d)):
            d = d.real
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

    def is_equal(self, other):
        return (super().is_equal(other) and
                matrix.util.is_equal(self.LD, other.LD))

    def is_almost_equal(self, other, rtol=1e-04, atol=1e-06):
        return (super().is_almost_equal(other) and
                matrix.util.is_almost_equal(self.LD, other.LD, rtol=rtol, atol=atol))

    # *** convert type *** #

    def as_LDL_Decomposition(self):
        return LDL_Decomposition(self.L, self.d, p=self.p)

    def as_type(self, type_str, copy=False):
        try:
            return super().as_type(type_str, copy=copy)
        except matrix.errors.NoDecompositionConversionImplementedError:
            if type_str == matrix.constants.LDL_DECOMPOSITION_TYPE:
                return self.as_LDL_Decomposition()
            elif type_str == matrix.constants.LL_DECOMPOSITION_TYPE:
                return self.as_LDL_Decomposition().as_LL_Decomposition()
            else:
                raise

    # *** features of decomposition *** #

    def is_sparse(self):
        return scipy.sparse.issparse(self.LD)

    def is_finite(self):
        return matrix.util.is_finite(self.LD)

    def is_positive_semidefinite(self):
        return np.all(self.d >= 0)

    def is_positive_definite(self):
        eps = np.finfo(self.d.dtype).resolution
        return np.all(self.d >= eps)

    def is_singular(self):
        eps = np.finfo(self.d.dtype).resolution
        return np.any(np.abs(self.d) < eps)

    # *** save and load *** #

    @property
    def _attribute_names(self):
        return ('LD', 'p')

    # *** multiply *** #

    def matrix_right_side_multiplication(self, x):
        return self.as_LDL_Decomposition().matrix_right_side_multiplication(x)

    def matrix_both_sides_multiplication(self, x, y=None):
        return self.as_LDL_Decomposition().matrix_both_sides_multiplication(x, y=y)

    def inverse_matrix_right_side_multiplication(self, x):
        return self.as_LDL_Decomposition().inverse_matrix_right_side_multiplication(x)

    def inverse_matrix_both_sides_multiplication(self, x, y=None):
        return self.as_LDL_Decomposition().inverse_matrix_both_sides_multiplication(x, y=y)


class LL_Decomposition(DecompositionBase):
    """ A matrix decomposition where :math:`LL^H` is the decomposed (permuted) matrix.

    `L` is a lower triangle matrix with ones on the diagonal.
    This decomposition is also called Cholesky decomposition.
    """

    type_str = matrix.constants.LL_DECOMPOSITION_TYPE
    """ :class:`str`: The type of this decomposition represented as string. """

    def __init__(self, L=None, p=None):
        """
        Parameters
        ----------
        L : numpy.ndarray or scipy.sparse.spmatrix
            The matrix `L` of the decomposition.
            optional, If it is not set yet, it must be set later.
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
    def composed_matrix(self):
        A = self.L @ self.L.transpose().conj()
        A = self.unpermute_matrix(A)
        return A

    # *** decomposition specific properties *** #

    @property
    def L(self):
        """:class:`numpy.matrix` or :class:`scipy.sparse.spmatrix`: The matrix `L` of the decomposition."""
        return self._L

    @L.setter
    def L(self, L):
        if L is not None:
            L = matrix.util.as_matrix(L)
            self._L = L
        else:
            try:
                del self._L
            except AttributeError:
                pass

    # *** compare methods *** #

    def is_equal(self, other):
        return (super().is_equal(other) and
                matrix.util.is_equal(self.L, other.L))

    def is_almost_equal(self, other, rtol=1e-04, atol=1e-06):
        return (super().is_almost_equal(other) and
                matrix.util.is_almost_equal(self.L, other.L, rtol=rtol, atol=atol))

    # *** convert type *** #

    @property
    def _d(self):
        """:class:`numpy.ndarray`: The diagonal vector of `L`."""
        L = self.L
        if not self.is_sparse():
            L = np.asarray(L)
        d = L.diagonal()
        if np.iscomplexobj(d) and np.all(np.isreal(d)):
            d = d.real
        return d

    def as_LDL_Decomposition(self):
        L = self.L
        p = self.p

        # d inverse
        d = self._d
        d_zero_mask = d == 0
        d_inverse = np.empty_like(d)
        d_inverse[d_zero_mask] = 0
        d_inverse[~d_zero_mask] = 1 / d[~d_zero_mask]
        assert np.all(np.isfinite(d_inverse[np.isfinite(d)]))

        # check entries where diagonal is zero
        n = self.n
        if np.any(d_zero_mask):
            for i in np.where(d_zero_mask)[0]:
                for j in range(i + 1, n):
                    if not np.isclose(L[j, i], 0):
                        raise matrix.errors.NoDecompositionPossibleWithProblematicSubdecompositionError(
                            self, matrix.constants.LDL_DECOMPOSITION_TYPE, p[i])

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
        d = d * d.conj()

        # construct new decomposition
        return LDL_Decomposition(L, d, p=p)

    def as_type(self, type_str, copy=False):
        try:
            return super().as_type(type_str, copy=copy)
        except matrix.errors.NoDecompositionConversionImplementedError:
            if type_str == matrix.constants.LDL_DECOMPOSITION_TYPE:
                return self.as_LDL_Decomposition()
            elif type_str == matrix.constants.LDL_DECOMPOSITION_COMPRESSED_TYPE:
                return self.as_LDL_Decomposition().as_LDL_DecompositionCompressed()
            else:
                raise

    # *** features of decomposition *** #

    def is_sparse(self):
        return scipy.sparse.issparse(self.L)

    def is_finite(self):
        return matrix.util.is_finite(self.L)

    def is_positive_semidefinite(self):
        return True

    def is_positive_definite(self):
        eps = np.finfo(self._d.dtype).resolution
        return np.all(self._d >= eps)

    def is_singular(self):
        eps = np.finfo(self._d.dtype).resolution
        return np.any(np.abs(self._d) < eps)

    # *** save and load *** #

    @property
    def _attribute_names(self):
        return ('L', 'p')

    # *** multiply *** #

    def matrix_right_side_multiplication(self, x):
        x = matrix.util.as_matrix_or_vector(x)
        x = x[self.p]
        x = self.L.transpose().conj() @ x
        x = self.L @ x
        x = x[self.p_inverse]
        return x

    def matrix_both_sides_multiplication(self, x, y=None):
        x = matrix.util.as_matrix_or_vector(x)
        x = x[self.p]
        L_H = self.L.transpose().conj()
        x = L_H @ x
        if y is not None:
            y = matrix.util.as_matrix_or_vector(y)
            y = y[self.p]
            y = L_H @ y
        else:
            y = x
        y = y.transpose().conj()
        return y @ x

    def inverse_matrix_right_side_multiplication(self, x):
        x = matrix.util.as_matrix_or_vector(x)
        self.check_invertible()
        x = x[self.p]
        x = matrix.util.solve_triangular(self.L, x, lower=True, unit_diagonal=False, overwrite_b=True)
        x = matrix.util.solve_triangular(self.L.transpose().conj(), x, lower=False, unit_diagonal=False, overwrite_b=True)
        x = x[self.p_inverse]
        return x

    def inverse_matrix_both_sides_multiplication(self, x, y=None):
        x = matrix.util.as_matrix_or_vector(x)
        self.check_invertible()
        x = x[self.p]
        x = matrix.util.solve_triangular(self.L, x, lower=True, unit_diagonal=False, overwrite_b=True)
        if y is not None:
            y = matrix.util.as_matrix_or_vector(y)
            y = y[self.p]
            y = matrix.util.solve_triangular(self.L, y, lower=True, unit_diagonal=False, overwrite_b=True)
        else:
            y = x
        y = y.transpose().conj()
        return y @ x


# functions handling decompositions

def save(filename, decomposition):
    """ Saves a decomposition.

    Parameters
    ----------
    filename : str
        Where the decomposition should be saved.
    decomposition : DecompositionBase
        The decomposition that should be saved.
    """

    # debug info
    matrix.logging.debug('Saving decomposition to {}.'.format(filename))
    # save
    decomposition.save(filename)


def load(filename):
    """ Loads a decomposition.

    Parameters
    ----------
    filename : str
        Where the decomposition is saved.
    """

    # debug info
    matrix.logging.debug('Loading decomposition from {}.'.format(filename))
    # load
    type_str = DecompositionBase._load_type(filename)
    decomposition_classes = (LDL_Decomposition, LDL_DecompositionCompressed, LL_Decomposition, DecompositionBase)
    for decomposition_class in decomposition_classes:
        if decomposition_class.type_str == type_str:
            decomposition = decomposition_class()
            decomposition.load(filename)
            return decomposition
    raise matrix.errors.DecompositionInvalidFile(filename)
