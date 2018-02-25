

# *** base exceptions *** #

class BaseError(Exception):
    """ This is the base exception for all exceptions in this package. """

    def __init__(self, message):
        self.message = message
        super().__init__(message)

    def __str__(self):
        return self.message


# *** matrix exceptions *** #

class MatrixError(BaseError):
    """ An exception related to a matrix. """

    def __init__(self, matrix, message=None):
        self.matrix = matrix
        if message is None:
            message = 'Error with matrix with shape {}'.format(self.matrix.shape)
        super().__init__(message)


class MatrixNotSquareError(MatrixError):
    """ A matrix is not a square matrix although this is required. """

    def __init__(self, matrix):
        message = 'Matrix with shape {} is not a square matrix.'.format(matrix.shape)
        super().__init__(matrix, message=message)


class MatrixNotFiniteError(MatrixError):
    """ A matrix has non-finite entries although a finite matrix is required. """

    def __init__(self, matrix):
        message = 'Matrix with shape {} has not finite entries.'.format(matrix.shape)
        super().__init__(matrix, message=message)


class MatrixSingularError(MatrixError):
    """ A matrix is singular although an invertible matrix is required. """

    def __init__(self, matrix):
        message = 'Matrix with shape {} is singular.'.format(matrix.shape)
        super().__init__(matrix, message=message)


class MatrixNotHermitianError(MatrixError):
    """ A matrix is not Hermitian although a Hermitian matrix is required. """

    def __init__(self, matrix, i=None, j=None):
        message = 'Matrix with shape {} is not Hermitian.'.format(matrix.shape)
        if i is not None and j is not None:
            if i != j:
                message += (' Its value at index ({i}, {j}) is {A_i_j}'
                            ' and its value at index ({j}, {i}) is {A_j_i}.'
                            ).format(i=i, j=j, A_i_j=matrix[i, j], A_j_i=matrix[j, i])
            else:
                message += (' Its {i}-th diagonal value {A_i_i} is complex.'
                            ).format(i=i, A_i_i=matrix[i, i])
        super().__init__(matrix, message=message)


class MatrixComplexDiagonalValueError(MatrixNotHermitianError):
    """ A matrix has complex diagonal values although real diagonal values are required. """

    def __init__(self, matrix, i=None):
        super().__init__(matrix, matrix, i=i, j=i)


# *** general decomposition exceptions *** #

class DecompositionError(BaseError):
    """ An exception related to a decomposition. """

    def __init__(self, decomposition, message=None):
        self.decomposition = decomposition
        if message is None:
            message = 'Error with decomposition {}'.format(self.decomposition)
        super().__init__(message)


class DecompositionNotFiniteError(DecompositionError):
    """ A decomposition of a matrix has non-finite entries although a finite matrix is required. """

    def __init__(self, decomposition):
        message = 'Decomposition {} has non-finite entries.'.format(decomposition)
        super().__init__(decomposition, message)


class DecompositionSingularError(DecompositionError):
    """ A decomposition represents a singular matrix although a non-singular matrix is required. """

    def __init__(self, decomposition):
        message = 'Decomposition {} represents a singular matrix.'.format(decomposition)
        super().__init__(decomposition, message)


class DecompositionInvalidFile(DecompositionError, OSError):
    """ A decomposition indicated that a decomposition should be loaded from an invalid file. """

    def __init__(self, filename):
        self.filename = filename
        message = 'File {} is not a valid decomposition file.'.format(filename)
        super().__init__(message)


class DecompositionInvalidDecompositionTypeFile(DecompositionInvalidFile):
    """ A decomposition indicated that a decomposition should be loaded from an file in which another decomposition type is stored. """

    def __init__(self, filename, type_file, type_needed):
        self.filename = filename
        message = 'File {} contains a decomposition of type {} but type {} is needed.'.format(filename, type_file, type_needed)
        super().__init__(message)


# *** decomposition not calculable exceptions *** #

class NoDecompositionPossibleError(BaseError):
    """ It is to possible to calculate a desired matrix decomposition. """

    def __init__(self, base, desired_type):
        self.desired_type = desired_type
        self.base = base
        message = 'Base {} can not be converted to type {}.'.format(base, desired_type)
        super().__init__(message)


class NoDecompositionPossibleWithProblematicSubdecompositionError(NoDecompositionPossibleError):
    """ It is not possible to calculate a desired matrix decomposition. Only a subdecompostion could be calculated """

    def __init__(self, base, desired_type, problematic_leading_principal_submatrix_index, subdecomposition=None):
        super().__init__(base, desired_type)
        self.problematic_leading_principal_submatrix_index = problematic_leading_principal_submatrix_index
        if subdecomposition is not None:
            self.subdecomposition = subdecomposition


class NoDecompositionPossibleTooManyEntriesError(NoDecompositionPossibleError):
    """ The matrix decomposition is not possible for this matrix because it would have too many entries. """

    def __init__(self, matrix, desired_type):
        super().__init__(matrix, desired_type)


class NoDecompositionConversionImplementedError(NoDecompositionPossibleError):
    """ A decomposition conversion is not implemented for this type. """

    def __init__(self, decomposition, desired_type):
        super().__init__(decomposition, desired_type)
