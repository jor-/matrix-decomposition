Errors
======

This is an overview about the exceptions that could arise in this package. They are available in `matrix.errors`:


If a matrix should be decomposed with `matrix.decompose` and the desired decomposition is not computable, the following exceptions can be raised:

NoDecompositionPossibleError
----------------------------
.. autoclass:: matrix.errors.NoDecompositionPossibleError
    :show-inheritance:

NoDecompositionPossibleWithProblematicSubdecompositionError
-----------------------------------------------------------
.. autoclass:: matrix.errors.NoDecompositionPossibleWithProblematicSubdecompositionError
    :show-inheritance:

NoDecompositionPossibleTooManyEntriesError
------------------------------------------
.. autoclass:: matrix.errors.NoDecompositionPossibleTooManyEntriesError
    :show-inheritance:

NoDecompositionConversionImplementedError
-----------------------------------------
.. autoclass:: matrix.errors.NoDecompositionConversionImplementedError
    :show-inheritance:


If a matrix has an invalid characteristic, the following exceptions can occur:

MatrixError
-----------
.. autoclass:: matrix.errors.MatrixError
    :show-inheritance:

MatrixNotSquareError
--------------------
.. autoclass:: matrix.errors.MatrixNotSquareError
    :show-inheritance:

MatrixNotFiniteError
--------------------
.. autoclass:: matrix.errors.MatrixNotFiniteError
    :show-inheritance:

MatrixSingularError
-------------------
.. autoclass:: matrix.errors.MatrixSingularError
    :show-inheritance:

MatrixNotHermitianError
-----------------------------------------
.. autoclass:: matrix.errors.MatrixNotHermitianError
    :show-inheritance:

MatrixComplexDiagonalValueError
-----------------------------------------
.. autoclass:: matrix.errors.MatrixComplexDiagonalValueError
    :show-inheritance:


If the matrix represented by a decomposition has an invalid characteristic, the following exceptions can occur:

DecompositionError
------------------
.. autoclass:: matrix.errors.DecompositionError
    :show-inheritance:

DecompositionNotFiniteError
---------------------------
.. autoclass:: matrix.errors.DecompositionNotFiniteError
    :show-inheritance:

DecompositionSingularError
--------------------------
.. autoclass:: matrix.errors.DecompositionSingularError
    :show-inheritance:

If a decomposition should be loaded from a file which is not a valid decomposition file, the following exception is raised:

DecompositionInvalidFile
------------------------
.. autoclass:: matrix.errors.DecompositionInvalidFile
    :show-inheritance:

If a decomposition should be loaded from a file which caontains a type which does not fit to the type of the decomposition where it should be loaded into, the following exception is raised:

DecompositionInvalidDecompositionTypeFile
-----------------------------------------
.. autoclass:: matrix.errors.DecompositionInvalidDecompositionTypeFile
    :show-inheritance:


The following exception is the base exception from which all other exceptions in this package are derived:

BaseError
---------
.. autoclass:: matrix.errors.BaseError
    :show-inheritance:

