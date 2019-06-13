Errors
======

This is an overview about the exceptions that could arise in this library. They are available in `matrix.errors`:


If a matrix should be decomposed with `matrix.decompose` and the desired decomposition is not computable, the following exceptions can be raised:

NoDecompositionPossibleError
----------------------------
.. autoexception:: matrix.errors.NoDecompositionPossibleError
    :show-inheritance:

NoDecompositionPossibleWithProblematicSubdecompositionError
-----------------------------------------------------------
.. autoexception:: matrix.errors.NoDecompositionPossibleWithProblematicSubdecompositionError
    :show-inheritance:

NoDecompositionPossibleTooManyEntriesError
------------------------------------------
.. autoexception:: matrix.errors.NoDecompositionPossibleTooManyEntriesError
    :show-inheritance:

NoDecompositionConversionImplementedError
-----------------------------------------
.. autoexception:: matrix.errors.NoDecompositionConversionImplementedError
    :show-inheritance:


If a matrix has an invalid characteristic, the following exceptions can occur:

MatrixError
-----------
.. autoexception:: matrix.errors.MatrixError
    :show-inheritance:

MatrixNotSquareError
--------------------
.. autoexception:: matrix.errors.MatrixNotSquareError
    :show-inheritance:

MatrixNotFiniteError
--------------------
.. autoexception:: matrix.errors.MatrixNotFiniteError
    :show-inheritance:

MatrixSingularError
-------------------
.. autoexception:: matrix.errors.MatrixSingularError
    :show-inheritance:

MatrixNotHermitianError
-----------------------------------------
.. autoexception:: matrix.errors.MatrixNotHermitianError
    :show-inheritance:

MatrixComplexDiagonalValueError
-----------------------------------------
.. autoexception:: matrix.errors.MatrixComplexDiagonalValueError
    :show-inheritance:


If the matrix represented by a decomposition has an invalid characteristic, the following exceptions can occur:

DecompositionError
------------------
.. autoexception:: matrix.errors.DecompositionError
    :show-inheritance:

DecompositionNotFiniteError
---------------------------
.. autoexception:: matrix.errors.DecompositionNotFiniteError
    :show-inheritance:

DecompositionSingularError
--------------------------
.. autoexception:: matrix.errors.DecompositionSingularError
    :show-inheritance:

If a decomposition should be loaded from a file which is not a valid decomposition file, the following exception is raised:

DecompositionInvalidFile
------------------------
.. autoexception:: matrix.errors.DecompositionInvalidFile
    :show-inheritance:

If a decomposition should be loaded from a file which contains a type that does not fit to the type of the decomposition where it should be loaded into, the following exception is raised:

DecompositionInvalidDecompositionTypeFile
-----------------------------------------
.. autoexception:: matrix.errors.DecompositionInvalidDecompositionTypeFile
    :show-inheritance:


The following exception is the base exception from which all other exceptions in this package are derived:

BaseError
---------
.. autoexception:: matrix.errors.BaseError
    :show-inheritance:

