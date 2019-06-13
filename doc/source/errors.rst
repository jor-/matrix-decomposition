Errors
======

This is an overview about the exceptions that could arise in this library. They are available in `matrix.errors`:


The following exception is the base exception from which all other exceptions in this package are derived:

BaseError
---------
.. autoexception:: matrix.errors.BaseError
    :show-inheritance:


If a matrix has an invalid properties, the following exceptions can occur:

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


If a desired decomposition is not computable, the following exceptions can be raised:

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


If the matrix, represented by a decomposition, has an invalid characteristic, the following exceptions can occur:

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


If a decomposition could not be loaded from a file, the following exceptions can be raised:

DecompositionInvalidFile
------------------------
.. autoexception:: matrix.errors.DecompositionInvalidFile
    :show-inheritance:

DecompositionInvalidDecompositionTypeFile
-----------------------------------------
.. autoexception:: matrix.errors.DecompositionInvalidDecompositionTypeFile
    :show-inheritance:


If the computation needs more iterations than the maximal number of iterations, the following exception occurs:

TooManyIterationsError
----------------------
.. autoexception:: matrix.errors.TooManyIterationsError
    :show-inheritance:

