Errors
======

This is an overview about the exceptions that could arise in this package.
They are available in `matrix.errors`:


The following exceptions can be raised if a matrix should be decomposed with
`matrix.decompose` and the desired decomposition is not computable.

MatrixNoDecompositionPossibleError
----------------------------------
.. autoclass:: matrix.errors.MatrixNoDecompositionPossibleError
    :show-inheritance:

MatrixNoLDLDecompositionPossibleError
-------------------------------------
.. autoclass:: matrix.errors.MatrixNoLDLDecompositionPossibleError
    :show-inheritance:

MatrixNoLLDecompositionPossibleError
------------------------------------
.. autoclass:: matrix.errors.MatrixNoLLDecompositionPossibleError
    :show-inheritance:

MatrixNoDecompositionPossibleWithProblematicSubdecompositionError
-----------------------------------------------------------------
.. autoclass:: matrix.errors.MatrixNoDecompositionPossibleWithProblematicSubdecompositionError
    :show-inheritance:

MatrixDecompositionNoConversionImplementedError
-----------------------------------------------
.. autoclass:: matrix.errors.MatrixDecompositionNoConversionImplementedError
    :show-inheritance:


The following exceptions can occur if a matrix has an invalid characteristic.

MatrixNotSquareError
--------------------
.. autoclass:: matrix.errors.MatrixNotSquareError
    :show-inheritance:

MatrixNotFiniteError
--------------------
.. autoclass:: matrix.errors.MatrixNotFiniteError
    :show-inheritance:


The following exceptions can occur if the matrix represented by a decomposition has an invalid characteristic.

MatrixDecompositionNotFiniteError
---------------------------------
.. autoclass:: matrix.errors.MatrixDecompositionNotFiniteError
    :show-inheritance:


The following exception is the base exception from which all other exceptions in this package are derived.

MatrixError
-----------
.. autoclass:: matrix.errors.MatrixError
    :show-inheritance:

