Functions
=========

Several functions are included in this package.
The most important ones are summarized here.


Positive semidefinite approximation of a matrix
-----------------------------------------------

.. autofunction:: matrix.approximation.positive_semidefinite.positive_semidefinite_matrix
.. autofunction:: matrix.approximation.positive_semidefinite.decomposition
.. autodata:: matrix.approximation.positive_semidefinite.APPROXIMATION_ONLY_PERMUTATION_METHODS
.. autofunction:: matrix.approximation.positive_semidefinite.GMW_81
.. autofunction:: matrix.approximation.positive_semidefinite.GMW_T1
.. autofunction:: matrix.approximation.positive_semidefinite.GMW_T2
.. autofunction:: matrix.approximation.positive_semidefinite.SE_90
.. autofunction:: matrix.approximation.positive_semidefinite.SE_99
.. autofunction:: matrix.approximation.positive_semidefinite.SE_T1


Nearest matrix with specific properties
---------------------------------------

.. automodule:: matrix.nearest
    :members:


Decompose a matrix
------------------

.. autofunction:: matrix.decompose

.. autodata:: matrix.UNIVERSAL_PERMUTATION_METHODS

.. autodata:: matrix.SPARSE_ONLY_PERMUTATION_METHODS

.. autodata:: matrix.DECOMPOSITION_TYPES


Examine a matrix
----------------

.. autofunction:: matrix.is_positive_semidefinite

.. autofunction:: matrix.is_positive_definite

.. autofunction:: matrix.is_invertible


Solve system of linear equations
--------------------------------

.. autofunction:: matrix.solve

