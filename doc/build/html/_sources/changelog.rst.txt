Changelog
=========


1.1
---
    * Positive semidefinite approximation algorithms of GMW and SE type have been added.
    * Permutation method, with numerical stability as main focus, has been added to positive semidefinite approximation algorithm.
    * Positive semidefinite approximation algorithm are moved into separate package.
      (`matrix.approximate` to `matrix.approximation.positive_semidefinite`)

1.0.1
-----
    * Approximation functions now also work if an overflows occurs.
    * NumPys matrix is avoided because it is deprecated now.

1.0
---
    * Approximation functions are slightly faster now.
    * Better overflow handling is now used in approximation functions.
    * Prebuild html documentation are now included.
    * Function for approximating a matrix by a positive semidefinite matrix (`matrix.approximate.positive_semidefinite_matrix`) has been removed.

0.8
---
    * Approximation functions have been replaced by more sophisticated approximation functions.
    * Explicit function for approximating a matrix by a positive (semi)definite matrix has been added.
    * Universal save and load functions have been added.
    * Decompositions have obtained is_equal and is_almost_equal methods.
    * Functions to multiply the matrix, represented by a decomposition, or its inverse with a matrix or a vector have been added.
    * It is now possible to pass permutation vectors to approximate and decompose methods.


0.7
---
    * Lineare systems associated to matrices or decompositions can now be solved.
    * Invertibility of matrices and decompositions can now be examined.
    * Decompositions can now be examined to see if they contain only finite values.


0.6
---
    * Decompositions are now saveable and loadable.


0.5
---
    * Matrices can now be approximated by decompositions.


0.4
---
    * Positive definiteness and positive semi-definiteness of matrices and decompositions can now be examined.


0.3
---
    * Dense and sparse matrices are now decomposable into several types (LL, LDL, LDL compressed).


0.2
---
    * Decompositons are now convertable to other decompositon types.
    * Decompositions are now comparable.


0.1
---
    * Several decompositions types (LL, LDL, LDL compressed) have been added.
    * Several permutation capabilities have been added.

