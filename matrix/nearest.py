"""
This module provides functions to compute matrices with minimal difference to an input matrix and
specific properties.
"""


import numpy as np

import matrix.errors


def symmetric_matrix(A):
    """
    Computes a symmetric matrix with minimal difference to `A`
    for any unitarily invariant norm.

    Parameters
    ----------
    A : numpy.ndarray
        `A` must be a square matrix.

    Returns
    -------
    B : numpy.ndarray
        A symmetric matrix with minimal difference to `A` for any unitarily invariant norm.

    Notes
    -----
    The optimality is proven in [1].

    References
    ----------
    [1] Fan, K., & Hoffman, A. (1955).
        Some Metric Inequalities in the Space of Matrices.
        Proceedings of the American Mathematical Society, 6(1), 111-116. doi:10.2307/2032662
    """

    A = np.asanyarray(A)
    B = (A + A.transpose()) / 2
    return B


def skew_symmetric_matrix(A):
    """
    Computes a skew-symmetric matrix with minimal difference to `A`
    for any unitarily invariant norm.

    Parameters
    ----------
    A : numpy.ndarray
        `A` must be a square matrix.

    Returns
    -------
    B : numpy.ndarray
        A skew-symmetric matrix with minimal difference to `A` for any unitarily invariant norm.
    """

    A = np.asanyarray(A)
    B = (A - A.transpose()) / 2
    return B


def positive_semidefinite_matrix(A, symmetric=False):
    """
    Computes a symmetric positive semidefinite matrix with minimal difference to `A`
    in the Frobenius norm.

    Parameters
    ----------
    A : numpy.ndarray
        `A` must be Hermitian.
    symmetric : bool
        Whether `A` can be assumed to be symmetric or not.
        optional, default: False

    Returns
    -------
    B : numpy.ndarray
        A symmetric positive semidefinite Hermitian matrix with minimal difference to `A`
        in the Frobenius norm.

    Notes
    -----
    The method is presented in [1].

    References
    ----------
    [1] Higham, N. J.
        Computing a nearest symmetric positive semidefinite matrix
        Linear Algebra and its Applications, 1988, 103, 103-118
    """

    A = np.asanyarray(A)
    if not symmetric:
        B = symmetric_matrix(A)
    else:
        B = A
    d, v = np.linalg.eigh(B)
    C = (v * np.maximum(d, 0)).dot(v.transpose())
    C = symmetric_matrix(C)  # to force numerical symmetry
    return C


def correlation_matrix(A, max_iterations=10**3):
    """
    Computes a correlation matrix closest to `A` in the Frobenius norm.
    A correlation matrix is a positive semidefinite matrix with ones on the diagonal.

    Parameters
    ----------
    A : numpy.ndarray
        `A` must be Hermitian.
    max_iterations : int
        The maximal number of iterations used for the calculation.

    Returns
    -------
    B : numpy.ndarray
        A correlation matrix closest to `A` in the Frobenius norm.

    Raises
    ----------
    TooManyIterationsError
        If the computation needs more iterations than the maximal number of iterations.

    Notes
    -----
    The method is presented in [1].
    For the computation some code of Michael Croucher is used. See this source code for the license.

    References
    ----------
    [1] Higham, N. J.
        Computing the Nearest Correlation Matrix---A Problem from Finance
        IMA J. Numer. Anal., 2002, 22, 329-343
    """

    class ExceededMaxIterationsError(Exception):
        """
        LICENSE
        ~~~~~~~
        Copyright (c) 2014, Michael Croucher
        All rights reserved.

        Redistribution and use in source and binary forms, with or without
        modification, are permitted provided that the following conditions are met:

        * Redistributions of source code must retain the above copyright notice, this
          list of conditions and the following disclaimer.

        * Redistributions in binary form must reproduce the above copyright notice,
          this list of conditions and the following disclaimer in the documentation
          and/or other materials provided with the distribution.

        * Neither the name of nearest_correlation nor the names of its
          contributors may be used to endorse or promote products derived from
          this software without specific prior written permission.

        THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
        AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
        IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
        DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
        FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
        DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
        SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
        CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
        OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
        OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
        """

        def __init__(self, msg, matrix=[], iteration=[], ds=[]):
            self.msg = msg
            self.matrix = matrix
            self.iteration = iteration
            self.ds = ds

        def __str__(self):
            return repr(self.msg)

    def nearcorr(A, tol=[], flag=0, max_iterations=100, n_pos_eig=0,
                 weights=None, verbose=False,
                 except_on_too_many_iterations=True):
        """
        X = nearcorr(A, tol=[], flag=0, max_iterations=100, n_pos_eig=0,
            weights=None, print=0)

        Finds the nearest correlation matrix to the symmetric matrix A.

        ARGUMENTS
        ~~~~~~~~~
        A is a symmetric numpy array or a ExceededMaxIterationsError object

        tol is a convergence tolerance, which defaults to 16*EPS.
        If using flag == 1, tol must be a size 2 tuple, with first component
        the convergence tolerance and second component a tolerance
        for defining "sufficiently positive" eigenvalues.

        flag = 0: solve using full eigendecomposition (EIG).
        flag = 1: treat as "highly non-positive definite A" and solve
        using partial eigendecomposition (EIGS). CURRENTLY NOT IMPLEMENTED

        max_iterations is the maximum number of iterations (default 100,
        but may need to be increased).

        n_pos_eig (optional) is the known number of positive eigenvalues
        of A. CURRENTLY NOT IMPLEMENTED

        weights is an optional vector defining a diagonal weight matrix diag(W).

        verbose = True for display of intermediate output.
        CURRENTLY NOT IMPLEMENTED

        except_on_too_many_iterations = True to raise an exeption when
        number of iterations exceeds max_iterations
        except_on_too_many_iterations = False to silently return the best result
        found after max_iterations number of iterations

        ABOUT
        ~~~~~~
        This is a Python port by Michael Croucher, November 2014
        Thanks to Vedran Sego for many useful comments and suggestions.

        Original MATLAB code by N. J. Higham, 13/6/01, updated 30/1/13.
        Reference:  N. J. Higham, Computing the nearest correlation
        matrix---A problem from finance. IMA J. Numer. Anal.,
        22(3):329-343, 2002.

        LICENSE
        ~~~~~~~
        Copyright (c) 2014, Michael Croucher
        All rights reserved.

        Redistribution and use in source and binary forms, with or without
        modification, are permitted provided that the following conditions are met:

        * Redistributions of source code must retain the above copyright notice, this
          list of conditions and the following disclaimer.

        * Redistributions in binary form must reproduce the above copyright notice,
          this list of conditions and the following disclaimer in the documentation
          and/or other materials provided with the distribution.

        * Neither the name of nearest_correlation nor the names of its
          contributors may be used to endorse or promote products derived from
          this software without specific prior written permission.

        THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
        AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
        IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
        DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
        FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
        DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
        SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
        CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
        OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
        OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
        """

        # If input is an ExceededMaxIterationsError object this
        # is a restart computation
        if (isinstance(A, ExceededMaxIterationsError)):
            ds = np.copy(A.ds)
            A = np.copy(A.matrix)
        else:
            ds = np.zeros(np.shape(A))

        eps = np.spacing(1)
        if not np.all((np.transpose(A) == A)):
            raise ValueError('Input Matrix is not symmetric')
        if not tol:
            tol = eps * np.shape(A)[0] * np.array([1, 1])
        if weights is None:
            weights = np.ones(np.shape(A)[0])
        X = np.copy(A)
        Y = np.copy(A)
        rel_diffY = np.inf
        rel_diffX = np.inf
        rel_diffXY = np.inf

        Whalf = np.sqrt(np.outer(weights, weights))

        iteration = 0
        while max(rel_diffX, rel_diffY, rel_diffXY) > tol[0]:
            iteration += 1
            if iteration > max_iterations:
                if except_on_too_many_iterations:
                    if max_iterations == 1:
                        message = "No solution found in "\
                                  + str(max_iterations) + " iteration"
                    else:
                        message = "No solution found in "\
                                  + str(max_iterations) + " iterations"
                    raise ExceededMaxIterationsError(message, X, iteration, ds)
                else:
                    # exceptOnTooManyIterations is false so just silently
                    # return the result even though it has not converged
                    return X

            Xold = np.copy(X)
            R = X - ds
            R_wtd = Whalf * R
            if flag == 0:
                X = positive_semidefinite_matrix(R_wtd)
            elif flag == 1:
                raise NotImplementedError("Setting 'flag' to 1 is currently\
                                     not implemented.")
            X = X / Whalf
            ds = X - R
            Yold = np.copy(Y)
            Y = np.copy(X)
            np.fill_diagonal(Y, 1)
            normY = np.linalg.norm(Y, 'fro')
            rel_diffX = np.linalg.norm(X - Xold, 'fro') / np.linalg.norm(X, 'fro')
            rel_diffY = np.linalg.norm(Y - Yold, 'fro') / normY
            rel_diffXY = np.linalg.norm(Y - X, 'fro') / normY

            X = np.copy(Y)

        return X

    try:
        return nearcorr(A, max_iterations=max_iterations)
    except ExceededMaxIterationsError as e:
        raise matrix.errors.TooManyIterationsError(result=e.matrix, iterations=max_iterations) from e
