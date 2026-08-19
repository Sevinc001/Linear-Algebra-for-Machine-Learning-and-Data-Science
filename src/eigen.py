"""
eigen.py
--------
From-scratch computation of eigenvalues and eigenvectors for symmetric
matrices, using the QR algorithm. This is the piece PCA relies on: PCA's
principal components are the eigenvectors of the data's covariance matrix.

Method
------
1. QR decomposition (via Gram-Schmidt): factor A = Q @ R
2. QR algorithm: repeatedly set A <- R @ Q. For symmetric matrices, this
   sequence converges to a diagonal matrix whose diagonal entries are the
   eigenvalues, while the accumulated product of Q matrices converges to
   the matrix of eigenvectors.

This method is simple to understand and works well for the symmetric,
positive semi-definite covariance matrices that PCA uses -- it is not the
most numerically robust general-purpose algorithm (production libraries
like LAPACK use more sophisticated shifted variants), but it is
transparent and sufficient for this project's purposes.
"""

import numpy as np
from matrix_operations import matrix_multiply, transpose, identity


def qr_decomposition(A: np.ndarray):
    """
    Decompose A = Q @ R using the Gram-Schmidt process, where Q has
    orthonormal columns and R is upper triangular.

    Equivalent to: numpy.linalg.qr(A)
    """
    A = np.asarray(A, dtype=float)
    n, m = A.shape
    Q = np.zeros((n, m))
    R = np.zeros((m, m))

    for j in range(m):
        v = A[:, j].copy()
        for i in range(j):
            R[i, j] = np.dot(Q[:, i], A[:, j])
            v = v - R[i, j] * Q[:, i]
        R[j, j] = np.linalg.norm(v)
        if np.isclose(R[j, j], 0):
            raise ValueError("Matrix columns are linearly dependent; QR decomposition failed")
        Q[:, j] = v / R[j, j]

    return Q, R


def eigen_decomposition(A: np.ndarray, n_iterations: int = 500, tol: float = 1e-10):
    """
    Compute eigenvalues and eigenvectors of a symmetric matrix A using the
    (unshifted) QR algorithm.

    Parameters
    ----------
    A : (n, n) symmetric matrix
    n_iterations : maximum number of QR iterations
    tol : convergence tolerance on off-diagonal elements

    Returns
    -------
    eigenvalues : (n,) array, sorted in descending order
    eigenvectors : (n, n) array, columns are the corresponding eigenvectors

    Equivalent to: numpy.linalg.eigh(A) (for symmetric matrices)
    """
    A = np.asarray(A, dtype=float)
    if not np.allclose(A, A.T, atol=1e-8):
        raise ValueError("eigen_decomposition currently supports symmetric matrices only")

    n = A.shape[0]
    A_k = A.copy()
    Q_total = identity(n)

    for _ in range(n_iterations):
        Q, R = qr_decomposition(A_k)
        A_k = matrix_multiply(R, Q)
        Q_total = matrix_multiply(Q_total, Q)

        off_diagonal = A_k - np.diag(np.diag(A_k))
        if np.max(np.abs(off_diagonal)) < tol:
            break

    eigenvalues = np.diag(A_k)
    eigenvectors = Q_total

    # Sort by eigenvalue, descending (largest variance first -- matters for PCA)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    # Normalize sign convention: make the largest-magnitude entry of each
    # eigenvector positive, for consistent, reproducible output.
    for i in range(eigenvectors.shape[1]):
        col = eigenvectors[:, i]
        max_idx = np.argmax(np.abs(col))
        if col[max_idx] < 0:
            eigenvectors[:, i] = -col

    return eigenvalues, eigenvectors


def power_iteration(A: np.ndarray, n_iterations: int = 1000, tol: float = 1e-10):
    """
    Find the dominant eigenvalue/eigenvector pair of A using power iteration.

    This is a simpler, more intuitive method than the QR algorithm, and is
    included mainly to build intuition: repeatedly applying A to a vector
    and renormalizing converges to the eigenvector of the largest-magnitude
    eigenvalue.

    Returns
    -------
    eigenvalue : float
    eigenvector : (n,) array, unit norm
    """
    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    rng = np.random.default_rng(0)
    v = rng.standard_normal(n)
    v = v / np.linalg.norm(v)

    eigenvalue = 0.0
    for _ in range(n_iterations):
        v_new = A @ v
        v_new_norm = np.linalg.norm(v_new)
        v_new = v_new / v_new_norm

        eigenvalue_new = v_new @ (A @ v_new)
        if abs(eigenvalue_new - eigenvalue) < tol:
            eigenvalue = eigenvalue_new
            v = v_new
            break
        eigenvalue = eigenvalue_new
        v = v_new

    return eigenvalue, v


if __name__ == "__main__":
    A = np.array([[4, 1], [1, 3]], dtype=float)

    values, vectors = eigen_decomposition(A)
    np_values, np_vectors = np.linalg.eigh(A)

    print("My eigenvalues:   ", values)
    print("NumPy eigenvalues:", np_values[::-1])

    print("\nMy eigenvectors:\n", vectors)
    print("\nNumPy eigenvectors (reversed columns):\n", np_vectors[:, ::-1])

    dom_value, dom_vector = power_iteration(A)
    print("\nDominant eigenvalue via power iteration:", dom_value)
    print("Dominant eigenvector via power iteration:", dom_vector)
