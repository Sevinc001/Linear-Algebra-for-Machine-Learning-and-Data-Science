"""
eigen.py

Eigenvalues/eigenvectors from scratch, using the QR algorithm. This is
what PCA actually needs -- the principal components are just the
eigenvectors of the covariance matrix.

How it works: repeatedly QR-decompose A and multiply back as R @ Q. For
symmetric matrices this converges to a diagonal matrix (eigenvalues on
the diagonal), and multiplying all the Q's together gives the eigenvectors.
Not the most robust method numerically (real libraries use shifted
variants), but it's simple enough to actually understand and it works
fine for the covariance matrices PCA uses.
"""

import numpy as np
from matrix_operations import matrix_multiply, transpose, identity


def qr_decomposition(A: np.ndarray):
    """
    Gram-Schmidt QR decomposition: A = Q @ R, Q has orthonormal columns,
    R is upper triangular. Same as numpy.linalg.qr(A).
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
    QR algorithm for symmetric matrices. Returns eigenvalues (descending)
    and their eigenvectors as columns. Same idea as numpy.linalg.eigh.
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

    # Sort largest eigenvalue first -- matters for PCA, since the first
    # component should be the direction of maximum variance.
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    # Eigenvectors are only defined up to sign, so pin down a consistent
    # convention (largest entry positive) for reproducible output.
    for i in range(eigenvectors.shape[1]):
        col = eigenvectors[:, i]
        max_idx = np.argmax(np.abs(col))
        if col[max_idx] < 0:
            eigenvectors[:, i] = -col

    return eigenvalues, eigenvectors


def power_iteration(A: np.ndarray, n_iterations: int = 1000, tol: float = 1e-10):
    """
    Simpler method, only finds the single dominant eigenpair: repeatedly
    apply A to a vector and renormalize. Kept this in mostly to sanity
    check eigen_decomposition against a second, more intuitive method.
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
