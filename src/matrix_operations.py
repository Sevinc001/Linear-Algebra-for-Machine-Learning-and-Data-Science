"""
matrix_operations.py
---------------------
From-scratch implementations of core matrix operations and linear
transformations, plus a general-purpose linear system solver (Gaussian
elimination with partial pivoting) used later by the eigenvalue module.
"""

import numpy as np


def matrix_multiply(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Multiply two matrices A (m x n) and B (n x p) -> result (m x p).
    Equivalent to: A @ B
    """
    A, B = np.asarray(A, dtype=float), np.asarray(B, dtype=float)
    m, n = A.shape
    n2, p = B.shape
    if n != n2:
        raise ValueError(f"Incompatible shapes for multiplication: {A.shape} and {B.shape}")

    result = np.zeros((m, p))
    for i in range(m):
        for j in range(p):
            total = 0.0
            for k in range(n):
                total += A[i, k] * B[k, j]
            result[i, j] = total
    return result


def transpose(A: np.ndarray) -> np.ndarray:
    """
    Transpose a matrix: swap rows and columns.
    Equivalent to: A.T
    """
    A = np.asarray(A, dtype=float)
    m, n = A.shape
    result = np.zeros((n, m))
    for i in range(m):
        for j in range(n):
            result[j, i] = A[i, j]
    return result


def identity(n: int) -> np.ndarray:
    """Build an n x n identity matrix. Equivalent to: np.eye(n)"""
    I = np.zeros((n, n))
    for i in range(n):
        I[i, i] = 1.0
    return I


def apply_linear_transformation(A: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Apply a linear transformation (matrix A) to a vector v.
    Equivalent to: A @ v
    """
    A, v = np.asarray(A, dtype=float), np.asarray(v, dtype=float)
    return matrix_multiply(A, v.reshape(-1, 1)).flatten()


def solve_linear_system(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Solve Ax = b using Gaussian elimination with partial pivoting.

    Parameters
    ----------
    A : (n, n) coefficient matrix (must be square, non-singular)
    b : (n,) right-hand side vector

    Returns
    -------
    (n,) solution vector x

    Equivalent to: np.linalg.solve(A, b)
    """
    A = np.array(A, dtype=float, copy=True)
    b = np.array(b, dtype=float, copy=True).reshape(-1, 1)

    n, m = A.shape
    if n != m:
        raise ValueError(f"Coefficient matrix must be square, got shape {A.shape}")

    M = np.hstack([A, b])

    for col in range(n):
        pivot_row = col + int(np.argmax(np.abs(M[col:, col])))
        if np.isclose(M[pivot_row, col], 0):
            raise ValueError("Matrix is singular; system has no unique solution")
        if pivot_row != col:
            M[[col, pivot_row]] = M[[pivot_row, col]]

        M[col] = M[col] / M[col, col]

        for row in range(col + 1, n):
            factor = M[row, col]
            M[row] = M[row] - factor * M[col]

    x = np.zeros(n)
    for row in reversed(range(n)):
        x[row] = M[row, -1] - np.dot(M[row, row + 1:n], x[row + 1:n])

    return x


def covariance_matrix(X: np.ndarray) -> np.ndarray:
    """
    Compute the covariance matrix of a dataset X (n_samples x n_features).
    Each feature is first mean-centered.

    Equivalent to: np.cov(X, rowvar=False)
    """
    X = np.asarray(X, dtype=float)
    n_samples = X.shape[0]
    X_centered = X - X.mean(axis=0)
    return matrix_multiply(transpose(X_centered), X_centered) / (n_samples - 1)


if __name__ == "__main__":
    A = np.array([[1, 2], [3, 4]])
    B = np.array([[5, 6], [7, 8]])

    print("A @ B =\n", matrix_multiply(A, B), "\nNumPy:\n", A @ B)
    print("\nA.T =\n", transpose(A), "\nNumPy:\n", A.T)

    v = np.array([1, 1])
    print("\nA @ v =", apply_linear_transformation(A, v), "| NumPy:", A @ v)

    b = np.array([1, 2])
    print("\nSolve Ax=b:", solve_linear_system(A, b), "| NumPy:", np.linalg.solve(A, b))
