"""
matrix_operations.py

Matrix multiply, transpose, and a Gaussian elimination solver, all written
by hand instead of using numpy directly. The solver here is basically the
same one I wrote for a linear algebra course assignment, just cleaned up
and made more general (partial pivoting instead of just diagonal pivots).
"""

import numpy as np


def matrix_multiply(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Standard matrix multiply, triple nested loop. Same as A @ B."""
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
    """Swap rows/cols. Same as A.T."""
    A = np.asarray(A, dtype=float)
    m, n = A.shape
    result = np.zeros((n, m))
    for i in range(m):
        for j in range(n):
            result[j, i] = A[i, j]
    return result


def identity(n: int) -> np.ndarray:
    """n x n identity matrix. Same as np.eye(n)."""
    I = np.zeros((n, n))
    for i in range(n):
        I[i, i] = 1.0
    return I


def apply_linear_transformation(A: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Apply matrix A to vector v. Same as A @ v."""
    A, v = np.asarray(A, dtype=float), np.asarray(v, dtype=float)
    return matrix_multiply(A, v.reshape(-1, 1)).flatten()


def solve_linear_system(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Solves Ax = b with Gaussian elimination. Added partial pivoting here
    (picks the row with the biggest value in each column before eliminating)
    since just using the diagonal can blow up numerically on some matrices.
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
    """Centers each feature then computes X^T X / (n-1). Feeds into PCA later."""
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
