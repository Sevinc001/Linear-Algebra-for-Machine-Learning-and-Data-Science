"""
vector_operations.py

Basic vector ops written from scratch instead of just calling numpy,
so I actually understand what's happening under the hood (dot product,
norms, distance, cosine similarity). Checked against numpy in the notebook.
"""

import numpy as np


def vector_add(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Element-wise add. Same as u + v, just written out manually."""
    u, v = np.asarray(u, dtype=float), np.asarray(v, dtype=float)
    _check_same_shape(u, v)
    return np.array([u[i] + v[i] for i in range(len(u))])


def vector_scale(u: np.ndarray, scalar: float) -> np.ndarray:
    """Scalar multiply, element by element. Same as scalar * u."""
    u = np.asarray(u, dtype=float)
    return np.array([scalar * x for x in u])


def dot_product(u: np.ndarray, v: np.ndarray) -> float:
    """sum(u_i * v_i) -- the building block for norms/distance below."""
    u, v = np.asarray(u, dtype=float), np.asarray(v, dtype=float)
    _check_same_shape(u, v)
    total = 0.0
    for i in range(len(u)):
        total += u[i] * v[i]
    return total


def l2_norm(u: np.ndarray) -> float:
    """Vector length: sqrt(u . u). Built on dot_product on purpose."""
    return np.sqrt(dot_product(u, u))


def l1_norm(u: np.ndarray) -> float:
    """Sum of absolute values. Less sensitive to outliers than l2_norm."""
    u = np.asarray(u, dtype=float)
    return float(sum(abs(x) for x in u))


def euclidean_distance(u: np.ndarray, v: np.ndarray) -> float:
    """Straight-line distance between two points: ||u - v||_2."""
    u, v = np.asarray(u, dtype=float), np.asarray(v, dtype=float)
    _check_same_shape(u, v)
    diff = np.array([u[i] - v[i] for i in range(len(u))])
    return l2_norm(diff)


def cosine_similarity(u: np.ndarray, v: np.ndarray) -> float:
    """
    Angle-based similarity: (u . v) / (||u|| * ||v||). Ignores magnitude,
    only cares about direction -- useful for comparing embeddings.
    """
    denom = l2_norm(u) * l2_norm(v)
    if np.isclose(denom, 0):
        raise ValueError("Cannot compute cosine similarity with a zero vector")
    return dot_product(u, v) / denom


def _check_same_shape(u: np.ndarray, v: np.ndarray) -> None:
    if u.shape != v.shape:
        raise ValueError(f"Vectors must have the same shape, got {u.shape} and {v.shape}")


if __name__ == "__main__":
    u = np.array([1, 2, 3])
    v = np.array([4, 5, 6])

    print("u + v        =", vector_add(u, v), "| NumPy:", u + v)
    print("2 * u         =", vector_scale(u, 2), "| NumPy:", 2 * u)
    print("u . v         =", dot_product(u, v), "| NumPy:", np.dot(u, v))
    print("||u||_2       =", l2_norm(u), "| NumPy:", np.linalg.norm(u))
    print("||u||_1       =", l1_norm(u), "| NumPy:", np.linalg.norm(u, ord=1))
    print("dist(u, v)    =", euclidean_distance(u, v), "| NumPy:", np.linalg.norm(u - v))
    print("cos_sim(u, v) =", cosine_similarity(u, v))
