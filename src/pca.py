"""
pca.py

PCA built entirely on top of the other modules here -- covariance_matrix
from matrix_operations.py, eigen_decomposition from eigen.py. Wrote it
with a fit/transform interface similar to sklearn's so it's easy to
compare the two directly.

Steps: center the data -> covariance matrix -> eigenvectors of that
matrix (sorted by eigenvalue) -> project data onto the top-k of them.
"""

import numpy as np
from matrix_operations import covariance_matrix, matrix_multiply
from eigen import eigen_decomposition


class PCA:
    def __init__(self, n_components: int):
        self.n_components = n_components
        self.mean_ = None
        self.components_ = None          # (n_components, n_features)
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None

    def fit(self, X: np.ndarray) -> "PCA":
        X = np.asarray(X, dtype=float)
        self.mean_ = X.mean(axis=0)

        cov = covariance_matrix(X)
        eigenvalues, eigenvectors = eigen_decomposition(cov)

        self.components_ = eigenvectors[:, : self.n_components].T
        self.explained_variance_ = eigenvalues[: self.n_components]
        self.explained_variance_ratio_ = eigenvalues[: self.n_components] / np.sum(eigenvalues)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.components_ is None:
            raise RuntimeError("PCA instance is not fitted yet. Call fit() first.")
        X = np.asarray(X, dtype=float)
        X_centered = X - self.mean_
        return matrix_multiply(X_centered, self.components_.T)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)


if __name__ == "__main__":
    from sklearn.datasets import load_iris
    from sklearn.decomposition import PCA as SklearnPCA

    X = load_iris().data

    my_pca = PCA(n_components=2)
    my_result = my_pca.fit_transform(X)

    sk_pca = SklearnPCA(n_components=2)
    sk_result = sk_pca.fit_transform(X)

    print("My explained variance ratio:    ", my_pca.explained_variance_ratio_)
    print("Sklearn explained variance ratio:", sk_pca.explained_variance_ratio_)

    # Sign of each component is arbitrary between implementations,
    # so compare magnitudes rather than raw values.
    print("\nFirst 5 rows, my PCA (abs):\n", np.abs(my_result[:5]))
    print("\nFirst 5 rows, sklearn PCA (abs):\n", np.abs(sk_result[:5]))
