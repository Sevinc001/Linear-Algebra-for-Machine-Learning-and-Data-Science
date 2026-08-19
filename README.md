# Linear Algebra for Machine Learning

A project I built to actually understand the linear algebra behind
machine learning, instead of just calling `sklearn.decomposition.PCA()`
without knowing what's happening inside it.

Every core operation -- vector arithmetic, matrix multiplication, solving
linear systems, eigen-decomposition -- is implemented from scratch in
NumPy and checked against NumPy's/scikit-learn's built-in equivalents.
Everything builds up to implementing **PCA from scratch** and applying it
to a real dataset.

## Background

This project follows a linear algebra for machine learning course I
completed on Coursera: [certificate](https://www.coursera.org/account/accomplishments/certificate/AEOJYIKG11T8).
The code here is my own, written independently after finishing the
course, applying the concepts to a new project rather than the course's
own assignments.

## Topics Covered

- Vectors -- addition, dot product, norms, distance, cosine similarity
- Matrices -- multiplication, transpose, identity
- Linear transformations -- geometric intuition, systems of linear equations
- Eigenvalues and eigenvectors -- QR algorithm, power iteration
- Covariance matrix
- PCA -- dimensionality reduction from first principles

## Machine Learning Application

- PCA from scratch, applied to the Iris dataset
- Dimensionality reduction (4 features -> 2 principal components)
- Side-by-side comparison with scikit-learn's `PCA` -- explained variance
  ratio and projected coordinates match

## Project StructureLinear-Algebra-for-Machine-Learning/
│
├── notebooks/
│ ├── 01_vectors.ipynb
│ ├── 02_matrices.ipynb
│ ├── 03_linear_transformations.ipynb
│ ├── 04_eigenvalues_eigenvectors.ipynb
│ └── 05_pca_from_scratch.ipynb
│
├── src/
│ ├── vector_operations.py
│ ├── matrix_operations.py
│ ├── eigen.py
│ └── pca.py
│
├── data/
├── README.md
└── requirements.txt

Each notebook imports directly from `src/`, so the same code backs both
the notebooks and any standalone scripts.

## How the pieces fit together

PCA isn't a separate black box here -- it's built directly on top of the
earlier modules. That's the point of the project: understanding *why* PCA
works, not just calling a library function.

## Running it

```bash
git clone <this-repo>
cd Linear-Algebra-for-Machine-Learning
pip install -r requirements.txt

python src/vector_operations.py
python src/matrix_operations.py
python src/eigen.py
python src/pca.py

jupyter notebook notebooks/
```

## Example result: PCA on the Iris dataset

Reduces the Iris dataset's 4 features to 2 principal components, keeping
about 97.8% of the total variance, and matches scikit-learn's `PCA`
exactly (aside from an arbitrary sign flip per component -- expected,
since eigenvectors are only defined up to sign).

| | Explained variance ratio (PC1, PC2) |
|---|---|
| My PCA | `[0.9246, 0.0531]` |
| scikit-learn PCA | `[0.9246, 0.0531]` |

Full walkthrough and plots: `notebooks/05_pca_from_scratch.ipynb`

## Tools

- Python
- NumPy
- Matplotlib
- scikit-learn (used only for the Iris dataset and for the final
  comparison -- not part of the implementation itself)

## License

MIT -- see [LICENSE](LICENSE).
