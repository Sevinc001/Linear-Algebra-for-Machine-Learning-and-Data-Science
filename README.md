# Linear Algebra for Machine Learning

A practical project exploring the linear algebra foundations of machine
learning through implementations from scratch and a PCA application.

Every core operation — vector arithmetic, matrix multiplication, solving
linear systems, eigen-decomposition — is implemented from scratch in
NumPy and checked against NumPy's/scikit-learn's built-in equivalents.
The project builds up to implementing **Principal Component Analysis**
from scratch and applying it to a real dataset.

## Topics Covered

- Vectors — addition, dot product, norms, distance, cosine similarity
- Matrices — multiplication, transpose, identity
- Linear transformations — geometric intuition, systems of linear equations
- Eigenvalues and eigenvectors — QR algorithm, power iteration
- Matrix operations — covariance matrix
- PCA — dimensionality reduction from first principles

## Machine Learning Application

- **PCA from scratch**, applied to the Iris dataset
- Dimensionality reduction (4 features → 2 principal components)
- Side-by-side **comparison with scikit-learn's `PCA`** — explained
  variance ratio and projected coordinates match

## Project Structure

```
Linear-Algebra-for-Machine-Learning/
│
├── notebooks/
│   ├── 01_vectors.ipynb
│   ├── 02_matrices.ipynb
│   ├── 03_linear_transformations.ipynb
│   ├── 04_eigenvalues_eigenvectors.ipynb
│   └── 05_pca_from_scratch.ipynb
│
├── src/
│   ├── vector_operations.py
│   ├── matrix_operations.py
│   ├── eigen.py
│   └── pca.py
│
├── data/                  # (empty — notebooks use scikit-learn's built-in Iris dataset)
├── README.md
└── requirements.txt
```

Each notebook imports directly from `src/`, so the same tested code
backs both the notebooks and any further scripts.

## How the pieces fit together

```
vector_operations.py  →  building blocks (dot product, norms)
        ↓
matrix_operations.py  →  matrix multiply, covariance matrix, linear solver
        ↓
eigen.py               →  eigenvalues/eigenvectors of the covariance matrix
        ↓
pca.py                 →  project data onto top-k eigenvectors = PCA
```

PCA isn't implemented as an isolated black box — it's built directly on
top of the vector and matrix modules earlier in the project, which is the
main point: showing *why* PCA works, not just calling a library function.

## Running it

```bash
git clone <this-repo>
cd Linear-Algebra-for-Machine-Learning
pip install -r requirements.txt

# Run any module directly for a quick sanity check against NumPy/sklearn
python src/vector_operations.py
python src/matrix_operations.py
python src/eigen.py
python src/pca.py

# Or open the notebooks
jupyter notebook notebooks/
```

## Example result: PCA on the Iris dataset

The from-scratch PCA implementation reduces the Iris dataset's 4 features
to 2 principal components, retaining **~97.8%** of the total variance,
and matches scikit-learn's `PCA` exactly (up to an arbitrary sign flip
per component — expected, since eigenvectors are only defined up to sign).

| | Explained variance ratio (PC1, PC2) |
|---|---|
| My PCA | `[0.9246, 0.0531]` |
| scikit-learn PCA | `[0.9246, 0.0531]` |

See `notebooks/05_pca_from_scratch.ipynb` for the full walkthrough and
plots.

## Tools

- Python
- NumPy
- Matplotlib
- scikit-learn (used only for the Iris dataset and for the final
  comparison — never for the implementations themselves)

## Notes

This project was built independently while studying linear algebra
fundamentals. All code is original; scikit-learn is used strictly as a
correctness benchmark, not as part of the implementation.

## License

MIT — see [LICENSE](LICENSE).
