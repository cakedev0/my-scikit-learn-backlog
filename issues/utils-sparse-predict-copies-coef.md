# Predicting on sparse input copies the whole coef_ (or components_) at each call

```json
{
    "estimators": ["LogisticRegression", "SGDClassifier", "LinearSVC", "RidgeClassifier", "Ridge", "MultinomialNB", "ComplementNB", "GaussianRandomProjection", "TruncatedSVD", "safe_sparse_dot"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 2,
    "difficulty": 2,
    "decision": 1,
    "decision_tags": ["float-order"]
}
```

## Classification

- **Priority (2)**: hits the main use case of linear models on sparse data (text classification with many classes): every `predict`/`decision_function`/`predict_proba` call on a sparse `X` allocates a full copy of `coef_`, even for a single row. With 200 classes and 50k features, predicting one sample allocates 76 MiB and takes ~30 ms instead of < 1 ms; with 1000 classes and 500k features it's 4 GB per call. Results are correct, and users can't easily work around it (densifying `X` is worse for large batches).
- **Difficulty (2)**: the fix fits in `safe_sparse_dot` (one branch), but it touches a hot path shared by many estimators and needs benchmarking on large batches, not only single rows.
- **Decision (1)**: a column-by-column product changes the floating-point summation layout (`float-order`), and for large batches with many classes it could be slower than scipy's single pass in some regimes (my quick check below shows it's faster, but it needs a proper benchmark). This is arguably a scipy limitation, so reporting it upstream is an alternative.

## Upstream search

Queries: "safe_sparse_dot coef_ copy sparse", "sparse predict copy coef_", "decision_function sparse slow", "decision_function sparse memory", "GaussianRandomProjection sparse memory", "safe_sparse_dot Fortran": no results. I did not search the scipy tracker.

## Describe the bug and its user-facing impact

Linear models compute `safe_sparse_dot(X, self.coef_.T)`. `coef_` is C-contiguous with shape `(n_classes, n_features)`, so `coef_.T` is F-contiguous. For a sparse `X`, this goes to scipy's `_matmul_multivector`, which calls `other.ravel()` before `csr_matvecs`: that is a C-order copy of the whole `(n_features, n_classes)` matrix, at every call, whatever the number of rows in `X`.

So with sparse inputs:

- each prediction call allocates a transient copy of the model's coefficients (76 MiB in the example below);
- online/small-batch prediction is ~10x slower than with the same data densified, and orders of magnitude slower than the O(nnz * n_classes) work it needs.

The binary case is not affected (`coef_` has one row, `coef_.T` is both C- and F-contiguous). The same pattern appears in:

- `LinearClassifierMixin.decision_function` / `LinearModel._decision_function` (`LogisticRegression`, `SGDClassifier`, `LinearSVC`, `RidgeClassifier`, `Perceptron`, multi-target `Ridge`, ...): `sklearn/linear_model/_base.py:390-391`;
- `MultinomialNB` and `ComplementNB` (`safe_sparse_dot(X, self.feature_log_prob_.T)`): `sklearn/naive_bayes.py:924` and `:1072`;
- `GaussianRandomProjection.transform` (`components_` is `(n_components, n_features)`, often hundreds of MB): `sklearn/random_projection.py:609`; with a 1000 x 50k `components_` (381 MiB), `transform` of a sparse `X` peaks at 389 MiB vs 8 MiB for a dense `X`;
- `TruncatedSVD.transform` / `PCA.transform` on sparse input (smaller `components_`, so less visible).

## Steps/Code to Reproduce

```python
import time
import tracemalloc

import numpy as np
import scipy.sparse as sp
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB

rng = np.random.RandomState(0)
n_classes, n_features = 200, 50_000
X = sp.random(4_000, n_features, density=0.001, format="csr", random_state=rng)
y = rng.randint(n_classes, size=X.shape[0])
x_one = X[:1]  # a single sparse sample with ~50 non-zeros

for clf in [SGDClassifier(max_iter=1, tol=None), MultinomialNB()]:
    clf.fit(X, y)
    for x in [x_one, x_one.toarray()]:
        tracemalloc.start()
        clf.predict(x)
        peak = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
        tic = time.perf_counter()
        for _ in range(20):
            clf.predict(x)
        duration = (time.perf_counter() - tic) / 20
        print(
            f"{type(clf).__name__:13s} predict on 1 {type(x).__name__:10s} row: "
            f"peak {peak / 2**20:5.1f} MiB, {duration * 1e3:5.1f} ms"
        )
```

(`coef_` and `feature_log_prob_` are 200 x 50,000 float64 = 76 MiB.)

## Expected Results

Predicting one sparse row should allocate O(n_classes) memory and take well under a millisecond.

## Actual Results

```
SGDClassifier predict on 1 csr_matrix row: peak  76.3 MiB,  30.4 ms
SGDClassifier predict on 1 ndarray    row: peak   0.0 MiB,   4.0 ms
MultinomialNB predict on 1 csr_matrix row: peak  76.3 MiB,  31.5 ms
MultinomialNB predict on 1 ndarray    row: peak   0.0 MiB,   3.8 ms
```

(Timings on this laptop with `OMP_NUM_THREADS=2`, `ConvergenceWarning` removed; the peaks are deterministic.)

## Root cause and possible fix

scipy's `_matmul_multivector` (`scipy/sparse/_compressed.py`, scipy 1.18.1) needs a C-contiguous dense operand and calls `other.ravel()`, which copies an F-contiguous `coef_.T`. Writing `(coef_ @ X.T).T` doesn't help: scipy's `__rmatmul__` transposes back to the same product.

Possible fix in `sklearn/utils/extmath.py` (`safe_sparse_dot`): when `a` is a 2D CSR/CSC matrix and `b` a 2D dense array that is F-contiguous but not C-contiguous, compute the product column by column, each column of `b` being contiguous so scipy's matvec doesn't copy:

```python
out = np.empty((a.shape[0], b.shape[1]), dtype=..., order="F")
for j in range(b.shape[1]):
    out[:, j] = a @ b[:, j]
```

Quick check with a 200 x 50k dense `W` and a CSR `X` with 0.1% non-zeros:

| | `X @ W.T` (current) | column by column |
|---|---|---|
| 1 row | 76.3 MiB, 33.7 ms | 0.0 MiB, 0.8 ms |
| 4000 rows | 82.4 MiB, 103 ms | 6.1 MiB, 65 ms |

with identical results (`np.allclose`). It also seems worth reporting to scipy, since any F-ordered dense right operand (e.g. any `.T`) pays a full copy.

## Versions

```
sklearn: 1.10.dev0, commit 289c4d5cfa (upstream/main; local checkout f570497251, no diff in sklearn/linear_model/_base.py, sklearn/naive_bayes.py, sklearn/random_projection.py, sklearn/utils/extmath.py)
python: 3.13.14, numpy: 2.5.3, scipy: 1.18.1, platform: linux
```
