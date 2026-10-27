# `OneHotEncoder.inverse_transform` on its own sparse output is ~35x slower than on the dense array

```json
{
    "estimators": ["OneHotEncoder"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 1,
    "difficulty": 2,
    "decision": 1,
    "decision_tags": ["complexity"]
}
```

## Classification

- **Priority (1)**: performance only, results are correct: `inverse_transform` of a one-hot matrix is rarely called on large data, ~3.5 µs per sample and feature only hurts at hundreds of thousands of rows, and densifying first is a workaround. (priority review, 2026-10-23: was 2)
- **Difficulty (2)**: the slow part is SciPy's `argmax(axis=1)`, so the fix needs a vectorized row argmax on the CSR block. `_inverse_binarize_multiclass` in `sklearn/preprocessing/_label.py` already has one, which could be factored out and reused.
- **Decision (1)**: no behavior change, but it adds a small helper instead of relying on SciPy (`complexity`). An upstream fix in SciPy would also work, but would only help users with a future SciPy version.

## Upstream search

- "OneHotEncoder inverse_transform" (54 results, first page reviewed), "OneHotEncoder inverse_transform sparse", "sparse argmax slow" (GitHub issue search, scikit-learn and SciPy).
- #34549, #14934 (closed): correctness of `inverse_transform` with `drop`/`handle_unknown`, unrelated to speed.
- #34392 (merged) "PRF Faster encoders thanks to F-order layout": speeds up `transform`, not `inverse_transform`.
- No scikit-learn or SciPy issue about the speed of sparse `argmax(axis=1)` found.

## Describe the bug and its user-facing impact

`OneHotEncoder().fit_transform(X)` returns a CSR matrix by default. Calling `inverse_transform` on it loops over the input features and calls `sub.argmax(axis=1)` on each column block. In SciPy (checked on 1.18.1), sparse `argmax` along an axis is a Python loop over the non-empty rows (`scipy/sparse/_data.py`, `_argminmax_axis`), so the cost is ~3.5 µs per row and per input feature.

On 40,000 samples and 20 features, `inverse_transform` takes ~2.7 s, against ~0.09 s on the densified matrix and ~0.02 s for `transform`. On 1M rows and 20 features it would take more than a minute. It also shows up in `ColumnTransformer`/`Pipeline.inverse_transform` and in any code that maps predictions in one-hot space back to categories.

## Steps/Code to Reproduce

```python
from time import perf_counter

import numpy as np
from sklearn.preprocessing import OneHotEncoder

rng = np.random.RandomState(0)
for n_samples in [10_000, 20_000, 40_000]:
    X = rng.randint(5, size=(n_samples, 20))
    enc = OneHotEncoder().fit(X)
    X_sparse = enc.transform(X)  # default sparse_output=True
    t0 = perf_counter()
    enc.transform(X)
    t1 = perf_counter()
    X_inv = enc.inverse_transform(X_sparse)
    t2 = perf_counter()
    X_inv_dense = enc.inverse_transform(X_sparse.toarray())
    t3 = perf_counter()
    assert np.array_equal(X_inv, X) and np.array_equal(X_inv_dense, X)
    print(f"n_samples={n_samples}: transform {t1 - t0:.3f}s, inverse_transform "
          f"sparse {t2 - t1:.3f}s, same matrix densified {t3 - t2:.3f}s")
```

## Expected Results

`inverse_transform` on the sparse output should cost about as much as on the dense array, i.e. be linear in the number of stored values with a numpy-level constant.

## Actual Results

Timings from a laptop loaded by other jobs:

```
n_samples=10000: transform 0.008s, inverse_transform sparse 0.681s, same matrix densified 0.020s
n_samples=20000: transform 0.011s, inverse_transform sparse 1.371s, same matrix densified 0.037s
n_samples=40000: transform 0.021s, inverse_transform sparse 2.716s, same matrix densified 0.088s
```

cProfile with 5,000 samples and 20 features: 0.50 s of 0.51 s in `scipy/sparse/_data.py:_argminmax_axis` (20 calls), with 100,000 calls to `numpy.argmax`, one per row and per feature.

## Root cause and possible fix

`sklearn/preprocessing/_encoders.py:1155-1157` (on 289c4d5cfa):

```python
sub = X[:, j : j + n_categories]
# for sparse X argmax returns 2D matrix, ensure 1D array
labels = np.asarray(sub.argmax(axis=1)).flatten()
```

Possible fix: for sparse `X`, convert to CSR once and compute the row argmax of each block with numpy operations on `data`/`indices`/`indptr`. `_inverse_binarize_multiclass` (`sklearn/preprocessing/_label.py:702-734`) already does this for `LabelBinarizer` (row max with `min_max_axis`, then `np.repeat` + `searchsorted` to find the first maximum), so it could become a shared private helper. Converting `X` to CSC once would also make the per-feature column slices cheaper. Reporting the slow axis `argmax` to SciPy is worth doing too, but the scikit-learn fix doesn't need to wait for it.

## Versions

```
sklearn: 1.10.dev0, commit 289c4d5cfa (upstream/main; checkout at f570497251, no diff in sklearn/preprocessing/_encoders.py)
python: 3.13, numpy: 2.5.3, scipy: 1.18.1, platform: linux
```
