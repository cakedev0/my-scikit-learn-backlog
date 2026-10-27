# QuantileRegressor fails on `coo_matrix` input when `sample_weight` contains zeros

```json
{
    "estimators": ["QuantileRegressor"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 1,
    "difficulty": 1,
    "decision": 0,
    "decision_tags": []
}
```

## Classification

- **Priority (1)**: crash on valid input (COO is in the accepted formats, zero weights are valid), but it needs both a COO matrix and a zero weight; converting to CSR is a trivial workaround.
- **Difficulty (1)**: drop `"coo"` from `accept_sparse` (it is converted to CSC/CSR by `validate_data` then) or convert before indexing.
- **Decision (0)**: no trade-off; HiGHS works on CSC internally anyway.

## Upstream search

- `gh search issues ... QuantileRegressor coo sample_weight`: only #21086 (merged, adds sparse support) and #20132.
- `gh search issues ... coo_matrix not subscriptable`, `safe_indexing coo`: no related issue.
- Local: `issues/decomposition-incremental-pca-transform-coo-dia-bsr.md` is the same kind of COO-indexing failure in another estimator, different code.

## Describe the bug and its user-facing impact

`QuantileRegressor.fit` accepts `["csc", "csr", "coo"]`. When some sample weights are zero, it removes those rows with `_safe_indexing(X, indices)` to help the LP solver. `coo_matrix` does not support row indexing, so the fit fails with `TypeError: 'coo_matrix' object is not subscriptable`. The same data fits fine as CSR, or as COO without zero weights.

On scipy 1.18.1, `coo_array` supports indexing and works; I haven't checked older scipy versions (the minimum is 1.11.4), where `coo_array` likely fails the same way.

## Steps/Code to Reproduce

```python
import numpy as np
from scipy import sparse
from sklearn.linear_model import QuantileRegressor

rng = np.random.RandomState(0)
X = rng.randn(20, 2)
y = X @ [1.0, 2.0] + rng.randn(20)
sample_weight = np.ones(20)
sample_weight[0] = 0

reg = QuantileRegressor(alpha=0.0)
print(reg.fit(sparse.csr_matrix(X), y, sample_weight=sample_weight).coef_)
print(reg.fit(sparse.coo_matrix(X), y).coef_)
reg.fit(sparse.coo_matrix(X), y, sample_weight=sample_weight)
```

## Expected Results

The COO fit with zero weights succeeds, like the CSR one.

## Actual Results

```
[0.83563366 1.88919908]
[0.83563366 1.88919908]
Traceback (most recent call last):
  File ".../q3.py", line 14, in <module>
    reg.fit(sparse.coo_matrix(X), y, sample_weight=sample_weight)
    ~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/arthur/scikit-learn/sklearn/base.py", line 1403, in wrapper
    return fit_method(estimator, *args, **kwargs)
  File "/home/arthur/scikit-learn/sklearn/linear_model/_quantile.py", line 194, in fit
    X = _safe_indexing(X, indices)
  File "/home/arthur/scikit-learn/sklearn/utils/_indexing.py", line 314, in _safe_indexing
    return _array_indexing(X, indices, indices_dtype, axis=axis)
  File "/home/arthur/scikit-learn/sklearn/utils/_indexing.py", line 62, in _array_indexing
    return array[key, ...] if axis == 0 else array[:, key]
           ~~~~~^^^^^^^^^^
  File "/home/arthur/scikit-learn/sklearn-env/lib/python3.13/site-packages/scipy/sparse/_coo.py", line 1945, in __getitem__
    raise TypeError("'coo_matrix' object is not subscriptable")
TypeError: 'coo_matrix' object is not subscriptable
```

## Root cause and possible fix

`sklearn/linear_model/_quantile.py:146` accepts COO, and `sklearn/linear_model/_quantile.py:194` row-indexes `X`. Since the HiGHS branch converts everything to CSC with `sparse.hstack(..., format="csc")` anyway, the simplest fix is `accept_sparse=["csc", "csr"]` so that `validate_data` converts COO upfront.

## Versions

```
sklearn: 1.10.dev0, commit 289c4d5cfa
python: 3.13, numpy: 2.5.3, scipy: 1.18.1, platform: linux
```
