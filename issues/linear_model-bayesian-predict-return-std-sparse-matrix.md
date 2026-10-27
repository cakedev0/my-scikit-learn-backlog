# BayesianRidge/ARDRegression.predict(return_std=True) is wrong or fails on a scipy sparse matrix

```json
{
    "estimators": ["BayesianRidge", "ARDRegression"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 1,
    "difficulty": 1,
    "decision": 0,
    "decision_tags": []
}
```

## Classification

- **Priority (1)**: `fit` only accepts dense data, so predicting on sparse data is unusual, but `predict` documents `{array-like, sparse matrix}` and, when `n_rows == n_features`, silently returns wrong standard deviations with a wrong shape.
- **Difficulty (1)**: densify (or use `safe_sparse_dot` and an elementwise product that works for sparse input) at the start of the `return_std` branch.
- **Decision (0)**: no trade-off.

## Upstream search

Queries (issues and PRs): "BayesianRidge predict return_std sparse", "BayesianRidge return_std sparse", "ARDRegression predict sparse".

- #18605 (closed): ARDRegression `predict(return_std=True)` crashing with `normalize=True`. Different cause.
- The current code comes from #33918 (merged, released in 1.9.0), "FIX BayesianRidge and ARDRegression predict to center X before computing std", which added `X = X - self.X_offset_`; for a `csr_matrix` this subtraction is what returns an `np.matrix`. So this is a regression of 1.9 for sparse matrix input.

## Describe the bug and its user-facing impact

`predict(X, return_std=True)` computes `X = X - self.X_offset_` and then `(np.dot(X, self.sigma_) * X).sum(axis=1)`. With a `scipy.sparse.csr_matrix`, `X - X_offset_` returns a dense `np.matrix`, for which `*` is a matrix product, not an elementwise product. So:

- if `n_rows != n_features`: `ValueError: shapes (5,3) and (5,3) not aligned`;
- if `n_rows == n_features`: no error, `y_std` has shape `(n, 1)` instead of `(n,)` and wrong values.

The mean prediction (`return_std=False`) works with sparse input. Sparse arrays (`csr_array`) give the right result since `*` is elementwise for them.

## Steps/Code to Reproduce

```python
import numpy as np
import scipy.sparse as sp
from sklearn.linear_model import ARDRegression, BayesianRidge

rng = np.random.RandomState(0)
X = rng.randn(30, 3)
y = X @ [1.0, 2.0, 3.0] + rng.randn(30)

for reg in [BayesianRidge().fit(X, y), ARDRegression().fit(X, y)]:
    print(type(reg).__name__)
    print("dense: ", reg.predict(X[:3], return_std=True)[1])
    print("sparse:", reg.predict(sp.csr_matrix(X[:3]), return_std=True)[1].tolist())
    try:
        reg.predict(sp.csr_matrix(X[:5]), return_std=True)
    except ValueError as e:
        print("5 rows:", e)
```

## Expected Results

The same 1D `y_std` for dense and sparse input.

## Actual Results

```
BayesianRidge
dense:  [1.09726728 1.16630655 1.05277207]
sparse: [[1.1151464830896474], [1.2009481014434886], [1.0575802047115865]]
5 rows: shapes (5,3) and (5,3) not aligned: 3 (dim 1) != 5 (dim 0)
ARDRegression
dense:  [1.0967656  1.1651663  1.05249277]
sparse: [[1.1141025343670945], [1.1993401735122249], [1.0569504112913382]]
5 rows: shapes (5,3) and (5,3) not aligned: 3 (dim 1) != 5 (dim 0)
```

## Root cause and possible fix

`sklearn/linear_model/_bayes.py:391-392` (`BayesianRidge.predict`) and `:811-813` (`ARDRegression.predict`) on upstream/main:

```python
X = X - self.X_offset_
sigmas_squared_data = (np.dot(X, self.sigma_) * X).sum(axis=1)
```

`X` is not validated in this branch (`_decision_function` validates its own copy). Fix: `X = validate_data(self, X, reset=False)` (dense only, since `fit` is dense only anyway) or `np.asarray(X - self.X_offset_)` before the product.

## Versions

```
sklearn: 1.10.dev0, commit 289c4d5cfa (upstream/main; reproduced on a local build whose _bayes.py is identical)
python: 3.13, numpy: 2.5.3, scipy: 1.18.1, platform: linux
```
