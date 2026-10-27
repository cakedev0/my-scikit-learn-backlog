# `PolynomialFeatures` silently returns wrong products for CSR input with unsorted indices

```json
{
    "estimators": ["PolynomialFeatures"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 2,
    "difficulty": 1,
    "decision": 0,
    "decision_tags": []
}
```

## Classification

- **Priority (2)**: silent wrong values, but the trigger is rare: sparse input is already a minority of uses, and a CSR matrix with unsorted indices is an edge case within it (it takes e.g. column reordering with a fancy index, `X[:, [3, 0, 1]]`). Maintainer calibration, 2026-10-23.
- **Difficulty (1)**: sort the indices (on a copy) before calling the Cython expansion.
- **Decision (0)**: one extra copy only for inputs with unsorted indices.

## Upstream search

Queries (issues and PRs): "PolynomialFeatures unsorted indices", "PolynomialFeatures sparse csr wrong", "has_sorted_indices", "unsorted indices sparse". No match. #12197 (merged, "Fast PolynomialFeatures on CSR matrices") introduced the Cython expansion; its tests only use sorted CSR inputs.

## Describe the bug and its user-facing impact

For CSR input and `degree` 2 or 3, `PolynomialFeatures.transform` computes the output column of each product from the input column indices `i, j(, k)`, assuming that they come in increasing order inside a row. When a row stores its indices out of order, products land in the wrong output columns (and several can be summed into the same one). No error or warning is raised.

Unsorted indices are easy to get: scipy returns them when indexing columns with a list, so `make_column_transformer((PolynomialFeatures(), [3, 0, 1]))` on a CSR matrix gives wrong features. CSC input and `degree > 3` go through `tocsc()`/`tocsr()` conversions that sort indices, so they are not affected.

## Steps/Code to Reproduce

```python
import numpy as np
import scipy.sparse as sp
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import PolynomialFeatures

X = np.array([[1.0, 2.0, 3.0, 4.0]])
ct = make_column_transformer(
    (PolynomialFeatures(degree=2, include_bias=False), [3, 0, 1]),
    sparse_threshold=0,
)
print(ct.fit(X).get_feature_names_out())
print("dense input: ", ct.fit_transform(X))
print("sparse input:", ct.fit_transform(sp.csr_array(X)))
```

## Expected Results

The same output for the dense and sparse inputs: `[[ 4.  1.  2. 16.  4.  8.  1.  2.  4.]]`.

## Actual Results

```
['polynomialfeatures__x3' 'polynomialfeatures__x0'
 'polynomialfeatures__x1' 'polynomialfeatures__x3^2'
 'polynomialfeatures__x3 x0' 'polynomialfeatures__x3 x1'
 'polynomialfeatures__x0^2' 'polynomialfeatures__x0 x1'
 'polynomialfeatures__x1^2']
dense input:  [[ 4.  1.  2. 16.  4.  8.  1.  2.  4.]]
sparse input: [[ 4.  1.  2. 16.  0.  4.  9.  2.  4.]]
```

The same happens without `ColumnTransformer`, e.g. `PolynomialFeatures().fit_transform(X_csr[:, [2, 0, 1]])`.

## Root cause and possible fix

`sklearn/preprocessing/_csr_polynomial_expansion.pyx:224-250` loops over `j_ptr >= i_ptr` (and `k_ptr >= j_ptr`) and maps `(i, j)` / `(i, j, k)` to an output column with `_deg2_column` / `_deg3_column`, which are only correct for `i <= j <= k`. Nothing sorts the indices before: `PolynomialFeatures.transform` (`sklearn/preprocessing/_polynomial.py:432-452`) passes the validated CSR matrix straight to `_create_expansion`, and `check_array` does not sort indices.

Fix: in the CSR branch of `transform`, use `X = X.sorted_indices()` when `not X.has_sorted_indices` (or call `sort_indices()` on the copy when `validate_data` already copied). Add a test with unsorted indices for degrees 2 and 3, with and without `interaction_only`.

## Versions

```
sklearn: 1.10.dev0, commit f570497251 (_polynomial.py and _csr_polynomial_expansion.pyx identical to upstream/main 289c4d5cfa)
python: 3.13.14, numpy: 2.5.3, scipy: 1.18.1, platform: linux
```
