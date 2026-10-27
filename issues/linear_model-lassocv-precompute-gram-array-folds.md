# LassoCV/ElasticNetCV(precompute=gram_array) reuse the full-data Gram matrix in every CV fold

```json
{
    "estimators": ["LassoCV", "ElasticNetCV"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 1,
    "difficulty": 1,
    "decision": 1,
    "decision_tags": ["new-warning"]
}
```

## Classification

- **Priority (1)**: the documented option "The Gram matrix can also be passed as an argument" cannot work in a CV estimator. It fails loudly (a `ValueError` blaming the user's Gram matrix, or one warning per fold), so results are not silently wrong, but the error is misleading.
- **Difficulty (1)**: do what `LarsCV` already does (warn and switch to `"auto"` for the CV folds), and possibly keep the array for the final refit.
- **Decision (1)**: `new-warning`: users passing a Gram array would get a warning instead of an error (`fit_intercept=False`) or instead of one warning per fold (`fit_intercept=True`). An alternative is to reject arrays in the parameter validation and fix the docstring, which is an API change.

## Upstream search

Queries (issues and PRs): "LassoCV precompute Gram", "precomputed Gram matrix cross-validation LassoCV", "LassoCV precompute", "LassoCV Gram matrix".

- #18919 (closed by its author): "Not expected result in LassoCV with fit_intercept=False and precomputed gram matrix". At that time the full-data Gram was silently used in the folds and gave a different result; the reporter closed it after realizing LassoCV fits on split data. No maintainer decision; since then the Gram validation turned this into the misleading `ValueError` below.
- #29565 (open): float32 Gram matrices failing `_check_precomputed_gram_matrix` in LassoCV. Related (same check), different cause (tolerance); a maintainer comment there notes that with `fit_intercept=True` the user Gram is ignored.
- #11014 / #11021 (closed): LassoCV setting `precompute=False` for the final refit. Different.

## Describe the bug and its user-facing impact

`LassoCV` and `ElasticNetCV` document `precompute : 'auto', bool or array-like of shape (n_features, n_features)` ("The Gram matrix can also be passed as an argument."). But the parameter is forwarded as is to the path computed on each training fold (`path_params = self.get_params()`, `sklearn/linear_model/_coordinate_descent.py:1873`, then `_path_residuals`), where the Gram matrix of the full data is not the Gram matrix of `X_train`:

- with `fit_intercept=False`, `_pre_fit` checks one element of the matrix against `X_train` and raises `ValueError: Gram matrix passed in via 'precompute' parameter did not pass validation ...`, which tells the user their (correct) Gram matrix is wrong;
- with `fit_intercept=True` (default), each fold emits `UserWarning: Gram matrix was provided but X was centered to fit intercept: recomputing Gram matrix.` and ignores it.

So a Gram array can never be used by these estimators. `LarsCV`/`LassoLarsCV` handle the same situation explicitly: `Parameter "precompute" cannot be an array in LarsCV. Automatically switch to "auto" instead.` (`sklearn/linear_model/_least_angle.py:1725-1731`).

## Steps/Code to Reproduce

```python
import warnings
from sklearn.datasets import make_regression
from sklearn.linear_model import LassoCV

X, y = make_regression(n_samples=60, n_features=5, noise=1, random_state=0)
gram = X.T @ X

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    LassoCV(precompute=gram, cv=3).fit(X, y)
print(len(w), "warnings:", w[0].message)

LassoCV(precompute=gram, cv=3, fit_intercept=False).fit(X, y)
```

## Expected Results

Either a single clear message that a Gram array cannot be used for the CV folds (as in `LarsCV`), or an error at parameter validation if arrays are not supported. The valid `gram` should not be reported as "not computed properly".

## Actual Results

```
4 warnings: Gram matrix was provided but X was centered to fit intercept: recomputing Gram matrix.
Traceback (most recent call last):
  File ".../repro_lassocv_gram.py", line 13, in <module>
    LassoCV(precompute=gram, cv=3, fit_intercept=False).fit(X, y)
  File "/home/arthur/scikit-learn/sklearn/linear_model/_coordinate_descent.py", line 2297, in fit
    return super().fit(X, y, sample_weight=sample_weight, **params)
  ...
  File "/home/arthur/scikit-learn/sklearn/linear_model/_base.py", line 893, in _pre_fit
    _check_precomputed_gram_matrix(X, precompute, X_offset, X_scale)
  File "/home/arthur/scikit-learn/sklearn/linear_model/_base.py", line 815, in _check_precomputed_gram_matrix
    raise ValueError(
ValueError: Gram matrix passed in via 'precompute' parameter did not pass validation when a single element was checked - please check that it was computed properly. For element (2,3) we computed 5.942605446740555 but the user-supplied value was 4.97369188054802.
```

(3 warnings from the 3 folds, 1 from the final refit.)

## Root cause and possible fix

`LinearModelCV.fit` passes `self.precompute` unchanged in `path_params` to `_path_residuals` (`sklearn/linear_model/_coordinate_descent.py:1873` and `:1946`), which calls `_pre_fit(X_train, ..., precompute, ...)` (`sklearn/linear_model/_base.py:875-893`).

Possible fix: in `LinearModelCV.fit`, if `precompute` is an array, warn once and use `"auto"` for the folds, like `LarsCV.fit` does. The array is still valid for the final refit on the whole data when `fit_intercept=False`, so it could be kept there.

## Versions

```
sklearn: 1.10.dev0, commit 289c4d5cfa (upstream/main; reproduced on a local build whose _coordinate_descent.py only differs in docstrings)
python: 3.13, numpy: 2.5.3, scipy: 1.18.1, platform: linux
```
