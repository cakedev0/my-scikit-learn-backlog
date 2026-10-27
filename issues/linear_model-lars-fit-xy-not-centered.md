# Lars.fit(Xy=...) gives wrong coefficients with fit_intercept=True, and crashes for a 1D Xy

```json
{
    "estimators": ["Lars", "LassoLars"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 1,
    "difficulty": 1,
    "decision": 1,
    "decision_tags": ["output-change"]
}
```

## Classification

- **Priority (1)**: silent wrong coefficients when following the docstring, but the `Xy` fit parameter is rarely used (it only saves one matrix-vector product).
- **Difficulty (1)**: center the user's `Xy` (`Xy - X_offset * y_offset * n_samples`, i.e. `X_c.T @ y_c`) or drop it with a warning when `fit_intercept=True`, and reshape a 1D `Xy` to 2D. A few lines in `Lars.fit` / `Lars._fit`.
- **Decision (1)**: `output-change` for users who passed an already-centered `Xy` with `fit_intercept=True` (the only way to get a correct result today). Recomputing `Xy` from the centered data and ignoring the argument (with a warning, as `ElasticNet` does for a user Gram) avoids that ambiguity.

## Upstream search

Queries (issues and PRs): "Lars Xy", "Lars Xy fit_intercept", "Lars fit Xy".

- #27887 (closed, fixed by #28419): `lars_path_gram` documented a 2D `Xy` it did not support. Same parameter name, different function and problem.
- #17789 (closed): `lars_path` overwriting `X` with `copy_X=True`. Unrelated.

#14145 (open) is about `LassoLarsIC.fit` lacking the `Xy` parameter: related API, different problem.

## Describe the bug and its user-facing impact

`Lars.fit` (and `LassoLars.fit`, which inherits it) documents:

```
Xy : array-like of shape (n_features,) or (n_features, n_targets), default=None
    Xy = np.dot(X.T, y) that can be precomputed. ...
```

Two problems:

1. With the default `fit_intercept=True`, `X` and `y` are centered before running LARS (and the Gram matrix is computed from the centered `X`), but the user's `Xy` is passed to `lars_path` as is. `lars_path` uses it as the initial correlations whether or not a Gram matrix is used. Passing `Xy = np.dot(X.T, y)` as documented thus gives silently wrong coefficients whenever the features or the target are not centered.
2. A 1D `Xy` of shape `(n_features,)`, documented as valid, crashes with an `IndexError` because `_fit` always indexes `Xy[:, k]`.

## Steps/Code to Reproduce

```python
import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import Lars

X, y = make_regression(n_samples=60, n_features=5, noise=1, random_state=0)
X += 3  # non-centered features, as usual

print(np.round(Lars().fit(X, y).coef_, 2))
# Xy as documented: "Xy = np.dot(X.T, y)", of shape (n_features, n_targets)
print(np.round(Lars().fit(X, y, Xy=np.dot(X.T, y)[:, None]).coef_, 2))
# Xy of shape (n_features,), also documented
Lars().fit(X, y, Xy=np.dot(X.T, y))
```

## Expected Results

The same `coef_` with and without the precomputed `Xy`, and no error for a 1D `Xy`.

## Actual Results

```
[42.41 62.97 26.08  6.15 39.79]
[40.45 60.06 22.99  2.27 36.24]
Traceback (most recent call last):
  File ".../repro_lars_xy.py", line 12, in <module>
    Lars().fit(X, y, Xy=np.dot(X.T, y))
  File "/home/arthur/scikit-learn/sklearn/base.py", line 1403, in wrapper
    return fit_method(estimator, *args, **kwargs)
  File "/home/arthur/scikit-learn/sklearn/linear_model/_least_angle.py", line 1195, in fit
    self._fit(
        X,
    ...<4 lines>...
        Xy=Xy,
    )
  File "/home/arthur/scikit-learn/sklearn/linear_model/_least_angle.py", line 1099, in _fit
    this_Xy = None if Xy is None else Xy[:, k]
IndexError: too many indices for array: array is 1-dimensional, but 2 were indexed
```

Passing `Xy` computed from the centered data (`(X - X.mean(0)).T @ (y - y.mean())`) gives the right coefficients, which confirms the cause.

## Root cause and possible fix

`sklearn/linear_model/_least_angle.py:1076-1105` (`Lars._fit`): `_preprocess_data` centers `X` and `y`, then `Xy[:, k]` is passed unchanged to `lars_path`, where it becomes `Cov` (line 554). Nothing reshapes a 1D `Xy` either.

Possible fix: in `_fit`, `Xy = np.asarray(Xy).reshape(n_features, -1)`, and when `fit_intercept=True`, either correct it (`Xy - n_samples * np.outer(X_offset, y_offset)` gives `X_c.T @ y_c`) or ignore it with a warning like `_pre_fit` does for a user-provided Gram matrix in coordinate descent ("Gram matrix was provided but X was centered to fit intercept: recomputing Gram matrix."). The docstring should also say that `Xy` must be computed on centered data, or that it is ignored when `fit_intercept=True`.

## Versions

```
sklearn: 1.10.dev0, commit 289c4d5cfa (upstream/main; reproduced on a local build whose _least_angle.py is identical)
python: 3.13, numpy: 2.5.3, scipy: 1.18.1, platform: linux
```
