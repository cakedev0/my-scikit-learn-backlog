# `GaussianMixture(max_iter=0, precisions_init=...)` fails with `AttributeError: ... no attribute 'covariances_'`

```json
{
    "estimators": ["GaussianMixture"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 1,
    "difficulty": 1,
    "decision": 0,
    "decision_tags": []
}
```

## Classification

- **Priority (1)**: crash on a supported combination (`max_iter=0` is allowed, tested by `test_max_iter_zero` and commented as "the user is assumed to have used 0-iters initialization"), and the natural way to build a fixed-parameter GMM. Loud error, and the workaround (set the fitted attributes by hand) is known but undocumented.
- **Difficulty (1)**: set `covariances_` from `precisions_init` in `_initialize`.
- **Decision (0)**: plain fix.

## Upstream search

Queries: "GaussianMixture max_iter 0", "GaussianMixture precisions_init covariances_", and `gh issue list --search` "GaussianMixture covariances_ AttributeError", "mixture max_iter=0" (the last two don't search comments). Related: #26415 (closed, wrong `precisions_cholesky_` from `precisions_init`, fixed by `_compute_precision_cholesky_from_precisions`), #9343 (closed PR, 2017, about `precisions_cholesky_` not being set). None covers `covariances_` with `max_iter=0`.

## Describe the bug and its user-facing impact

When `precisions_init` is given, `GaussianMixture._initialize` sets `precisions_cholesky_` but never `covariances_` (it is only set in the `precisions_init is None` branch). With `max_iter > 0`, the first M-step sets it, so nobody notices. With `max_iter=0`, `fit` calls `_get_parameters()` right after the initialization, which reads `self.covariances_` and raises. This happens whether or not `weights_init` and `means_init` are also given, and for every `covariance_type`.

`max_iter=0` with user-provided parameters is how users build a GMM with known parameters (to call `score_samples`, `predict_proba` or `sample`), so this fails exactly in the case where all initial parameters are given.

## Steps/Code to Reproduce

```python
import numpy as np
from sklearn.mixture import GaussianMixture

X = np.random.RandomState(0).randn(100, 2)
gm = GaussianMixture(
    n_components=2,
    covariance_type="spherical",
    weights_init=[0.5, 0.5],
    means_init=[[0, 0], [1, 1]],
    precisions_init=[1.0, 1.0],
    max_iter=0,
)
gm.fit(X)
```

## Expected Results

`fit` returns a model with `weights_`, `means_`, `precisions_` from the init parameters and `covariances_` equal to their inverses, as for `max_iter=0` without `precisions_init`.

## Actual Results

```
  File "/home/arthur/scikit-learn/sklearn/mixture/_base.py", line 199, in fit
    self.fit_predict(X, y)
    ~~~~~~~~~~~~~~~~^^^^^^
  File "/home/arthur/scikit-learn/sklearn/base.py", line 1403, in wrapper
    return fit_method(estimator, *args, **kwargs)
  File "/home/arthur/scikit-learn/sklearn/mixture/_base.py", line 259, in fit_predict
    best_params = self._get_parameters()
  File "/home/arthur/scikit-learn/sklearn/mixture/_gaussian_mixture.py", line 920, in _get_parameters
    self.covariances_,
    ^^^^^^^^^^^^^^^^^
AttributeError: 'GaussianMixture' object has no attribute 'covariances_'. Did you mean: 'covariance_type'?
```

## Root cause and possible fix

`sklearn/mixture/_gaussian_mixture.py:850-883` (`_initialize`): the `else` branch (`precisions_init is not None`) only computes `precisions_cholesky_`. Also set `self.covariances_` to the inverse of `precisions_init` (`1 / precisions_init` for diag/spherical, a matrix inverse, e.g. via the Cholesky factor, for full/tied).

## Versions

```
sklearn: 1.10.dev0, commit 289c4d5cfa
python: 3.13.14, numpy: 2.5.3, scipy: 1.18.1, platform: linux
```
