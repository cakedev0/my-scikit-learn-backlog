# `BayesianGaussianMixture(covariance_type="tied")`: `lower_bound_` is a 1-element array, and `verbose=2` crashes

```json
{
    "estimators": ["BayesianGaussianMixture"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 1,
    "difficulty": 1,
    "decision": 0,
    "decision_tags": []
}
```

## Classification

- **Priority (1)**: the model itself is correct. The fitted `lower_bound_` / `lower_bounds_` have the wrong type (array of shape `(1,)` instead of float), and `verbose=2` raises a `TypeError` with the NumPy 2.5 used here (converting ndim > 0 arrays to scalars was deprecated in NumPy 1.25). Only for `covariance_type="tied"`.
- **Difficulty (1)**: reduce `_log_wishart_norm(...)` to a scalar in the tied branch.
- **Decision (0)**: plain fix.

## Upstream search

Queries: "BayesianGaussianMixture tied verbose", "BayesianGaussianMixture lower_bound_", and `gh issue list --search` "BayesianGaussianMixture verbose TypeError", "only 0-dimensional arrays can be converted mixture". Related: #28559 (merged, added `lower_bounds_`), #10870 (merged, `lower_bound_` with `n_init > 1`). Nothing about the tied case.

## Describe the bug and its user-facing impact

For `covariance_type="tied"`, `BayesianGaussianMixture._compute_lower_bound` returns an ndarray of shape `(1,)` instead of a scalar. As a result:

- `lower_bound_` (documented as float) is `array([...])` and `lower_bounds_` is a list of 1-element arrays, so e.g. `np.diff(bgmm.lower_bounds_)` returns an empty array (it diffs along the last axis of shape `(n_iter, 1)`);
- `verbose=2` formats the change of lower bound with `%.5f` and fails with `TypeError: only 0-dimensional arrays can be converted to Python scalars` on NumPy 2.5.3 (older NumPy versions only emit a `DeprecationWarning` for this conversion).

## Steps/Code to Reproduce

```python
import numpy as np
from sklearn.mixture import BayesianGaussianMixture

X = np.random.RandomState(0).randn(100, 2)
bgmm = BayesianGaussianMixture(n_components=2, covariance_type="tied", random_state=0)
bgmm.fit(X)
print(repr(bgmm.lower_bound_))
print(repr(bgmm.lower_bounds_[:2]))

bgmm.set_params(verbose=2).fit(X)
```

## Expected Results

`lower_bound_` is a float, `lower_bounds_` a list of floats (as for the other covariance types and for `GaussianMixture`), and `verbose=2` prints the iterations.

## Actual Results

```
array([-121.33806867])
[array([-122.90758859]), array([-122.055957])]
Initialization 0
Traceback (most recent call last):
  File "r_bgmm_tied.py", line 10, in <module>
    bgmm.set_params(verbose=2).fit(X)
  File "/home/arthur/scikit-learn/sklearn/mixture/_base.py", line 199, in fit
    self.fit_predict(X, y)
  File "/home/arthur/scikit-learn/sklearn/base.py", line 1403, in wrapper
    return fit_method(estimator, *args, **kwargs)
  File "/home/arthur/scikit-learn/sklearn/mixture/_base.py", line 272, in fit_predict
    self._print_verbose_msg_iter_end(n_iter, change)
  File "/home/arthur/scikit-learn/sklearn/mixture/_base.py", line 599, in _print_verbose_msg_iter_end
    "  Iteration %d\t time lapse %.5fs\t ll change %.5f"
    % (n_iter, cur_time - self._iter_prev_time, diff_ll)
TypeError: only 0-dimensional arrays can be converted to Python scalars
```

## Root cause and possible fix

`sklearn/mixture/_bayesian_mixture.py:800`:

```python
log_wishart = self.n_components * np.float64(
    _log_wishart_norm(self.degrees_of_freedom_, log_det_precisions_chol, n_features)
)
```

In the tied case `degrees_of_freedom_` and `log_det_precisions_chol` are scalars, but `_log_wishart_norm` sums `gammaln(...)` of shape `(n_features, 1)` over axis 0, so it returns shape `(1,)`. `np.float64(array([x]))` returns the array unchanged, not a scalar. Use `np.sum(...)` (or `[0]`) instead of `np.float64(...)`.

## Versions

```
sklearn: 1.10.dev0, commit 289c4d5cfa
python: 3.13.14, numpy: 2.5.3, scipy: 1.18.1, platform: linux
```
