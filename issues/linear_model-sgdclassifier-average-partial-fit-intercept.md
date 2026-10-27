# SGDClassifier(average=True).partial_fit loses the (non-averaged) intercept between calls

```json
{
    "estimators": ["SGDClassifier"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 2,
    "difficulty": 1,
    "decision": 0,
    "decision_tags": []
}
```

## Classification

- **Priority (2)**: silent wrong intercepts on a documented use (`partial_fit` for out-of-core learning, with ASGD via `average`). The model still looks reasonable, so it's hard to notice, but the intercept is biased towards 0 for binary problems and is a non-averaged, stale value for multiclass ones. `fit` is not affected.
- **Difficulty (1)**: store the returned standard intercept in `_standard_intercept` in both branches, as `BaseSGDRegressor._fit_regressor` and `SGDOneClassSVM._fit_one_class` already do.
- **Decision (0)**: results change only for `average` + several `partial_fit` calls, where they are currently wrong.

## Upstream search

Queries: "SGDClassifier average partial_fit", "SGDClassifier average partial_fit intercept", "average intercept partial_fit". Related but different: #5246 (closed), a crash on the second multiclass `partial_fit` with `average`, fixed long ago in the same code; #10011 (closed), `warm_start` vs one-go `fit` with SGD's learning rate schedule.

## Describe the bug and its user-facing impact

With `average` enabled, `SGDClassifier` keeps two sets of parameters: the plain SGD iterate (`_standard_coef`, `_standard_intercept`), from which the next `partial_fit` call resumes, and the averaged one (`_average_coef`, `_average_intercept`), exposed as `coef_`/`intercept_`. The coefficients are updated in place by the Cython solver, but the intercept is returned as a float and must be stored back by Python code. In `_fit_binary` and `_fit_multiclass` this only happens before averaging starts (`average > t_ - 1`). With `average=True`, averaging starts after the first sample, so:

- binary: `_standard_intercept` is never updated. Each `partial_fit` call restarts SGD from the initial intercept (0), and the averaged intercept is pulled towards 0;
- multiclass: the loop `self.intercept_[i] = intercept` writes the standard intercepts into `self.intercept_`, which from the second call on is the `_average_intercept` array. So the averaged intercepts are overwritten by the non-averaged ones, and `_standard_intercept` stays at the values of the first call.

As a result, `n` calls to `partial_fit` on the same data no longer match `fit(max_iter=n)` (they do match with `average=False`, and for `SGDRegressor` with `average=True`).

## Steps/Code to Reproduce

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import SGDClassifier

params = dict(loss="log_loss", learning_rate="constant", eta0=0.01, shuffle=False,
              tol=None, random_state=0)

# Imbalanced classes, so that the intercepts are far from 0.
for weights in [[0.9], [0.8, 0.1]]:
    X, y = make_classification(n_samples=500, n_classes=len(weights) + 1,
                               n_informative=3, weights=weights, flip_y=0,
                               random_state=0)
    for average in [False, True]:
        fit = SGDClassifier(max_iter=50, average=average, **params).fit(X, y)
        pfit = SGDClassifier(average=average, **params)
        for _ in range(50):
            pfit.partial_fit(X, y, classes=np.unique(y))
        print(f"{len(weights) + 1} classes, average={average}:\n"
              f"  intercept_ fit:              {fit.intercept_.round(3)}\n"
              f"  intercept_ 50 x partial_fit: {pfit.intercept_.round(3)}")
```

## Expected Results

With `shuffle=False` and a constant learning rate, 50 calls to `partial_fit` on the full data give the same model as `fit(max_iter=50)`, with or without `average`.

## Actual Results

```
2 classes, average=False:
  intercept_ fit:              [-3.373]
  intercept_ 50 x partial_fit: [-3.373]
2 classes, average=True:
  intercept_ fit:              [-2.899]
  intercept_ 50 x partial_fit: [-0.747]
3 classes, average=False:
  intercept_ fit:              [ 1.501 -3.845 -3.048]
  intercept_ 50 x partial_fit: [ 1.501 -3.845 -3.048]
3 classes, average=True:
  intercept_ fit:              [ 1.403 -3.261 -2.523]
  intercept_ 50 x partial_fit: [ 0.872 -1.601 -1.264]
```

## Root cause and possible fix

`sklearn/linear_model/_stochastic_gradient.py` (commit 289c4d5cfa):

- `_fit_binary`, lines 780-787: `self._standard_intercept = np.atleast_1d(intercept)` is only in the `else` branch (averaging not started yet).
- `_fit_multiclass`, lines 831-832: `self.intercept_[i] = intercept` writes into whatever `self.intercept_` is, which is `_average_intercept` once averaging has started (line 841); `_standard_intercept` is only refreshed in the `else` branch (line 844).

Fix: always store the standard intercepts, then pick the exposed ones, as in `_fit_regressor` (lines 1763-1766):

```python
# _fit_binary
if self.average > 0:
    self._standard_intercept = np.atleast_1d(intercept)
    if self.average <= self.t_ - 1:
        self.coef_ = self._average_coef.reshape(1, -1)
        self.intercept_ = self._average_intercept
    else:
        self.coef_ = self._standard_coef.reshape(1, -1)
        self.intercept_ = self._standard_intercept
```

and in `_fit_multiclass`, write the returned intercepts into `self._standard_intercept[i]` when `average > 0` (into `self.intercept_[i]` otherwise).

## Versions

```
sklearn: 1.10.dev0, commit 289c4d5cfa (upstream/main, pure-Python files overlaid on the local build: no compiled file differs in sklearn/linear_model)
python: 3.13.14, numpy: 2.5.3, scipy: 1.18.1, platform: linux
```
