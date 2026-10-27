# `AdaBoostRegressor.staged_predict` calls `predict` on every base estimator again at each stage

```json
{
    "estimators": ["AdaBoostRegressor"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 1,
    "difficulty": 1,
    "decision": 0,
    "decision_tags": []
}
```

## Classification

- **Priority (1)**: `staged_predict` and `staged_score` are monitoring tools that few people call on `AdaBoostRegressor`, but they make `n_estimators * (n_estimators + 1) / 2` base-estimator `predict` calls (1275 for the default 50), ~3x slower than needed. Results are correct.
- **Difficulty (1)**: predict once per base estimator before the loop and slice the cached array, a few lines.
- **Decision (0)**: identical output. Memory goes up to one `(n_samples, n_estimators)` float array, which `predict` already allocates.

## Upstream search

- "AdaBoostRegressor staged_predict", "staged_predict quadratic", "AdaBoost staged slow" (GitHub issue search): no results.
- No local duplicate (`grep -ril staged_predict issues/`).

## Describe the bug and its user-facing impact

`AdaBoostRegressor.staged_predict` yields `self._get_median_predict(X, limit=i)` for `i = 1..n_estimators`, and `_get_median_predict` calls `est.predict(X)` on the first `limit` estimators every time. So stage `i` re-predicts with all `i` previous estimators, and the whole generator makes O(n_estimators²) base `predict` calls, each with its own input validation and tree traversal.

`AdaBoostClassifier.staged_predict` / `staged_decision_function`, by contrast, add one estimator's contribution per stage. `AdaBoostRegressor.staged_score` (inherited from `BaseWeightBoosting`, `sklearn/ensemble/_weight_boosting.py:269`) goes through `staged_predict`, so it has the same cost.

Caching the base predictions gives the same stages ~3x faster in the reproducer below. The weighted median itself is still recomputed from scratch at each stage (an `argsort` over the first `i` columns), so the total stays quadratic in `n_estimators`, but with a much smaller constant.

## Steps/Code to Reproduce

```python
from time import perf_counter

import numpy as np
from sklearn.datasets import make_regression
from sklearn.ensemble import AdaBoostRegressor


def staged_predict_cached(ada, X):
    # Same weighted median as AdaBoostRegressor, but each base estimator
    # predicts X only once.
    predictions = np.array([est.predict(X) for est in ada.estimators_]).T
    rows = np.arange(X.shape[0])
    for limit in range(1, len(ada.estimators_) + 1):
        p = predictions[:, :limit]
        sorted_idx = np.argsort(p, axis=1)
        weight_cdf = np.cumsum(ada.estimator_weights_[sorted_idx], axis=1)
        median_idx = (weight_cdf >= 0.5 * weight_cdf[:, -1:]).argmax(axis=1)
        yield p[rows, sorted_idx[rows, median_idx]]


X, y = make_regression(n_samples=10_000, n_features=10, noise=10, random_state=0)
for n_estimators in [25, 50, 100]:
    ada = AdaBoostRegressor(n_estimators=n_estimators, random_state=0).fit(X, y)
    t0 = perf_counter()
    staged = list(ada.staged_predict(X))
    t1 = perf_counter()
    cached = list(staged_predict_cached(ada, X))
    t2 = perf_counter()
    same = all(np.array_equal(a, b) for a, b in zip(staged, cached))
    print(f"n_estimators={len(ada.estimators_)}: staged_predict {t1 - t0:.2f}s, "
          f"with cached base predictions {t2 - t1:.2f}s, identical: {same}")
```

## Expected Results

Each base estimator predicts `X` once, as in `AdaBoostClassifier`'s staged methods.

## Actual Results

Timings from a laptop loaded by other jobs:

```
n_estimators=25: staged_predict 0.26s, with cached base predictions 0.09s, identical: True
n_estimators=50: staged_predict 0.95s, with cached base predictions 0.28s, identical: True
n_estimators=100: staged_predict 3.66s, with cached base predictions 1.16s, identical: True
```

## Root cause and possible fix

`sklearn/ensemble/_weight_boosting.py` (on 289c4d5cfa):

- line 1061, in `_get_median_predict`: `predictions = np.array([est.predict(X) for est in self.estimators_[:limit]]).T`;
- line 1122, in `staged_predict`: `yield self._get_median_predict(X, limit=i)` inside the loop over stages.

Fix: split `_get_median_predict` into the base predictions and the weighted median of a given predictions array, compute the predictions once in `staged_predict`, and pass `predictions[:, :i]` at each stage, as `staged_predict_cached` does above.

## Versions

```
sklearn: 1.10.dev0, commit 289c4d5cfa (upstream/main; checkout at f570497251, no diff in sklearn/ensemble/_weight_boosting.py)
python: 3.13, numpy: 2.5.3, scipy: 1.18.1, platform: linux
```
