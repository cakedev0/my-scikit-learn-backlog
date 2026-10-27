# HistGradientBoosting refitted without `sample_weight` still refuses recursion partial dependence

```json
{
    "estimators": ["HistGradientBoostingClassifier", "HistGradientBoostingRegressor", "partial_dependence"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 1,
    "difficulty": 1,
    "decision": 0,
    "decision_tags": []
}
```

## Classification

- **Priority (1)**: loud error, only after refitting the same instance without weights (meta-estimators clone, so they don't hit it), with an easy workaround (`method="brute"` or a fresh estimator).
- **Difficulty (1)**: reset the flag at the start of `fit`.
- **Decision (0)**: no trade-off.

## Upstream search

Core API dump of all open issues/PRs and of issues labeled `module:ensemble`/`module:inspection`, grepped for partial dependence and sample weight (the full-text `gh search issues` query failed on the shared search API rate limit, so this relies on titles only). Related: #25210 (open) asks for recursion PD support with `sample_weight`, a feature request, not this stale state. No match.

## Describe the bug and its user-facing impact

`fit` sets `self._fitted_with_sw = True` when `sample_weight` is passed, and `_compute_partial_dependence_recursion` raises `NotImplementedError` when the flag is set. The flag is never reset, so an estimator fitted once with weights and then refitted without them still refuses `method="recursion"` (and `PartialDependenceDisplay.from_estimator(..., method="recursion")`), although the current model was fitted without weights. With `method="auto"`, `partial_dependence` picks `"recursion"` and the user gets the error too.

## Steps/Code to Reproduce

```python
import numpy as np
from sklearn.datasets import make_regression
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import partial_dependence

X, y = make_regression(n_samples=200, n_features=3, random_state=0)
est = HistGradientBoostingRegressor(max_iter=5)
est.fit(X, y, sample_weight=np.ones_like(y))
est.fit(X, y)  # refit without sample_weight
partial_dependence(est, X, features=[0], method="recursion")
```

## Expected Results

The second `fit` is independent of the first one (no `warm_start`), so the recursion PD works, as for a fresh estimator fitted without weights.

## Actual Results

```
NotImplementedError: HistGradientBoostingRegressor does not support partial dependence plots with the 'recursion' method when sample weights were given during fit time.
```

## Root cause and possible fix

`sklearn/ensemble/_hist_gradient_boosting/gradient_boosting.py` on the commit below: `fit` sets `self._fitted_with_sw = True` inside `if sample_weight is not None:` (line 459, with a "TODO: remove when PDP supports sample weights") and never sets it to `False`. `_compute_partial_dependence_recursion` checks `getattr(self, "_fitted_with_sw", False)` (line 1314). Fix: `self._fitted_with_sw = sample_weight is not None` before the check. With `warm_start=True`, the flag should stay set if any of the fits used weights, so `self._fitted_with_sw = (warm-starting and previous flag) or sample_weight is not None` is the exact version.

## Versions

```
sklearn: 1.10.dev0, commit 289c4d5cfa (upstream/main; run on a local editable build whose gradient_boosting.py differs from upstream/main only in docstrings)
python: 3.13, numpy: 2.5.3, scipy: 1.18.1, platform: linux
```
