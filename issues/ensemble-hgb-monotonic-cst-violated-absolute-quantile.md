# HistGradientBoostingRegressor violates `monotonic_cst` with `loss="absolute_error"` and `loss="quantile"`

```json
{
    "estimators": ["HistGradientBoostingRegressor"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 3,
    "difficulty": 3,
    "decision": 1,
    "decision_tags": ["output-change"]
}
```

## Classification

- **Priority (3)**: silent violation of a documented guarantee. Monotonic quantile regression (pricing, risk, growth curves) is a natural use case, and users rely on the constraint without checking it. The violations are not numerical noise: up to ~0.5 on a target of scale ~1 in the reproducer.
- **Difficulty (3)**: the leaf values are recomputed after the tree is grown, outside of the bound propagation done by the grower. A correct fix needs to enforce the constraint on the recomputed values. A naive clipping of each updated leaf to the bounds stored by the grower (`children_lower_bound`/`children_upper_bound`) did not remove all violations in my quick test, so this needs some thought.
- **Decision (1)**: fitted models with these losses and `monotonic_cst` will change (`output-change`), which is the point. Raising an error for this combination would be a cheaper alternative but removes a useful feature.

## Upstream search

Core API dump of all open issues/PRs and of all issues labeled `module:ensemble`, grepped for monotonic, quantile, absolute, pinball, leaf values. `gh search issues` for "monotonic quantile HistGradientBoosting" (no result; "monotonic_cst absolute_error" failed on the shared search API rate limit). Related but different: #22173 (open, about the speed of the leaf values update), #27748 (closed, doc on monotonic constraints). No issue reports the violation.

## Describe the bug and its user-facing impact

The User Guide defines a monotonic increase constraint as `x_1 <= x_1' => F(x_1, x_2) <= F(x_1', x_2)`, and `monotonic_cst` is accepted with every loss of `HistGradientBoostingRegressor`. With the non-differentiable losses (`"absolute_error"` and `"quantile"`), the fitted model is not monotonic: predictions decrease along a feature constrained to be increasing.

It happens on simple data with noisy, non-monotonic signal (which is when constraints matter). `"squared_error"` respects the constraint on the same data.

## Steps/Code to Reproduce

```python
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

rng = np.random.RandomState(0)
X = rng.uniform(size=(1000, 1))
y = np.sin(8 * X[:, 0]) + rng.standard_t(df=1.5, size=1000)

X_grid = np.linspace(0, 1, 300).reshape(-1, 1)
for params in [
    dict(loss="squared_error"),
    dict(loss="absolute_error"),
    dict(loss="quantile", quantile=0.9),
]:
    est = HistGradientBoostingRegressor(monotonic_cst=[1], random_state=0, **params)
    pred = est.fit(X, y).predict(X_grid)
    print(f"{params}: min(diff(pred)) = {np.diff(pred).min():+.4f}")
```

## Expected Results

`min(diff(pred)) >= 0` for every loss, as for `"squared_error"`.

## Actual Results

```
{'loss': 'squared_error'}: min(diff(pred)) = +0.0000
{'loss': 'absolute_error'}: min(diff(pred)) = -0.0051
{'loss': 'quantile', 'quantile': 0.9}: min(diff(pred)) = -0.5155
```

With 2 features and other seeds I see violations between -0.004 and -1.1 for `"quantile"` and up to -0.14 for `"absolute_error"`.

## Root cause and possible fix

The grower enforces the constraint on the Newton leaf values: each split computes `value_left`/`value_right` within bounds, and the children of a constrained split are separated by `mid` (`sklearn/ensemble/_hist_gradient_boosting/grower.py:573-593` on the commit below).

For losses with `differentiable = False`, `fit` then overwrites every leaf value with `_update_leaves_values` (`sklearn/ensemble/_hist_gradient_boosting/gradient_boosting.py:68`, called at line 851): `leaf.value = grower.shrinkage * loss.fit_intercept_only(y_true - raw_prediction)` (line 105), i.e. the (weighted) median or quantile of the residuals in the leaf. This ignores the bounds of the leaf, so two sibling leaves of a constrained split can end up in the wrong order.

Possible fixes:

- after the leaf update, re-enforce the constraint on the tree, e.g. by propagating bounds top-down again with the new leaf values (a correct version of "clip each leaf to its bounds");
- or raise an error for `monotonic_cst` with non-differentiable losses (cheap but loses a feature that mostly works).

## Versions

```
sklearn: 1.10.dev0, commit 289c4d5cfa (upstream/main; run on a local editable build whose HGB code is identical to upstream/main except docstrings and the feature subsampling code in splitting.pyx, unused here)
python: 3.13, numpy: 2.5.3, scipy: 1.18.1, platform: linux
```
