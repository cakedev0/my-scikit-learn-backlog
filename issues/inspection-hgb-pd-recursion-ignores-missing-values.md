# `partial_dependence` with `method="recursion"` sends missing values to the wrong child in HistGradientBoosting

```json
{
    "estimators": ["partial_dependence", "PartialDependenceDisplay", "HistGradientBoostingRegressor", "HistGradientBoostingClassifier"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 2,
    "difficulty": 1,
    "decision": 0,
    "decision_tags": []
}
```

## Classification

- **Priority (2)**: silent wrong value at the NaN grid point, with the default `method="auto"` on HGB. It needs a NaN in the grid, which happens for low-cardinality features with missing values (fewer unique values than `grid_resolution`) or with NaN in `custom_values`.
- **Difficulty (1)**: a few lines in the Cython tree walker, mirroring what `_predict_one_from_raw_data` already does.
- **Decision (0)**: brings `'recursion'` in line with `'brute'` and with `predict`.

## Upstream search

`gh issue list --search` (GraphQL, the REST search was rate limited): "partial dependence nan grid", "partial_dependence missing_go_to_left", "partial dependence missing values HistGradientBoosting", "partial dependence recursion missing values".

- #25401 (closed) and #25647 (closed): crashes with NaN in the grid, fixed by allowing NaN in the grid. They don't cover how `'recursion'` routes that NaN.
- #27441 (open): `'recursion'` computes a conditional PD for trees. Different issue.
- #34554 (open PR): `'recursion'` ignores categorical splits in HGB and trees, and the PR makes it raise for categorical target features. Same function, but it does not touch the missing-value routing reported here.

## Describe the bug and its user-facing impact

When the partial dependence grid contains NaN (missing values are kept in the grid by `np.unique` for features with fewer than `grid_resolution` unique values, or passed through `custom_values`), `method="recursion"` on `HistGradientBoosting*` always sends the NaN grid point to the right child of the nodes that split on the target feature. It ignores the `missing_go_to_left` flag learned during `fit`, so the partial dependence at the "missing" grid point is the one of another value. `method="auto"` picks `'recursion'` for HGB, so this is what users get by default, including in `PartialDependenceDisplay.from_estimator`.

`DecisionTreeRegressor` and `RandomForestRegressor` handle this correctly: their recursion walker in `sklearn/tree/_tree.pyx` checks `missing_go_to_left`.

## Steps/Code to Reproduce

```python
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import partial_dependence

rng = np.random.RandomState(0)
x0 = rng.randint(0, 3, 1000).astype(float)
x0[rng.rand(1000) < 0.3] = np.nan
X = np.c_[x0, rng.randn(1000)]
y = 10 * np.nan_to_num(x0) + X[:, 1]  # missing values behave like x0 == 0

hgb = HistGradientBoostingRegressor(random_state=0).fit(X, y)
brute = partial_dependence(hgb, X, features=[0], method="brute")
recursion = partial_dependence(hgb, X, features=[0], method="recursion")
print("grid:              ", brute["grid_values"][0])
print("brute:             ", brute["average"].round(2))
print("recursion:         ", recursion["average"].round(2))
# 'recursion' ignores the baseline prediction, so the difference with 'brute'
# should be a constant offset
print("recursion - brute: ", (recursion["average"] - brute["average"]).round(2))
```

## Expected Results

As documented in `partial_dependence`, `'recursion'` gives the same values as `'brute'` up to a constant offset (the baseline prediction). The NaN grid point should give ~0 with `'brute'` and ~-7.2 with `'recursion'`.

## Actual Results

```
grid:               [ 0.  1.  2. nan]
brute:              [[-2.000e-02  9.980e+00  1.998e+01 -1.000e-02]]
recursion:          [[-7.23  2.75 12.67 12.67]]
recursion - brute:  [[-7.22 -7.23 -7.3  12.69]]
```

The NaN grid point gets the partial dependence of `x0 == 2` instead of the one of `x0 == 0`.

## Root cause and possible fix

`sklearn/ensemble/_hist_gradient_boosting/_predictor.pyx:224` (`_compute_partial_dependence`) routes the target feature with:

```python
if X[sample_idx, feature_idx] <= current_node.num_threshold:
    node_idx_stack[stack_size] = current_node.left
else:
    node_idx_stack[stack_size] = current_node.right
```

`NaN <= threshold` is always false, so NaN goes right. The prediction walker of the same file (lines 58 and 75) first checks `isnan` and follows `node.missing_go_to_left`. The fix is to do the same in `_compute_partial_dependence`. The categorical branch is missing too, which #34554 handles separately by raising for categorical target features.

## Versions

```
sklearn: 1.10.dev0, commit f570497251 (local branch; _predictor.pyx and
_partial_dependence.py are identical on upstream/main 289c4d5cfa, the local
branch only differs in HGB's splitting.pyx)
python: 3.13, numpy: 2.5.3, scipy: 1.18.1, platform: linux
```

See also: `ensemble-hgb-categorical-column-reorder-interaction-cst-pdp.md` (same recursion code path, wrong feature when a categorical feature is not first).
