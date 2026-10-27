# `partial_dependence` (default `method="auto"`) is wrong for categorical target features of `DecisionTreeRegressor` and `RandomForestRegressor`

```json
{
    "estimators": ["partial_dependence", "PartialDependenceDisplay", "DecisionTreeRegressor", "RandomForestRegressor"],
    "existing_issue": "https://github.com/scikit-learn/scikit-learn/pull/34554",
    "reproduced": true,
    "priority": 2,
    "difficulty": 1,
    "decision": 1,
    "decision_tags": ["new-error"]
}
```

## Classification

- **Priority (2)**: silent wrong partial dependence with default arguments (`method="auto"` picks `"recursion"` for these two estimators) whenever the raw categories are not already `0..n_categories-1`, e.g. integer codes like 10/20/30. String categories crash instead. Categorical support is not released yet (1.10.dev0), and an approved fix is open.
- **Difficulty (1)**: route categorical target features to `"brute"` in `"auto"` and raise for `"recursion"`, as #34554 does. Supporting recursion properly needs the grid to go through the fitted encoder.
- **Decision (1)**: `new-error`: explicit `method="recursion"` on a categorical target feature raises instead of returning (wrong) values.

## Upstream search

Queries: `partial_dependence recursion categorical`.

- #34554 (open PR, two approvals, "Waiting for Second Reviewer"): "FIX raise NotImplementedError for recursion PDP with categorical target features". It describes exactly this bug for `DecisionTreeRegressor`, `RandomForestRegressor` and `HistGradientBoosting*`, and makes `"auto"` fall back to `"brute"`. There is no separate issue, hence the PR link in `existing_issue`.
- #35162 (closed): refactor of the shared categorical encoding, does not change the recursion path.

## Describe the bug and its user-facing impact

Trees and forests ordinal-encode categorical columns before fitting, so the bitsets in `tree_` refer to encoded categories (0, 1, 2, ...). `predict` re-encodes `X`, but `_compute_partial_dependence_recursion` passes the raw grid values straight to `tree_.compute_partial_dependence`. A raw value is then cast to `uint8` and looked up in the bitset as if it were an encoded category.

With numeric categories that are not `0..n-1`, the partial dependence is silently wrong: below, the true effect is "only category 20 gives 1", but recursion returns 1 for every category. With string categories (pandas `category` dtype), it fails with `ValueError: could not convert string to float: 'a'`. Only the target features matter; categorical features that are not plotted are averaged over and are fine.

## Steps/Code to Reproduce

```python
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import partial_dependence
from sklearn.tree import DecisionTreeRegressor

rng = np.random.RandomState(0)
X = rng.choice([10.0, 20.0, 30.0], size=(60, 1))
y = (X[:, 0] == 20.0).astype(float)
for est in [DecisionTreeRegressor(categorical_features=[0], random_state=0),
            RandomForestRegressor(n_estimators=5, categorical_features=[0], random_state=0)]:
    est.fit(X, y)
    auto = partial_dependence(est, X, [0], categorical_features=[0])  # method="auto"
    brute = partial_dependence(est, X, [0], categorical_features=[0], method="brute")
    print(type(est).__name__, auto["grid_values"][0], auto["average"][0], brute["average"][0])
```

## Expected Results

`method="auto"` gives the same values as `method="brute"`: `[0. 1. 0.]`.

## Actual Results

```
DecisionTreeRegressor [10. 20. 30.] [1. 1. 1.] [0. 1. 0.]
RandomForestRegressor [10. 20. 30.] [1. 1. 1.] [0. 1. 0.]
```

## Root cause and possible fix

`BaseDecisionTree._compute_partial_dependence_recursion` (`sklearn/tree/_classes.py:1655-1681`) and `ForestRegressor._compute_partial_dependence_recursion` (`sklearn/ensemble/_forest.py:1222-1254`) don't apply `_preprocess_X` to the grid. Encoding the grid is not trivial, since the grid only holds the target columns while the encoder expects all columns. #34554 makes `"auto"` use `"brute"` and `"recursion"` raise `NotImplementedError` for categorical target features (in `sklearn/inspection/_partial_dependence.py`). Encoding the target columns with `_categorical_encoder.categories_` would keep recursion available, at the cost of more code.

## Versions

```
sklearn: 1.10.dev0, commit 289c4d5cfa
python: 3.13, numpy: 2.5.3, scipy: 1.18.1, platform: linux
```
