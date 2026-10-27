# Forests with `categorical_features` warn "X does not have valid feature names" on every fit with a DataFrame

```json
{
    "estimators": ["RandomForestClassifier", "RandomForestRegressor", "ExtraTreesClassifier", "ExtraTreesRegressor"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 2,
    "difficulty": 1,
    "decision": 0,
    "decision_tags": []
}
```

## Classification

- **Priority (2)**: the warning fires on the main use case of the new feature (a DataFrame with categorical columns, `categorical_features` given by name or `"from_dtype"`), and it tells users that their input is wrong when it is not. Results are correct. The feature is not released yet (1.10.dev0).
- **Difficulty (1)**: replace the second `validate_data(..., reset=False)` call by `check_array`/`check_X_y` plus `_check_n_features`, as `DecisionTreeClassifier`/`DecisionTreeRegressor` already do.
- **Decision (0)**: no trade-off.

## Upstream search

Queries: `"does not have valid feature names" RandomForest categorical`, `feature names warning forest categorical_features`, `gh pr list --search "categorical forest feature names warning"`. Only #34832 (merged, adds the feature) comes up. No match.

## Describe the bug and its user-facing impact

`BaseForest.fit` first calls `validate_data(..., reset=True, skip_check_array=True)` on the DataFrame to store `feature_names_in_`, then encodes the categorical columns into a NumPy array, then calls `validate_data(X_encoded, y, reset=False)`. With `reset=False`, `validate_data` checks the feature names of the encoded array against `feature_names_in_` and warns because the array has none.

Every `fit` of a forest on a DataFrame with categorical features emits the warning. Users running with `-W error` (or in test suites turning `UserWarning` into errors) get a failure. The single-tree estimators don't warn, since they use `check_array` + `_check_n_features` for this second step.

## Steps/Code to Reproduce

```python
import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

rng = np.random.RandomState(0)
X = pd.DataFrame({"color": rng.choice(["a", "b", "c"], 100), "x": rng.randn(100)})
y = rng.randn(100)

for est in [DecisionTreeRegressor(categorical_features=["color"]),
            RandomForestRegressor(n_estimators=3, categorical_features=["color"])]:
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        est.fit(X, y)
    print(type(est).__name__, [str(w.message) for w in record])
```

## Expected Results

No warning for either estimator: `fit` and `predict` receive the same DataFrame.

## Actual Results

```
DecisionTreeRegressor []
RandomForestRegressor ['X does not have valid feature names, but RandomForestRegressor was fitted with feature names']
```

## Root cause and possible fix

`sklearn/ensemble/_forest.py:373-381`:

```python
X, y = validate_data(
    self, X, y, multi_output=True, dtype=np.float32,
    ensure_all_finite=False, reset=False,
)
```

`X` is the encoded ndarray at this point, and `validate_data(reset=False)` runs `_check_feature_names`. The comment above it explains why `reset=True` can't be used, but `reset=False` has the opposite problem. Use `check_X_y(X, y, multi_output=True, dtype=np.float32, ensure_all_finite=False)` followed by `_check_n_features(self, X, reset=False)`, like `BaseDecisionTree._fit` does (`sklearn/tree/_classes.py:302-311`). `predict` is not affected (`_validate_X_predict` already uses `check_array`).

## Versions

```
sklearn: 1.10.dev0, commit 289c4d5cfa
python: 3.13, numpy: 2.5.3, scipy: 1.18.1, platform: linux
```
