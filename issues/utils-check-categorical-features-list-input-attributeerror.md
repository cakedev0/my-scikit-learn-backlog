# `categorical_features` with a list of lists as `X` raises `AttributeError: 'list' object has no attribute 'shape'`

```json
{
    "estimators": ["DecisionTreeClassifier", "DecisionTreeRegressor", "RandomForestClassifier", "ExtraTreesRegressor", "HistGradientBoostingClassifier", "HistGradientBoostingRegressor"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 1,
    "difficulty": 1,
    "decision": 0,
    "decision_tags": []
}
```

## Classification

- **Priority (1)**: crash on valid array-like input (lists of lists are accepted everywhere else), but loud, and `np.asarray(X)` is an easy workaround. It needs an explicit `categorical_features` mask or index list: with the HGB default `"from_dtype"`, the helper returns before reaching the failing line. It has been there in HGB for several releases without a report.
- **Difficulty (1)**: one line, use `_num_features(X)` instead of `X.shape[1]`.
- **Decision (0)**: no trade-off.

## Upstream search

Queries: `categorical_features list "has no attribute 'shape'"`, `HistGradientBoosting categorical_features list input AttributeError`, `categorical_features list shape`, `_check_categorical_features`. Related but different: #33946 (merged, moved the helper to `utils.validation` unchanged), #35162 (closed, refactor of the shared encoding). No match.

## Describe the bug and its user-facing impact

`_check_categorical_features` is called on the raw `X` before any validation, so that it can read dataframe dtypes and column names. For non-dataframe input it reads `X.shape[1]`, which fails for a Python list. All estimators sharing the helper are affected: the trees and forests that gained `categorical_features` on main, and `HistGradientBoosting*` (already in released versions).

## Steps/Code to Reproduce

```python
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

X = [[0, 1.0], [1, 2.0], [2, 0.5], [0, 3.0]]
y = [0, 1, 1, 0]
for est in [DecisionTreeClassifier(categorical_features=[0]),
            RandomForestClassifier(n_estimators=2, categorical_features=[0]),
            HistGradientBoostingClassifier(categorical_features=[0], max_iter=2)]:
    try:
        est.fit(X, y)
        print(type(est).__name__, "ok")
    except Exception as e:
        print(type(est).__name__, f"{type(e).__name__}: {e}")
```

## Expected Results

The three estimators fit, as they do with `np.asarray(X)` or without `categorical_features`.

## Actual Results

```
DecisionTreeClassifier AttributeError: 'list' object has no attribute 'shape'
RandomForestClassifier AttributeError: 'list' object has no attribute 'shape'
HistGradientBoostingClassifier AttributeError: 'list' object has no attribute 'shape'
```

## Root cause and possible fix

`sklearn/utils/validation.py:2539`: `n_features = X.shape[1]`. Replace with `n_features = _num_features(X)` (defined in the same module, handles lists, dataframes and arrays). The downstream `ColumnTransformer`/`OrdinalEncoder` already accept lists.

## Versions

```
sklearn: 1.10.dev0, commit 289c4d5cfa
python: 3.13, numpy: 2.5.3, scipy: 1.18.1, platform: linux
```
