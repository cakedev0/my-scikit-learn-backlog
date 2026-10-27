# Trees in `estimators_` of a forest fitted with `categorical_features` crash in `predict`

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

- **Priority (2)**: calling `predict` on the individual trees of a forest is a common pattern (per-tree predictions for prediction intervals, variance estimates, debugging), and it crashes with an unhelpful `AttributeError` as soon as `categorical_features` is used. There is no public workaround: the trees expect the encoded input, and the encoding is only reachable through private API. The feature is not released yet (1.10.dev0), so it can be fixed before users hit it.
- **Difficulty (1)**: give each tree a reference to the forest's fitted `_preprocessor` (it is fitted once, sharing it costs nothing), or raise a clear error. A few lines in `_forest.py` and `_classes.py`.
- **Decision (0)**: no trade-off, the forest's own predictions are unchanged.

## Upstream search

Queries: `forest estimators_ predict categorical NoneType transform`, `forest categorical individual trees predict`, plus a look at #34832 (merged, adds categorical support to forests) and #35162 (closed, refactor of the shared encoding). No match.

## Describe the bug and its user-facing impact

`BaseForest.fit` encodes the categorical columns once and passes the encoded array to the trees with `categorical_counts`. In that path, `BaseDecisionTree._fit` sets `self._preprocessor = None` but keeps `is_categorical_` set. Then `tree.predict(X)`, `tree.apply(X)` and `tree.decision_path(X)` with the default `check_input=True` go to `_preprocess_X(X, reset=False)`, which calls `self._preprocessor.transform(X)` on `None`.

The forest itself predicts fine because it encodes `X` and calls the trees with `check_input=False`. Any user code that loops over `rf.estimators_` breaks, whatever the input (raw categories or already encoded values).

## Steps/Code to Reproduce

```python
import numpy as np
from sklearn.ensemble import RandomForestRegressor

rng = np.random.RandomState(0)
X = np.c_[rng.choice([10, 20, 30], size=100), rng.randn(100)]
y = (X[:, 0] == 20) + X[:, 1]

rf = RandomForestRegressor(n_estimators=3, categorical_features=[0], random_state=0)
rf.fit(X, y)
print(rf.predict(X[:2]))
print([tree.predict(X[:2]) for tree in rf.estimators_])
```

## Expected Results

Each tree predicts on the raw `X` like a standalone `DecisionTreeRegressor(categorical_features=[0])` would, and the mean of the per-tree predictions equals `rf.predict(X)`.

## Actual Results

```
[-0.68658948  1.01487332]
Traceback (most recent call last):
  File "r_tree_predict.py", line 11, in <module>
    print([tree.predict(X[:2]) for tree in rf.estimators_])
           ~~~~~~~~~~~~^^^^^^^
  File "/home/arthur/scikit-learn/sklearn/tree/_classes.py", line 736, in predict
    X = self._validate_X_predict(X, check_input)
  File "/home/arthur/scikit-learn/sklearn/tree/_classes.py", line 679, in _validate_X_predict
    X = self._preprocess_X(X, reset=False)
  File "/home/arthur/scikit-learn/sklearn/tree/_classes.py", line 644, in _preprocess_X
    X_transformed = self._preprocessor.transform(X)
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AttributeError: 'NoneType' object has no attribute 'transform'
```

## Root cause and possible fix

`sklearn/tree/_classes.py:263` sets `self._preprocessor = None` when the tree is fitted by a forest (`categorical_counts` given). `_validate_X_predict` (`sklearn/tree/_classes.py:675-680`) only skips the encoding when `check_input=False` (line 708 has the `_preprocessor is not None` guard, the `check_input=True` branch does not).

Possible fix: pass the forest's fitted `_preprocessor` (and `_categorical_encoder`) to `_parallel_build_trees` (`sklearn/ensemble/_forest.py:543`) and store it on each tree instead of `None`. The trees then accept the same raw input as the forest. The objects are shared, so there is no memory cost. The alternative, raising an informative error, is simpler but leaves users with no public way to get per-tree predictions.

## Versions

```
sklearn: 1.10.dev0, commit 289c4d5cfa
python: 3.13, numpy: 2.5.3, scipy: 1.18.1, platform: linux
```
