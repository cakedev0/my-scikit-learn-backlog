# `KNNImputer.get_feature_names_out` drops empty features even with `keep_empty_features=True`

```json
{
    "estimators": ["KNNImputer"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 1,
    "difficulty": 1,
    "decision": 0,
    "decision_tags": []
}
```

## Classification

- **Priority (1)**: needs both the opt-in `keep_empty_features=True` and a training column that is entirely missing; the main symptom (`set_output` to pandas/polars) is a loud length-mismatch error, and only names (not values) are wrong otherwise. (priority review, 2026-10-23: was 2)
- **Difficulty (1)**: one line in `get_feature_names_out`; `SimpleImputer` and `IterativeImputer` already do it right.
- **Decision (0)**: names should match the output columns.

## Upstream search

Queries (issues and PRs): "KNNImputer keep_empty_features get_feature_names_out", "KNNImputer feature names". No match. `keep_empty_features` was added in #24770, which updated `transform` but not `get_feature_names_out` (added earlier in #21078). `sklearn/impute/tests/test_common.py::test_keep_empty_features` only checks the output shape, not the names.

## Describe the bug and its user-facing impact

With `keep_empty_features=True`, `KNNImputer.transform` keeps the columns that were all-missing during `fit` (filled with 0), but `get_feature_names_out` still filters them out with `_valid_mask`. The names are shorter than the output, so:

- `set_output(transform="pandas")` or `"polars"` raises a length mismatch error;
- with `add_indicator=True`, the indicator names are shifted onto the wrong columns (3 names for 4 columns with one empty feature among 3).

`SimpleImputer` and `IterativeImputer` return all input names in the same situation.

## Steps/Code to Reproduce

```python
import numpy as np
from sklearn.impute import KNNImputer

X = np.array([[1.0, np.nan], [3.0, np.nan], [5.0, np.nan]])  # 2nd column is empty

imputer = KNNImputer(keep_empty_features=True).fit(X)
print("transform(X).shape[1]:  ", imputer.transform(X).shape[1])
print("get_feature_names_out():", imputer.get_feature_names_out())

imputer.set_output(transform="pandas").transform(X)
```

## Expected Results

`get_feature_names_out()` returns `['x0', 'x1']`, and the pandas output has these two columns, as with `SimpleImputer(keep_empty_features=True)`.

## Actual Results

```
transform(X).shape[1]:   2
get_feature_names_out(): ['x0']
Traceback (most recent call last):
  ...
ValueError: Length mismatch: Expected axis has 2 elements, new values have 1 elements
```

## Root cause and possible fix

`sklearn/impute/_knn.py:406`:

```python
names = input_features[self._valid_mask]
```

ignores `self.keep_empty_features`, while `transform` (`_knn.py:287-291` and below) keeps all columns in that case. Fix:

```python
names = input_features if self.keep_empty_features else input_features[self._valid_mask]
```

plus a names check in `test_keep_empty_features`.

## Versions

```
sklearn: 1.10.dev0, commit f570497251 (sklearn/impute/_knn.py identical to upstream/main 289c4d5cfa)
python: 3.13.14, numpy: 2.5.3, scipy: 1.18.1, pandas: 3.0.6, platform: linux
```
