# `IterativeImputer(keep_empty_features=True)` writes into the user's input (fails on pandas 3 DataFrames)

```json
{
    "estimators": ["IterativeImputer"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 1,
    "difficulty": 1,
    "decision": 0,
    "decision_tags": []
}
```

## Classification

- **Priority (1)**: needs both the opt-in `keep_empty_features=True` and a training column that is entirely missing, on the experimental `IterativeImputer`; on pandas 3 it is a loud error, and the silent in-place write only hits inputs that `validate_data` passes through uncopied. (priority review, 2026-10-23: was 2)
- **Difficulty (1)**: copy before the assignment, a one-line change.
- **Decision (0)**: an imputer must not modify its input (there is no `copy` parameter on `IterativeImputer`).

## Upstream search

Queries: "IterativeImputer keep_empty_features", "IterativeImputer read-only".

- #35157 (closed, "Needs Triage", closed by its author 2026-10-18 without any comment): "IterativeImputer(keep_empty_features=True) modifies the input array in place". Same root cause; this report adds the pandas 3 failure (`assignment destination is read-only`), which makes it visible for DataFrame users. The bug is still present on main.
- #35159 (closed by the bot because the issue was not triaged): an unreviewed fix.
- #25883, #29375/#29779 (closed/merged): other `keep_empty_features` bugs (shape mismatch, skipped iterations), already fixed; #29779 introduced the current branch.

## Describe the bug and its user-facing impact

With `keep_empty_features=True`, `IterativeImputer._initial_imputation` fills the all-missing columns directly in the array returned by `validate_data`. `validate_data` does not copy an input that is already a float, Fortran-ordered array, so:

- on a numpy array in Fortran order (or any float array that `validate_data` passes through), the user's array is modified: the all-missing column is overwritten with the fill value;
- on a pandas 3 DataFrame (copy-on-write, so `np.asarray(df)` is a read-only view), `fit`, `fit_transform` and `transform` raise `ValueError: assignment destination is read-only`.

This happens in both `fit_transform` and `transform`.

## Steps/Code to Reproduce

```python
import numpy as np
import pandas as pd
from sklearn.impute import IterativeImputer

# numpy, Fortran-ordered float array: the input is modified in place
X = np.asfortranarray([[1.0, 2.0, np.nan], [np.nan, 1.0, np.nan], [3.0, np.nan, np.nan], [4.0, 5.0, np.nan]])
IterativeImputer(keep_empty_features=True).fit_transform(X)
print(X[:, 2])  # was all nan

# pandas DataFrame (pandas 3, copy-on-write): fails
df = pd.DataFrame({"a": [1.0, np.nan, 3.0, 4.0], "b": [2.0, 1.0, np.nan, 5.0], "c": np.nan})
IterativeImputer(keep_empty_features=True).fit_transform(df)
```

## Expected Results

`X` unchanged (`[nan nan nan nan]`), and the DataFrame is imputed without error, as with `keep_empty_features=False` or with `SimpleImputer(keep_empty_features=True)`.

## Actual Results

```
[0. 0. 0. 0.]
Traceback (most recent call last):
  File ".../E.py", line 12, in <module>
    IterativeImputer(keep_empty_features=True).fit_transform(df)
  File "/home/arthur/scikit-learn/sklearn/impute/_iterative.py", line 762, in fit_transform
    X, Xt, mask_missing_values, complete_mask = self._initial_imputation(
        X, in_fit=True
    )
  File "/home/arthur/scikit-learn/sklearn/impute/_iterative.py", line 655, in _initial_imputation
    Xt[:, self._is_empty_feature] = X_filled[:, self._is_empty_feature]
    ~~^^^^^^^^^^^^^^^^^^^^^^^^^^^
ValueError: assignment destination is read-only
```

(pandas 3.0.6)

## Root cause and possible fix

`sklearn/impute/_iterative.py:650-655`:

```python
else:
    # mark empty features as not missing and keep the original
    # imputation
    mask_missing_values[:, self._is_empty_feature] = False
    Xt = X
    Xt[:, self._is_empty_feature] = X_filled[:, self._is_empty_feature]
```

`X` is the validated input, which can be the caller's buffer. Fix: `Xt = X.copy()` (only when there are empty features, to avoid a copy otherwise), or pass `copy=True` to `validate_data` in that branch. The `keep_empty_features=False` branch is fine because `X[:, ~self._is_empty_feature]` makes a copy.

## Versions

```
sklearn: 1.10.dev0, commit f570497251 (code identical on upstream/main 289c4d5cfa)
python: 3.13.14, numpy: 2.5.3, scipy: 1.18.1, pandas: 3.0.6, platform: linux
```
