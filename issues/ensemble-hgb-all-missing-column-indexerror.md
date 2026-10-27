# HistGradientBoosting fit crashes on a feature with only missing values (regression since 1.9.0)

```json
{
    "estimators": ["HistGradientBoostingClassifier", "HistGradientBoostingRegressor"],
    "existing_issue": "https://github.com/scikit-learn/scikit-learn/issues/35164",
    "reproduced": true,
    "priority": 3,
    "difficulty": 1,
    "decision": 0,
    "decision_tags": []
}
```

## Classification

- **Priority (3)**: regression in 1.9.0 that makes `fit` crash on valid input (it worked in 1.8). A column that is entirely missing in a training set or a CV fold is common in real data, and the only workaround is to drop the column beforehand.
- **Difficulty (1)**: return empty thresholds when no non-missing value is left, before building `distinct_mask`.
- **Decision (0)**: restores the previous behavior.

## Upstream search

#35164 (open, needs triage) reports this exact bug with a slightly different reproducer. A commenter has a local fix but no PR is open (checked the issue timeline for cross-references). I'm filing it here to record the classification and the history, which the issue doesn't give: it is a regression from #29641 (merged, "FIX Added sample weight handling to BinMapper under HGBT", in 1.9.0), which replaced the midpoint computation with `sliding_window_view(distinct_values, 2)`: that raises `ValueError: window shape cannot be larger than input array shape` on an empty array. #34248 (on main only) then changed the error into an `IndexError`.

## Describe the bug and its user-facing impact

`fit` raises `IndexError: index 0 is out of bounds for axis 0 with size 0` as soon as one feature has no non-missing value in the training data. With `sample_weight`, it also happens when all non-missing values of a feature have zero weight. In 1.8 the feature just got no thresholds and was never used for splits. Reading the 1.9.1 code, 1.9.0 and 1.9.1 raise `ValueError: window shape cannot be larger than input array shape` instead (I checked that `sliding_window_view(np.array([]), 2)` raises this, not the full 1.9.1 fit).

## Steps/Code to Reproduce

```python
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

rng = np.random.RandomState(0)
X = np.c_[rng.normal(size=100), np.full(100, np.nan)]  # second column fully missing
y = X[:, 0]
HistGradientBoostingRegressor().fit(X, y)
```

## Expected Results

The model fits, as in 1.8, and ignores the all-missing feature.

## Actual Results

```
joblib.externals.loky.process_executor._RemoteTraceback: 
"""
Traceback (most recent call last):
  ...
  File "/home/arthur/scikit-learn/sklearn/ensemble/_hist_gradient_boosting/binning.py", line 80, in _find_binning_thresholds
IndexError: index 0 is out of bounds for axis 0 with size 0
"""

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  ...
  File "/home/arthur/scikit-learn/sklearn/ensemble/_hist_gradient_boosting/gradient_boosting.py", line 606, in fit
  File "/home/arthur/scikit-learn/sklearn/ensemble/_hist_gradient_boosting/gradient_boosting.py", line 1125, in _bin_data
  ...
  File "/home/arthur/scikit-learn/sklearn/ensemble/_hist_gradient_boosting/binning.py", line 289, in fit
  ...
IndexError: index 0 is out of bounds for axis 0 with size 0
```

(joblib frames trimmed.)

## Root cause and possible fix

`_find_binning_thresholds` (`sklearn/ensemble/_hist_gradient_boosting/binning.py:79-80` on the commit below) removes the missing values (and the zero-weight samples), then does:

```python
distinct_mask = np.empty(len(col_data), dtype=bool)
distinct_mask[0] = True
```

which fails when `col_data` is empty. In 1.8, the midpoints of an empty array of distinct values were an empty threshold array. In 1.9.x, `sliding_window_view(distinct_values, 2)` fails on it. Fix: `if len(col_data) == 0: return np.asarray([])` right after the filtering (or `n_distincts <= 1`). A non-regression test with an all-NaN column, with and without `sample_weight`, is missing.

## Versions

```
sklearn: 1.10.dev0, commit 289c4d5cfa (upstream/main; run on a local editable build with the same binning.py)
python: 3.13, numpy: 2.5.3, scipy: 1.18.1, platform: linux
```
