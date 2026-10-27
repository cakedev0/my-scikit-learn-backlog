# RadiusNeighborsRegressor predicts a garbage integer instead of NaN for empty neighborhoods with integer targets

```json
{
    "estimators": ["RadiusNeighborsRegressor"],
    "existing_issue": "https://github.com/scikit-learn/scikit-learn/issues/12960",
    "reproduced": true,
    "priority": 2,
    "difficulty": 1,
    "decision": 0,
    "decision_tags": []
}
```

## Classification

- **Priority (2)**: silent garbage prediction (`-9.22e18`) without the documented "predicting NaN" warning, for any integer target (counts, ages...) and any query point without neighbors in the radius. Only the empty-neighborhood case is affected, and a float `y` is a workaround.
- **Difficulty (1)**: one line, build `empty_obs` with a floating dtype.
- **Decision (0)**: no trade-off; predictions for empty neighborhoods become NaN as already documented by the warning.

## Upstream search

- `RadiusNeighborsRegressor integer`: #12960 (open, "RadiusNeighborsRegression is inconsistent when extrapolation occurs"; a comment from 2019 identifies this exact `np.full_like(_y[0], np.nan)` problem), #34716 (open PR "FIX RadiusNeighborsRegressor silently returns a sentinel for empty neighborhoods with integer targets", fixes it and has a positive review), #33653 (closed PR, same fix, closed by its author to discuss the approach first).
- `RadiusNeighborsRegressor nan outlier`: #3760 (closed, older NaN handling), #25330 (merged, unrelated `nan_euclidean` support).

Filed for the backlog since the issue and the fixing PR are both still open.

## Describe the bug and its user-facing impact

When a query point has no training point within `radius`, `RadiusNeighborsRegressor.predict` is meant to return NaN for it and warn "One or more samples have no neighbors within specified radius; predicting NaN.". With an integer `y`, the NaN placeholder is cast to the integer dtype, which gives an arbitrary value (`-9223372036854775808` on this machine, `0` on others). Since the result is not NaN, the warning is not raised either: the only hint is a numpy `RuntimeWarning: invalid value encountered in cast`.

## Steps/Code to Reproduce

```python
import numpy as np
from sklearn.neighbors import RadiusNeighborsRegressor

X = np.array([[0.0], [1.0], [2.0]])
y = np.array([1, 2, 3])
reg = RadiusNeighborsRegressor(radius=0.5).fit(X, y)
print(reg.predict([[0.1], [100.0]]))  # second point has no neighbor
```

## Expected Results

`[1. nan]` with the `UserWarning` "One or more samples have no neighbors within specified radius; predicting NaN.", as with `y.astype(float)`.

## Actual Results

```
/home/arthur/scikit-learn/sklearn-env/lib/python3.13/site-packages/numpy/_core/numeric.py:475: RuntimeWarning: invalid value encountered in cast
  multiarray.copyto(res, fill_value, casting='unsafe')
[ 1.00000000e+00 -9.22337204e+18]
```

## Root cause and possible fix

`sklearn/neighbors/_regression.py:481`: `empty_obs = np.full_like(_y[0], np.nan)` inherits the integer dtype of `y`. Fix: `np.full_like(_y[0], np.nan, dtype=np.float64)` (or the result dtype of averaging `_y`), as done in the open PR #34716.

## Versions

```
sklearn: 1.10.dev0, commit f570497251 (no diff in sklearn/neighbors vs upstream/main 289c4d5cfa)
python: 3.13, numpy: 2.5.3, scipy: 1.18.1, platform: linux
```
