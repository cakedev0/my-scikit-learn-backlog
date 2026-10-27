# KernelDensity(bandwidth="scott") crashes on a list of lists

```json
{
    "estimators": ["KernelDensity"],
    "existing_issue": "https://github.com/scikit-learn/scikit-learn/issues/29443",
    "reproduced": true,
    "priority": 1,
    "difficulty": 1,
    "decision": 0,
    "decision_tags": []
}
```

## Classification

- **Priority (1)**: crash on valid array-like input, only with the string bandwidths; converting to an array first is an obvious workaround.
- **Difficulty (1)**: move `validate_data` above the bandwidth estimation.
- **Decision (0)**: no trade-off.

## Upstream search

- `KernelDensity scott bandwidth`: #29443 (open, "KernelDensity(bandwidth='silverman') doesn't throw proper error for 1d X", same root cause), #34991 (open PR fixing it for both 1d arrays and lists), #31436 (closed PR, earlier attempt).

Filed for the backlog since the issue and the PR are still open.

## Describe the bug and its user-facing impact

With `bandwidth="scott"` or `"silverman"`, `KernelDensity.fit` reads `X.shape` before validating `X`, so a list of lists (documented as valid `array-like`) raises `AttributeError`, and a 1d array raises `IndexError` instead of the usual "Expected 2D array" message.

## Steps/Code to Reproduce

```python
from sklearn.neighbors import KernelDensity
KernelDensity(bandwidth="scott").fit([[0.0], [1.0], [3.0]])
```

## Expected Results

The model fits, as with `bandwidth=1.0` or with `np.array(X)`.

## Actual Results

```
  File "/home/arthur/scikit-learn/sklearn/neighbors/_kde.py", line 221, in fit
    self.bandwidth_ = X.shape[0] ** (-1 / (X.shape[1] + 4))
                      ^^^^^^^
AttributeError: 'list' object has no attribute 'shape'
```

## Root cause and possible fix

`sklearn/neighbors/_kde.py:219-225`: the bandwidth is estimated from `X.shape` before `X = validate_data(...)` on line 229. Swap them, as in #34991.

## Versions

```
sklearn: 1.10.dev0, commit f570497251 (no diff in sklearn/neighbors vs upstream/main 289c4d5cfa)
python: 3.13, numpy: 2.5.3, scipy: 1.18.1, platform: linux
```
