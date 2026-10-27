# `PowerTransformer` maps a constant feature with missing values to ~-4e31

```json
{
    "estimators": ["PowerTransformer", "power_transform"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 2,
    "difficulty": 1,
    "decision": 0,
    "decision_tags": []
}
```

## Classification

- **Priority (2)**: silent, absurd output (values around -4e31 instead of 0) with the default `method="yeo-johnson"`. Needs a constant feature that also has missing values, which is not rare in real tabular data (e.g. a flag column that is only filled for some rows).
- **Difficulty (1)**: use `np.nanmean`/`np.nanvar` and the per-feature count of non-missing values in the constant-feature check.
- **Decision (0)**: the constant-feature handling already exists, it just misses NaNs.

## Upstream search

Queries: "PowerTransformer constant feature nan", "PowerTransformer constant box-cox", "PowerTransformer yeo-johnson constant".

- #26566 (merged): leaves constant features unchanged for Yeo-Johnson; it uses `np.mean`/`np.var`, hence this bug.
- #26303 (closed, fixed by #26400): unhelpful error for an all-NaN feature with Box-Cox. Different case.
- #29595 (closed): standardization sensitivity on nearly constant data. Different case (no NaN).

## Describe the bug and its user-facing impact

Since #26566, `PowerTransformer(method="yeo-johnson")` leaves constant features unchanged (`lambda=1`) instead of running the lambda optimization on them. The constant check uses `np.mean` and `np.var`, which return `nan` as soon as the feature has a missing value. The feature is then not detected as constant, the optimizer returns an extreme lambda (~82 here), and the transformed column is about `-4e31` for every non-missing row, after standardization.

## Steps/Code to Reproduce

```python
import numpy as np
from sklearn.preprocessing import PowerTransformer

rng = np.random.RandomState(0)
X = np.c_[rng.lognormal(size=10), np.full(10, 3.0)]  # second feature is constant
pt = PowerTransformer().fit(X)
print(pt.lambdas_, pt.transform(X)[:3, 1])

X[0, 1] = np.nan  # same constant feature, with one missing value
pt = PowerTransformer().fit(X)
print(pt.lambdas_, pt.transform(X)[:3, 1])
```

## Expected Results

Same lambda (`1.0`) and same output (zeros, `nan` kept for the missing value) in both cases.

## Actual Results

```
[-0.38116064  1.        ] [0. 0. 0.]
[-0.38116064 81.94596415] [            nan -4.05648192e+31 -4.05648192e+31]
```

## Root cause and possible fix

`sklearn/preprocessing/_data.py:3404-3423`, in `PowerTransformer._fit`:

```python
n_samples = X.shape[0]
mean = np.mean(X, axis=0, dtype=np.float64)
var = np.var(X, axis=0, dtype=np.float64)
...
is_constant_feature = _is_constant_feature(var[i], mean[i], n_samples)
```

Fix: use `np.nanmean`/`np.nanvar` and pass the number of non-missing values of each feature as `n_samples`. The rest of the method (`_yeo_johnson_optimize` and the transform) already ignores NaNs.

Related but different: with `method="box-cox"`, a constant feature makes `fit` raise `ValueError: Data must not be constant.` from scipy, with or without NaNs, because the constant-feature shortcut is only applied for Yeo-Johnson.

## Versions

```
sklearn: 1.10.dev0, commit f570497251 (code identical on upstream/main 289c4d5cfa)
python: 3.13.14, numpy: 2.5.3, scipy: 1.18.1, platform: linux
```
