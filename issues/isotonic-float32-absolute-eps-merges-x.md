# `IsotonicRegression` merges distinct float32 `X` values closer than 1e-6 (absolute tolerance)

```json
{
    "estimators": ["IsotonicRegression"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 1,
    "difficulty": 2,
    "decision": 1,
    "decision_tags": ["output-change"]
}
```

## Classification

- **Priority (1)**: silent, but needs float32 `X` (estimators and `CalibratedClassifierCV` usually feed float64) whose values differ by less than 1e-6, i.e. a feature living on a ~1e-5 scale or below; for values of order 0.01-1 the merge is below what float32 resolves meaningfully. Workaround: cast `X` to float64. (priority review, 2026-10-23: was 2)
- **Difficulty (2)**: the tolerance must become relative to the magnitude of `x` while still fixing the original problem (near-duplicate `X` values producing infinite slopes in `interp1d`, #10903).
- **Decision (1)**: results change for float32 inputs with nearby values (`output-change`), which is the point of the fix.

## Upstream search

GraphQL searches "IsotonicRegression float32 make_unique", "IsotonicRegression close X values merged", "isotonic _make_unique eps resolution", "isotonic float32", "CalibratedClassifierCV isotonic float32 collapse".

- #18639 (merged, "BUG Add tol to _make_unique to avoid inf values in IsotonicRegression", fixes #10903 / #17758): introduced the tolerance. The PR description motivates it with `1e-15` for float64 ("this agrees with Python: `1 == 1+1e-15` is False"); the implementation uses `np.finfo(dtype).resolution`, i.e. `1e-6` for float32, as an absolute threshold. The float32 consequence is not discussed.
- No other match.

## Describe the bug and its user-facing impact

`IsotonicRegression.fit` sorts `X` and calls `_make_unique` to merge duplicate `X` values (averaging their `y`). Two values are considered equal when `x - current_x < eps` with `eps = np.finfo(dtype).resolution`, an absolute threshold: `1e-15` for float64 but `1e-6` for float32. Since `fit` keeps float32 input as float32, any float32 `X` values closer than 1e-6 to the first value of their group are merged.

For features whose values are small (here `1e-7` to `1e-6`) the whole training set collapses into one point and `predict` returns a constant. For values in `[0, 1e-3]` with many samples, many distinct values get merged and the fit loses resolution. The float64 fit of the same data is correct, so the result silently depends on the input dtype.

## Steps/Code to Reproduce

```python
import numpy as np
from sklearn.isotonic import IsotonicRegression

X = np.arange(1, 11) * 1e-7  # e.g. small rates or scores
y = np.arange(10.0)

for dtype in [np.float64, np.float32]:
    ir = IsotonicRegression().fit(X.astype(dtype), y.astype(dtype))
    print(f"{dtype.__name__}: X_thresholds_={ir.X_thresholds_}")
    print(f"         predict={ir.predict(X.astype(dtype))}")
```

## Expected Results

float32 and float64 give the same fit (up to float32 precision): `X` values `1e-7, ..., 1e-6` are perfectly representable and distinct in float32, and `y` is already increasing.

## Actual Results

```
float64: X_thresholds_=[1.e-07 2.e-07 3.e-07 4.e-07 5.e-07 6.e-07 7.e-07 8.e-07 9.e-07 1.e-06]
         predict=[0. 1. 2. 3. 4. 5. 6. 7. 8. 9.]
float32: X_thresholds_=[1.e-07]
         predict=[4.5 4.5 4.5 4.5 4.5 4.5 4.5 4.5 4.5 4.5]
```

## Root cause and possible fix

`sklearn/_isotonic.pyx:92-96`:

```cython
cdef floating eps = np.finfo(dtype).resolution
...
if x - current_x >= eps:
```

The tolerance is absolute, so it is only meaningful for values of order 1. A relative tolerance, e.g. `x - current_x >= eps * max(fabs(x), fabs(current_x))` (with a tiny absolute floor for values near 0), would still merge values that are equal up to rounding (the case of #10903) without merging distinct small values. Alternatively, merge only exact duplicates and guard against infinite slopes where they are created (the interpolation step).

## Versions

```
sklearn: 1.10.dev0, commit 289c4d5cfa (upstream/main; local HEAD f570497251 has no diff in sklearn/_isotonic.pyx or sklearn/isotonic.py)
python: 3.13, numpy: 2.5.3, scipy: 1.18.1, platform: linux
```
