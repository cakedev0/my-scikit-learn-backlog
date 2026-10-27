# KMeans(copy_x=False) fails on read-only input instead of copying it

```json
{
    "estimators": ["KMeans", "k_means", "BisectingKMeans"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 1,
    "difficulty": 1,
    "decision": 0,
    "decision_tags": []
}
```

## Classification

- **Priority (1)**: loud error, needs both the non-default `copy_x=False` and a read-only array (`np.frombuffer`, a joblib memmap, a shared buffer); the error message does not mention the cause, and passing `copy_x=True` is the workaround.
- **Difficulty (1)**: add `force_writeable=True` to the `validate_data` call in `fit`, as #29018 did for the other estimators.
- **Decision (0)**: no trade-off; read-only input gets copied, which is what `copy_x=False` means in the docs ("a copy will be made even if `copy_x` is False" in other unsuitable cases).

## Upstream search

#14481 (closed): "Read-only X + copy=False fails for most estimators". Maintainers agreed `copy=False` means "it's ok to modify the input", not "error if we can't". It was closed by #29018 (merged), which added `force_writeable` to `check_array`/`validate_data`, used it in the estimators that modify `X` in place, and added `test_check_inplace_ensure_writeable`. That test only sets `copy=False` or `copy_X=False`, so estimators whose parameter is named `copy_x` (`KMeans`, `BisectingKMeans`) are skipped and were not fixed. Queries: `KMeans read-only`, `copy_x read-only`, `KMeans "output array is read-only"`.

## Describe the bug and its user-facing impact

With `copy_x=False`, `KMeans.fit` and `BisectingKMeans.fit` center `X` in place (`X -= X_mean`) and restore it at the end. If `X` is a read-only array, `validate_data(..., copy=False)` returns it as is and the in-place subtraction raises `ValueError: output array is read-only`. Other estimators with a `copy=False` option handle the same input by making a copy.

## Steps/Code to Reproduce

```python
import numpy as np
from sklearn.cluster import KMeans, BisectingKMeans

X = np.random.RandomState(0).rand(100, 3)
X.flags.writeable = False  # e.g. np.frombuffer, a joblib memmap, a shared buffer

for est in [KMeans(n_clusters=3, copy_x=False), BisectingKMeans(n_clusters=3, copy_x=False)]:
    try:
        est.fit(X)
        print(type(est).__name__, "ok")
    except ValueError as e:
        print(type(est).__name__, "ValueError:", e)
```

## Expected Results

Both fits succeed (with an internal copy), like `copy_x=True` and like estimators fixed in #29018.

## Actual Results

```
KMeans ValueError: output array is read-only
BisectingKMeans ValueError: output array is read-only
```

## Root cause and possible fix

- `sklearn/cluster/_kmeans.py:1447-1455`: `validate_data(..., copy=self.copy_x)` without `force_writeable=True`, then `X -= X_mean` at `:1474`.
- `sklearn/cluster/_bisect_k_means.py:397` and `:416`: same pattern.
- `sklearn/tests/test_common.py:417-422`: the common test only knows `copy` and `copy_X`.

Fix: pass `force_writeable=True` in both `fit` methods, and make the common test also handle `copy_x`.

## Versions

```
sklearn: 1.10.dev0, commit 289c4d5cfa
python: 3.13, numpy: 2.5.3, scipy: 1.18.1, platform: linux
```
