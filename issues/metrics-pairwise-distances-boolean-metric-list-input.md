# `pairwise_distances` with a boolean metric fails on lists and DataFrames (`AttributeError: ... no attribute 'dtype'`)

```json
{
    "estimators": ["pairwise_distances", "pairwise_distances_chunked"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 1,
    "difficulty": 1,
    "decision": 0,
    "decision_tags": []
}
```

## Classification

- **Priority (1)**: a crash with an unhelpful error on valid input (the parameter validation accepts `"array-like"`), but loud, and `np.asarray(X)` is a trivial workaround. Estimators validate `X` to an ndarray before calling `pairwise_distances`, so only direct calls are affected.
- **Difficulty (1)**: compute the warning condition after `check_pairwise_arrays`, or read the dtype from `np.asarray`.
- **Decision (0)**: no trade-off.

## Upstream search

Queries: "pairwise_distances jaccard", "boolean metric dtype", "Data was converted to boolean".

- #32495 (open): spurious `DataConversionWarning` from `pairwise_distances_argmin` with `jaccard` on bool arrays. Same lines, different symptom.
- #32511 (open, "Remove unnecessary dtype conversion from pairwise_distances_argmin_*"): refactors this check into a helper that calls `xp.asarray` first, which would fix this crash as a side effect. #33535 (open) only touches the argmin functions with `getattr(X, "dtype", None)`.
- #34465, #33426, #32510 (closed): earlier attempts at #32495.

## Describe the bug and its user-facing impact

For the scipy boolean metrics (`"jaccard"`, `"dice"`, `"rogerstanimoto"`, `"russellrao"`, `"sokalsneath"`, `"yule"`), `pairwise_distances` checks `X.dtype != bool` to decide whether to emit a `DataConversionWarning`, before `X` and `Y` are converted to arrays. Python lists and pandas DataFrames (which have `dtypes`, not `dtype`) then raise `AttributeError`, even when they hold booleans. All other metrics accept these inputs. `pairwise_distances_chunked` goes through the same code.

## Steps/Code to Reproduce

```python
import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances

X = [[True, False, True], [False, True, True]]
print(pairwise_distances(np.asarray(X), metric="jaccard"))
for X_in in (X, pd.DataFrame(X)):
    try:
        pairwise_distances(X_in, metric="jaccard")
    except Exception as e:
        print(type(X_in).__name__, "->", repr(e))
```

## Expected Results

The same matrix as for the ndarray input, for the list and the DataFrame.

## Actual Results

```
[[0.         0.66666667]
 [0.66666667 0.        ]]
list -> AttributeError("'list' object has no attribute 'dtype'")
DataFrame -> AttributeError("'DataFrame' object has no attribute 'dtype'")
```

## Root cause and possible fix

`sklearn/metrics/pairwise.py:2440`:

```python
if dtype is bool and (X.dtype != bool or (Y is not None and Y.dtype != bool)):
```

runs on the raw inputs, before `check_pairwise_arrays` at line 2444. Converting first (e.g. `np.asarray` / `check_array(..., dtype=None)`) and checking the dtype of the result, or emitting the warning based on the arrays before the `dtype=bool` cast, fixes it. The check was added in #13382 (2019), when `warn_on_dtype` was removed from `check_array`.

## Versions

```
sklearn: 1.10.dev0, commit f570497251 (no change vs upstream/main 289c4d5cfa in the files involved)
python: 3.13, numpy: 2.5, scipy: 1.18, platform: linux
```

See also: `issues/cluster-optics-boolean-metric-list-input.md` (same `X.dtype` check in `OPTICS.fit`, reachable from the estimator).
