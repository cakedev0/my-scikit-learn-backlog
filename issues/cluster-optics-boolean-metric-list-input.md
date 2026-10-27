# OPTICS with a boolean metric fails on lists and DataFrames (`AttributeError: ... no attribute 'dtype'`)

```json
{
    "estimators": ["OPTICS"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 1,
    "difficulty": 1,
    "decision": 0,
    "decision_tags": []
}
```

## Classification

- **Priority (1)**: loud crash with an unhelpful error on valid input (`fit` documents `X` as array-like), but only for the boolean metrics, and `np.asarray(X)` is a trivial workaround.
- **Difficulty (1)**: read the dtype after converting `X` (e.g. `np.asarray`/`validate_data` first, or check `getattr(X, "dtype", None)`), a couple of lines.
- **Decision (0)**: no trade-off.

## Upstream search

Queries: "OPTICS jaccard", "OPTICS boolean metric". No match. Same pattern in `pairwise_distances`, filed locally as `issues/metrics-pairwise-distances-boolean-metric-list-input.md`.

## Describe the bug and its user-facing impact

`OPTICS.fit` checks `X.dtype != bool` to decide whether to warn about the conversion to boolean, before `X` goes through `validate_data`. With a boolean metric (`"jaccard"`, `"dice"`, `"rogerstanimoto"`, `"russellrao"`, `"sokalsneath"`, `"yule"`), a Python list or a pandas DataFrame (which has `dtypes`, not `dtype`) raises `AttributeError`, even when it holds booleans. With any other metric these inputs work.

## Steps/Code to Reproduce

```python
import numpy as np
import pandas as pd
from sklearn.cluster import OPTICS

X = np.random.RandomState(0).rand(20, 5) > 0.5
print(OPTICS(metric="jaccard", min_samples=3).fit(X).labels_[:5])
for X_in in [X.tolist(), pd.DataFrame(X)]:
    try:
        OPTICS(metric="jaccard", min_samples=3).fit(X_in)
    except Exception as e:
        print(f"{type(X_in).__name__}: {type(e).__name__}: {e}")
```

## Expected Results

The list and the DataFrame are clustered like the ndarray.

## Actual Results

```
[0 0 0 0 0]
list: AttributeError: 'list' object has no attribute 'dtype'
DataFrame: AttributeError: 'DataFrame' object has no attribute 'dtype'
```

(run with `-W ignore` to hide an unrelated divide-by-zero `RuntimeWarning` from the xi extraction).

## Root cause and possible fix

`sklearn/cluster/_optics.py:322`: `if dtype is bool and X.dtype != bool:` runs before `X = validate_data(self, X, dtype=dtype, accept_sparse="csr")` (line 330). Fix: compute the warning condition from the input dtype in a container-agnostic way, e.g. `getattr(X, "dtype", None)` / `np.asarray(X).dtype` for non-sparse input, or validate once without dtype conversion first. `pairwise_distances` has the same pattern, see `issues/metrics-pairwise-distances-boolean-metric-list-input.md`; both could be fixed together.

## Versions

```
sklearn: 1.10.dev0, commit 289c4d5cfa (upstream/main; local checkout f570497251 has no change in sklearn/cluster/_optics.py)
python: 3.13, numpy: 2.5, scipy: 1.18, platform: linux
```
