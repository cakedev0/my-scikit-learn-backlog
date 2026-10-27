# `ColumnTransformer.fit` is quadratic in the number of selected column names (regression of #16327)

```json
{
    "estimators": ["ColumnTransformer", "make_column_transformer"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 2,
    "difficulty": 1,
    "decision": 0,
    "decision_tags": []
}
```

## Classification

- **Priority (2)**: a performance regression in 1.9 on a common meta-estimator, already reported and fixed once (#16327). It only hurts wide dataframes (tens of thousands of columns), where `fit` spends seconds looking up column names before any transformer runs.
- **Difficulty (1)**: build a `{name: index}` dict once and look names up in it, a few lines in one function.
- **Decision (0)**: same result, no trade-off.

## Upstream search

- "ColumnTransformer slow columns", "ColumnTransformer \"many columns\"", "_get_column_indices" (GitHub issue search).
- #16327 "ColumnTransformer performance very slow when columns specification a very large list of column names" (closed): the exact same symptom, fixed by #16431 (merged in 2020), which switched the pandas path to `all_columns.get_loc(col)`.
- `git log -S "get_loc(col)"` shows that #31127 "ENH add narwhals as dependency" (commit 8c62b6b66a, first released in 1.9.0) removed the pandas-specific path and routed pandas through the `column_names.index(col)` list lookup that was used for polars and the interchange protocol since #27315. So pandas regressed in 1.9.0, and polars has had the quadratic lookup since its support was added.
- #33628 (merged) only changed the error message for missing columns.
- No open issue about the regression.

## Describe the bug and its user-facing impact

`ColumnTransformer.fit` resolves string column selections with `list.index` on the list of all column names, once per selected name. That is O(n_selected * n_columns) instead of O(n_selected + n_columns).

With a dataframe of 40,000 columns and a transformer selecting half of them by name, `fit` spends ~2.3 s in this lookup, against ~30 ms for a dict lookup, and doubling the number of columns multiplies the time by ~4. #16327 reported 20 to 25 s for 50,000 columns with 40,000 selected. Wide dataframes with name-based selections (genomics, pre-encoded features, `make_column_selector` returning many names) hit this on every `fit`, so also on every CV split and grid-search candidate.

## Steps/Code to Reproduce

```python
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer

for n_columns in [5_000, 10_000, 20_000, 40_000]:
    columns = [f"feature_{i}" for i in range(n_columns)]
    X = pd.DataFrame(np.zeros((10, n_columns)), columns=columns)
    ct = ColumnTransformer([("keep", "passthrough", columns[: n_columns // 2])])
    t0 = perf_counter()
    ct.fit(X)
    t1 = perf_counter()
    # hash-based lookup of the same column indices
    lookup = {name: i for i, name in enumerate(X.columns)}
    [lookup[name] for name in columns[: n_columns // 2]]
    t2 = perf_counter()
    print(f"n_columns={n_columns}: ColumnTransformer.fit {t1 - t0:.2f}s, "
          f"dict lookup of the indices {t2 - t1:.4f}s")
```

## Expected Results

`fit` time roughly linear in the number of columns, as it was after #16431 (pandas `get_loc` is a hash lookup).

## Actual Results

Timings from a laptop loaded by other jobs, so only the growth matters:

```
n_columns=5000: ColumnTransformer.fit 0.07s, dict lookup of the indices 0.0031s
n_columns=10000: ColumnTransformer.fit 0.18s, dict lookup of the indices 0.0072s
n_columns=20000: ColumnTransformer.fit 0.60s, dict lookup of the indices 0.0127s
n_columns=40000: ColumnTransformer.fit 2.27s, dict lookup of the indices 0.0296s
```

cProfile of the 20,000-column `fit`: 2.08 s of 2.32 s are in `{method 'index' of 'list' objects}` (20,000 calls), all from `_get_column_indices`.

## Root cause and possible fix

`sklearn/utils/_indexing.py:409` (on 289c4d5cfa):

```python
return [column_names.index(col) for col in selected_columns]
```

`column_names` is the plain list of dataframe column names (line 377), so each lookup is a linear scan. The slice bounds at lines 399 and 402 do the same but only twice per call, which is fine.

Fix: build `name_to_index = {name: i for i, name in enumerate(column_names)}` once and use `name_to_index[col]`, keeping the `KeyError`/`ValueError` path that lists missing columns. Duplicate column names: `list.index` returns the first match, the dict comprehension keeps the last one, so build the dict in reverse order (or with `setdefault`) to keep the current behavior. `_check_categorical_features` in `sklearn/utils/validation.py` uses the same `feature_names.index(name)` pattern for `categorical_features` given as names, but it is less likely to matter there.

## Versions

```
sklearn: 1.10.dev0, commit 289c4d5cfa (upstream/main; checkout at f570497251, no diff in sklearn/utils/_indexing.py)
python: 3.13, numpy: 2.5.3, scipy: 1.18.1, pandas installed, platform: linux
```
