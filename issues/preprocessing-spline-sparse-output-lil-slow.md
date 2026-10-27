# `SplineTransformer(sparse_output=True)` is 15x to 80x slower than dense output with periodic, constant or linear extrapolation

```json
{
    "estimators": ["SplineTransformer"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 1,
    "difficulty": 2,
    "decision": 0,
    "decision_tags": []
}
```

## Classification

- **Priority (1)**: performance only, results are correct, and it needs the opt-in `sparse_output=True` combined with periodic extrapolation or out-of-range/missing values; ~1.6 s per feature for 200k rows is slow but not blocking, and `sparse_output=False` is a workaround. (priority review, 2026-10-23: was 2)
- **Difficulty (2)**: the fix rewrites the sparse branches to work on CSR/COO arrays directly (column wrap with modulo + `sum_duplicates`, row masking, adding the extrapolated entries), in one method, with existing tests comparing sparse and dense outputs.
- **Decision (0)**: same output, only the sparse construction changes.

## Upstream search

- "SplineTransformer sparse", "SplineTransformer sparse_output", "SplineTransformer slow" (GitHub issue search).
- #20998 / #24145 (closed / merged): the issue and PR that added `sparse_output`. The LIL conversions come from that PR, to avoid `SparseEfficiencyWarning`.
- #28043 / #33953 (merged): missing value support, which added one more `tolil()` for NaN rows.
- No issue or PR about the speed of the sparse path.

## Describe the bug and its user-facing impact

With `sparse_output=True`, `SplineTransformer.transform` converts the per-feature design matrix to LIL (`tolil()`) to edit it, as soon as:

- `extrapolation="periodic"` (always),
- `extrapolation="constant"` or `"linear"` and some values fall outside the training range,
- or the input has missing values.

Converting to LIL builds one Python list per row, and the LIL slicing and assignment that follow also loop over rows in Python. On 200,000 samples and a single feature, `transform` takes ~1.6 s with periodic extrapolation against ~0.02 s for the dense output (~80x), and ~0.4 s for constant or linear extrapolation (~15x). This is per input feature, so 10 periodic features on 1M rows take on the order of a minute, while `extrapolation="continue"` (no edit) shows no gap at all.

## Steps/Code to Reproduce

```python
from time import perf_counter

import numpy as np
from sklearn.preprocessing import SplineTransformer

rng = np.random.RandomState(0)
X_train = rng.uniform(0, 1, size=(1_000, 1))
SplineTransformer().fit(X_train).transform(X_train)  # warm-up
for extrapolation in ["continue", "periodic", "constant", "linear"]:
    for n_samples in [50_000, 100_000, 200_000]:
        # values outside of [0, 1] trigger the extrapolation code
        X = rng.uniform(-0.1, 1.1, size=(n_samples, 1))
        times = {}
        for sparse_output in [False, True]:
            spline = SplineTransformer(
                extrapolation=extrapolation, sparse_output=sparse_output
            ).fit(X_train)
            t0 = perf_counter()
            out = spline.transform(X)
            times[sparse_output] = perf_counter() - t0
            if sparse_output:
                np.testing.assert_allclose(out.toarray(), dense)
            else:
                dense = out
        print(f"{extrapolation:>8} n_samples={n_samples}: dense {times[False]:.3f}s, "
              f"sparse_output=True {times[True]:.3f}s")
```

## Expected Results

The sparse output costs about the same as the dense output (as with `extrapolation="continue"`), since the design matrix has only `degree + 1` non-zeros per row.

## Actual Results

Timings from a laptop loaded by other jobs:

```
continue n_samples=50000: dense 0.009s, sparse_output=True 0.009s
continue n_samples=100000: dense 0.011s, sparse_output=True 0.012s
continue n_samples=200000: dense 0.021s, sparse_output=True 0.019s
periodic n_samples=50000: dense 0.006s, sparse_output=True 0.375s
periodic n_samples=100000: dense 0.010s, sparse_output=True 0.778s
periodic n_samples=200000: dense 0.019s, sparse_output=True 1.567s
constant n_samples=50000: dense 0.005s, sparse_output=True 0.117s
constant n_samples=100000: dense 0.010s, sparse_output=True 0.130s
constant n_samples=200000: dense 0.026s, sparse_output=True 0.417s
  linear n_samples=50000: dense 0.007s, sparse_output=True 0.066s
  linear n_samples=100000: dense 0.014s, sparse_output=True 0.201s
  linear n_samples=200000: dense 0.030s, sparse_output=True 0.438s
```

cProfile of the periodic case (200,000 samples): of 2.15 s, 1.25 s are in `lil_matrix.__init__`/`tocsr().tolil()`, 0.73 s in LIL column slicing (`_get_row_ranges`) and 0.54 s in LIL `__setitem__`/`__iadd__`.

## Root cause and possible fix

`sklearn/preprocessing/_polynomial.py` (on 289c4d5cfa), in `SplineTransformer.transform`:

- line 1073: periodic case, `XBS_sparse = XBS_sparse.tolil()` then `XBS_sparse[:, :degree] += XBS_sparse[:, -degree:]` and `XBS_sparse[:, :-degree]`;
- line 1101: constant/linear, zeroing the out-of-range rows;
- line 1114: zeroing the NaN rows;
- lines 1154, 1169, 1200, 1216: writing the extrapolated values.

Possible fix, all on the CSR output of `BSpline.design_matrix`:

- periodic: with `m = n_columns - degree` the number of output splines, wrap the last `degree` columns onto the first ones, `indices = np.where(indices >= m, indices - m, indices)`, build the CSR with `shape=(n, m)` and call `sum_duplicates()`;
- zeroing rows: multiply `data` by the row mask expanded with `np.repeat(mask, np.diff(indptr))`, then `eliminate_zeros()`;
- extrapolated values: the affected rows are zero at that point, so build the new entries as a COO matrix (row indices from the mask, column `j`, the computed values) and add it.

## Versions

```
sklearn: 1.10.dev0, commit 289c4d5cfa (upstream/main; checkout at f570497251, no diff in sklearn/preprocessing/_polynomial.py)
python: 3.13, numpy: 2.5.3, scipy: 1.18.1, platform: linux
```
