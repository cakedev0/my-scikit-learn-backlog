# Bray-Curtis distance is wrong on sparse input (`pairwise_distances_argmin_min`, `DistanceMetric`)

```json
{
    "estimators": ["pairwise_distances_argmin_min", "pairwise_distances_argmin", "DistanceMetric"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 1,
    "difficulty": 1,
    "decision": 0,
    "decision_tags": []
}
```

## Classification

- **Priority (1)**: silent wrong distances, but only reachable through `DistanceMetric.pairwise` and `pairwise_distances_argmin(_min)` on sparse input with `metric="braycurtis"`: `pairwise_distances` and `NearestNeighbors` reject sparse input for this metric, so it is a rare direct-call path within an uncommon metric. (priority review, 2026-10-23: was 2)
- **Difficulty (1)**: swap `x1_data[i1]` and `x2_data[i2]` in the two tail loops, and return 0 when `denom == 0` like the dense `dist`.
- **Decision (0)**: no trade-off.

## Upstream search

Queries: "braycurtis", "braycurtis sparse", "DistanceMetric sparse".

No match. #23604 (merged, "FEA CSR support for all `DistanceMetric`") introduced the CSR implementations; #22427 (open, "Improve the metrics API") is unrelated.

## Describe the bug and its user-facing impact

`BrayCurtisDistance.dist_csr` merges the stored indices of the two rows. When one row runs out of stored entries, the loop that consumes the remaining entries of the other row reads the wrong array: it adds `x1_data[i1]` while iterating over `i2`, and `x2_data[i2]` while iterating over `i1`. Since `i1 == x1_end` at that point, it reads the first value of the next row (or past the end of `data` for the last row) instead of the remaining values of the current row.

This path is used by `DistanceMetric.pairwise` on CSR input and by the `ArgKmin` reduction, so `pairwise_distances_argmin_min` and `pairwise_distances_argmin` return wrong distances for `metric="braycurtis"` whenever `X` or `Y` is sparse (CSR/CSR and CSR/dense: the dense side is wrapped as a CSR view). In addition, two rows with no stored values give `0 / 0 = nan`, while the dense implementation returns 0.

## Steps/Code to Reproduce

```python
import numpy as np
import scipy.sparse as sp
from scipy.spatial.distance import cdist
from sklearn.metrics import DistanceMetric, pairwise_distances_argmin_min

X = np.array([[1.0, 1.0, 0.0], [5.0, 0.0, 0.0]])
Y = np.array([[0.0, 1.0, 2.0]])

dm = DistanceMetric.get_metric("braycurtis")
print("scipy     :", cdist(X, Y, "braycurtis").ravel())
print("dense     :", dm.pairwise(X, Y).ravel())
print("csr/csr   :", dm.pairwise(sp.csr_matrix(X), sp.csr_matrix(Y)).ravel())
print("csr/dense :", dm.pairwise(sp.csr_matrix(X), Y).ravel())

# Same code path through a public function
_, d = pairwise_distances_argmin_min(sp.csr_matrix(Y), sp.csr_matrix(X), metric="braycurtis")
print("pairwise_distances_argmin_min(csr):", d, "expected:", cdist(Y, X, "braycurtis").min())

# Two all-zero rows
Z = np.array([[0.0, 0.0], [1.0, 0.0]])
print(dm.pairwise(Z)[0, 0], dm.pairwise(sp.csr_matrix(Z))[0, 0])
```

## Expected Results

Sparse and dense inputs give the same distances as scipy: `[0.6 1.]`, a minimum distance of `0.6`, and 0 between two all-zero rows.

## Actual Results

```
scipy     : [0.6 1. ]
dense     : [0.6 1. ]
csr/csr   : [0.75 1.  ]
csr/dense : [0.75 1.  ]
pairwise_distances_argmin_min(csr): [0.75] expected: 0.6
0.0 nan
```

## Root cause and possible fix

`sklearn/metrics/_dist_metrics.pyx.tp:1961-1970`:

```cython
        if i1 == x1_end:
            while i2 < x2_end:
                num += fabs(x1_data[i1])   # should be x2_data[i2]
                denom += fabs(x1_data[i1])
                i2 = i2 + 1
        else:
            while i1 < x1_end:
                num += fabs(x2_data[i2])   # should be x1_data[i1]
                denom += fabs(x2_data[i2])
                i1 = i1 + 1

        return num / denom                 # nan when both rows are empty
```

Swap the arrays in the two tail loops and return 0 when `denom == 0`, as `BrayCurtisDistance.dist` does (lines 1917-1920). `test_dist_metrics.py::test_cdist` does compare CSR and dense inputs, but on `rng.random_sample` data with no zeros: every CSR row stores all its entries, so the tail loops never run. Running the sparse/dense comparison on data with zeros (the same goes for Canberra and Mahalanobis, see `metrics-canberra-csr-explicit-zeros.md` and `metrics-mahalanobis-csr-stale-buffer.md`) would catch it.

## Versions

```
sklearn: 1.10.dev0, commit f570497251 (no change vs upstream/main 289c4d5cfa in the files involved)
python: 3.13, numpy: 2.5, scipy: 1.18, platform: linux
```
