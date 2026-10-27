# Canberra distance is wrong for mixed sparse/dense input and CSR with explicit zeros

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

- **Priority (1)**: silent wrong distances and argmins, but only through `DistanceMetric.pairwise` and `pairwise_distances_argmin(_min)` with sparse input and `metric="canberra"`: `pairwise_distances` and `NearestNeighbors` reject sparse input for this metric, so few users can reach it. (priority review, 2026-10-23: was 2)
- **Difficulty (1)**: in `dist_csr`, skip the term when `|x1| + |x2| == 0` and add `1` only when the single stored value is non-zero.
- **Decision (0)**: no trade-off.

## Upstream search

Queries: "canberra sparse", "DistanceMetric sparse", "braycurtis sparse".

No match. #23604 (merged) added the CSR implementations, and #24556 (merged) the CSR/CSR euclidean specializations; neither discusses this.

## Describe the bug and its user-facing impact

Canberra is `sum_i |x_i - y_i| / (|x_i| + |y_i|)`, with `0/0` terms skipped (scipy and the dense sklearn implementation both do this). `CanberraDistance.dist_csr` assumes that a stored value is non-zero: when an index is stored in only one row it adds `1`, and when it is stored in both it divides without checking the denominator.

The base class documents that `dist_csr` "must be robust to the presence of explicit zeros in the CSR representation", and the sparse/dense pairs rely on it: the dense side is wrapped as a CSR view with every index stored (`SparseDenseDatasetsPair`, `DistanceMetric._pairwise_sparse_dense`). So:

- CSR/dense or dense/CSR: each coordinate where both vectors are 0 counts as 1 instead of 0.
- CSR input with an explicitly stored 0: the same, or `nan` when both rows store a 0 at the same index.

Through `ArgKmin`, `pairwise_distances_argmin` and `pairwise_distances_argmin_min` return wrong distances and can return the wrong nearest row.

## Steps/Code to Reproduce

```python
import numpy as np
import scipy.sparse as sp
from scipy.spatial.distance import cdist
from sklearn.metrics import DistanceMetric, pairwise_distances_argmin_min

dm = DistanceMetric.get_metric("canberra")

# Sparse vs dense input: the 2 coordinates where both vectors are 0 count as 1 each
X = np.array([[1.0, 0.0, 0.0]])
print("scipy     :", cdist(X, X, "canberra").ravel())
print("dense     :", dm.pairwise(X, X).ravel())
print("csr/csr   :", dm.pairwise(sp.csr_matrix(X), sp.csr_matrix(X)).ravel())
print("csr/dense :", dm.pairwise(sp.csr_matrix(X), X).ravel())

# CSR matrix with an explicitly stored zero
Xe = sp.csr_matrix((np.array([1.0, 0.0]), np.array([0, 1]), np.array([0, 2])), shape=(1, 3))
print("explicit zero vs csr:", dm.pairwise(Xe, sp.csr_matrix(X)).ravel())
print("explicit zero vs itself:", dm.pairwise(Xe, Xe).ravel())

# Public function: wrong nearest neighbor
Q = np.array([[0.0, 0.9, 0.0]])
Y = np.array([[0.9, 0.5, 0.6], [0.8, 0.5, 0.0]])
print("expected:", cdist(Q, Y, "canberra").argmin(), cdist(Q, Y, "canberra").min())
print("dense Q, csr Y:", pairwise_distances_argmin_min(Q, sp.csr_matrix(Y), metric="canberra"))
```

## Expected Results

All variants give the scipy result: distance 0 between a vector and itself, and for the last call the nearest row is `1` at distance `1.2857...`.

## Actual Results

```
scipy     : [0.]
dense     : [0.]
csr/csr   : [0.]
csr/dense : [2.]
explicit zero vs csr: [1.]
explicit zero vs itself: [nan]
expected: 1 1.2857142857142858
dense Q, csr Y: (array([0]), array([2.28571429]))
```

## Root cause and possible fix

`sklearn/metrics/_dist_metrics.pyx.tp:1866-1891` (`CanberraDistance.dist_csr`): the `ix1 == ix2` branch divides by `fabs(x1) + fabs(x2)` without the `denom > 0` check of `dist` (line 1841), and the `ix1 < ix2` / `ix1 > ix2` branches and both tail loops add `1.` unconditionally. They should add `1.` only if the stored value is non-zero (`|x| / |x| = 1`, `0 / 0` skipped). CSR/CSR without explicit zeros is correct, which is what `test_dist_metrics.py::test_cdist` exercises (its random data has no zeros, so the mixed formats never see a stored zero either).

## Versions

```
sklearn: 1.10.dev0, commit f570497251 (no change vs upstream/main 289c4d5cfa in the files involved)
python: 3.13, numpy: 2.5, scipy: 1.18, platform: linux
```
