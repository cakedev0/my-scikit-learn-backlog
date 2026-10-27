# `davies_bouldin_score` treats clusters with the same centroid as perfectly separated

```json
{
    "estimators": ["davies_bouldin_score"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 1,
    "difficulty": 1,
    "decision": 2,
    "decision_tags": ["unclear-spec", "output-change"]
}
```

## Classification

- **Priority (1)**: silently gives the best possible score to the worst configuration, but exactly coincident centroids mostly happen on symmetric synthetic data (e.g. `make_circles(noise=None)`).
- **Difficulty (1)**: only set the diagonal of `centroid_distances` to `inf`, and drop or change the "all centroid distances are 0" shortcut.
- **Decision (2)**: `test_davies_bouldin_score` asserts 0.0 "when all the mean cluster are equal", so the current value is a deliberate convention, and the math gives `+inf` (the ratio `(s_i + s_j) / d_ij` with `d_ij = 0`). Someone needs to decide between `inf`, `nan` + warning, or keeping 0.0 for the fully degenerate case and only fixing the partial case (`unclear-spec`, `output-change`).

## Upstream search

Queries: "davies_bouldin_score", "davies_bouldin centroid", "davies_bouldin zero".

- #12611 (closed) reported the "divide by zero" RuntimeWarning of the diagonal; #12760 (merged) fixed it by setting all zero centroid distances to `inf`, without distinguishing the diagonal. The original code (#10827) turned `inf` ratios into `nan` and used `nanmax`, so off-diagonal coincident centroids have been ignored since the start, apparently as a side effect.
- #28528 (open, "Update cluster metrics to handle edge cases") returns `inf` for `n_labels in (1, n_samples)`, a different degenerate case.
- I read the review comments of #10827: the 0.0 return for equal centroids is tested but not discussed there.

## Describe the bug and its user-facing impact

`davies_bouldin_score` sets every zero entry of the centroid distance matrix to `inf` to skip the diagonal. Off-diagonal zeros (two clusters with the same centroid) are skipped too, so a pair of fully overlapping clusters contributes a ratio of 0, the best value, instead of the largest one. When all centroids coincide, the function returns `0.0` directly.

The result is discontinuous: moving one cluster by 1e-3 turns a "perfect" 0.0 into a score in the thousands.

## Steps/Code to Reproduce

```python
import numpy as np
from sklearn.datasets import make_circles
from sklearn.metrics import davies_bouldin_score

# Two concentric rings: both cluster centroids are at the origin.
X, y = make_circles(n_samples=200, noise=None, factor=0.5, random_state=0)
print("concentric rings:", davies_bouldin_score(X, y))
X[:, 0] += 1e-3 * y  # shift the inner ring by 1e-3
print("inner ring shifted by 1e-3:", davies_bouldin_score(X, y))

# Three clusters, the first two share their centroid (0, 0).
X = np.array([[-1, 0], [1, 0], [0, -2], [0, 2], [10, 0], [11, 0]], dtype=float)
y = [0, 0, 1, 1, 2, 2]
print("3 clusters, 2 coincident centroids:", davies_bouldin_score(X, y))
X[2:4, 0] += 1e-3
print("same, shifted by 1e-3:", davies_bouldin_score(X, y))
```

## Expected Results

Coincident centroids are the worst case for Davies-Bouldin (lower is better), so the score should be very large (`inf` by the formula), not the best possible value 0.0, and close to the score of the slightly shifted configurations.

## Actual Results

```
concentric rings: 0.0
inner ring shifted by 1e-3: 1500.0000000000362
3 clusters, 2 coincident centroids: 0.20634920634920637
same, shifted by 1e-3: 2000.079372638664
```

## Root cause and possible fix

`sklearn/metrics/cluster/_unsupervised.py:477-482`: the shortcut returning 0.0 when all centroid distances are close to 0, and

```python
centroid_distances[centroid_distances == 0] = xp.inf
```

which is meant for the diagonal only. Setting only the diagonal to `inf` gives `inf` for the partial case. The fully degenerate case is the one asserted by the test, so changing it is a decision for maintainers. The absolute tolerance used by the same shortcut is a separate problem, see `metrics-davies-bouldin-absolute-tolerance-scale.md`.

Side note: in the same test, the two "General case" checks are `pytest.approx(davies_bouldin_score(X, labels), ...)` without `assert`, so they check nothing.

## Versions

```
sklearn: 1.10.dev0, commit f570497251 (no change vs upstream/main 289c4d5cfa in the files involved)
python: 3.13, numpy: 2.5, scipy: 1.18, platform: linux
```
