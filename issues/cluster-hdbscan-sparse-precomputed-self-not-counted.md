# HDBSCAN with a sparse precomputed distance matrix does not count the point itself in `min_samples`

```json
{
    "estimators": ["HDBSCAN"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 1,
    "difficulty": 1,
    "decision": 1,
    "decision_tags": ["output-change"]
}
```

## Classification

- **Priority (1)**: silent, but the sparse-precomputed result is exactly the dense result for `min_samples + 1`, a valid hyperparameter users rarely pick precisely anyway, and sparse precomputed input is a minority path; closer to a documentation/consistency issue than a wrong clustering. (priority review, 2026-10-23: was 2)
- **Difficulty (1)**: store an explicit zero diagonal before computing the mutual reachability graph, as `DBSCAN` and `OPTICS` already do.
- **Decision (1)**: `output-change`: sparse precomputed fits change (they become equal to dense ones). The "fewer than `min_samples` neighbors" check also becomes one neighbor less strict, which is what the docstring says anyway.

## Upstream search

Queries: "HDBSCAN precomputed sparse min_samples", "HDBSCAN sparse diagonal". No match. Related but different: #27436 (closed, misleading error when the sparse k-NN graph is disconnected), #31915 (closed, in-place modification of precomputed matrices), #27508 (closed, DBSCAN doc about modifying sparse precomputed matrices).

## Describe the bug and its user-facing impact

`min_samples` is documented as "the number of samples in a neighborhood for a point to be considered as a core point. This includes the point itself." On the dense paths (features, or dense precomputed matrix) the core distance is the distance to the `min_samples`-th neighbor counting the point itself (distance 0).

With a sparse precomputed matrix, the core distance is the `min_samples`-th smallest *stored* value of the row. The diagonal is usually not stored (`csr_matrix(D)` drops the zeros, `kneighbors_graph(..., mode="distance")` excludes self by default), so the point itself is not counted and HDBSCAN behaves as if `min_samples` was one larger. The same distances give different clusters depending on whether they are passed dense or sparse.

`DBSCAN` and `OPTICS` handle this by calling `X.setdiag(X.diagonal())` on sparse precomputed input "so each point is its own neighbor"; HDBSCAN does not.

## Steps/Code to Reproduce

```python
import numpy as np
from scipy import sparse
from sklearn.cluster import HDBSCAN
from sklearn.datasets import make_blobs
from sklearn.metrics import pairwise_distances

X, _ = make_blobs(n_samples=200, centers=4, cluster_std=1.5, random_state=0)
D = pairwise_distances(X)  # all pairwise distances, zeros on the diagonal

for name, X_fit in [("dense", D), ("sparse", sparse.csr_matrix(D))]:
    labels = HDBSCAN(metric="precomputed", min_samples=5, copy=True).fit(X_fit).labels_
    print(f"{name:>6}: n_clusters={labels.max() + 1}, n_noise={(labels == -1).sum()}")

# the sparse path behaves as if min_samples was one larger
labels = HDBSCAN(metric="precomputed", min_samples=6, copy=True).fit(D).labels_
print(f" dense, min_samples=6: n_clusters={labels.max() + 1}, n_noise={(labels == -1).sum()}")
```

## Expected Results

Same clustering for the dense and the sparse matrix, since they hold exactly the same distances (9 clusters and 124 noise points, as with the feature matrix `X`).

## Actual Results

```
 dense: n_clusters=9, n_noise=124
sparse: n_clusters=5, n_noise=99
 dense, min_samples=6: n_clusters=4, n_noise=95
```

I also checked with the private `_single_linkage_tree_` that the merge heights of the sparse fit with `min_samples=5` are exactly those of the dense fit with `min_samples=6` (the labels of the last line differ from the sparse ones only through tie-breaking in the MST). Storing an explicit zero diagonal in the sparse matrix gives the dense result back.

## Root cause and possible fix

`sklearn/cluster/_hdbscan/_reachability.pyx`, `_sparse_mutual_reachability_graph`: the core distance is `np.partition(row_data, min_samples - 1)[min_samples - 1]` over the stored entries of the row, which only include the point itself if the diagonal is stored. `HDBSCAN.fit` (`sklearn/cluster/_hdbscan/hdbscan.py`, `elif issparse(X):` branch) does not add the diagonal, unlike `DBSCAN.fit` (`sklearn/cluster/_dbscan.py`) and `OPTICS.fit` (`sklearn/cluster/_optics.py`).

Fix: in the sparse precomputed branch, make the diagonal explicit (`X.setdiag(X.diagonal())` on a copy, or inside the existing `copy` logic) and keep rows sorted. The "There exists points with fewer than {min_samples} neighbors" check in `_brute_mst` then counts the point itself too, matching the docstring. The self loops get weight `core_i` in the mutual reachability graph and are ignored by `minimum_spanning_tree`.

## Versions

```
sklearn: 1.10.dev0, commit 289c4d5cfa (upstream/main; local checkout f570497251 has no change in sklearn/cluster/_hdbscan)
python: 3.13, numpy: 2.5, scipy: 1.18, platform: linux
```
