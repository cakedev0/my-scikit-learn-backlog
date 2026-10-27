# HDBSCAN ignores `alpha` with `algorithm="brute"` (precomputed, sparse and non-tree metrics)

```json
{
    "estimators": ["HDBSCAN"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 2,
    "difficulty": 1,
    "decision": 1,
    "decision_tags": ["output-change"]
}
```

## Classification

- **Priority (2)**: silent: `alpha` has no effect on the labels whenever the brute path is used, which is the only path for `metric="precomputed"`, sparse input and metrics like `"cosine"`. `alpha` is a non-default, rarely tuned parameter, which keeps it from being a 3.
- **Difficulty (1)**: compute the core distances before dividing by `alpha` (or only scale the pairwise term), a few lines in `_hdbscan_brute`.
- **Decision (1)**: `output-change`: results of brute fits with `alpha != 1` change, but they become consistent with the tree-based algorithms and with the definition of robust single linkage.

## Upstream search

Queries: "HDBSCAN alpha", "HDBSCAN brute alpha" (issues and PRs). No match. Hits were unrelated (#31915 about `copy` with precomputed matrices, #24857 linkage cleanup, #22616 the original port). `git log upstream/main -- sklearn/cluster/_hdbscan` shows no related fix.

## Describe the bug and its user-facing impact

`HDBSCAN(alpha=...)` gives different results depending on `algorithm`. With `"kd_tree"`/`"ball_tree"` (Prim's path), the mutual reachability distance is `max(core_i, core_j, d(i, j) / alpha)`, as in robust single linkage. With `"brute"`, the whole distance matrix is divided by `alpha` *before* the core distances are computed, so the core distances are also divided by `alpha`. All mutual reachability distances are then scaled by the same factor `1 / alpha`, which leaves the cluster hierarchy, and hence the labels, unchanged.

So on the brute path `alpha` is silently a no-op (it only rescales the distance axis, which shifts the meaning of `cluster_selection_epsilon` and of `dbscan_clustering(cut_distance)`). The brute path is used with `algorithm="auto"` for `metric="precomputed"`, sparse `X`, and any metric not supported by KDTree/BallTree (e.g. `"cosine"`), so users there have no way to use `alpha`.

## Steps/Code to Reproduce

```python
import numpy as np
from sklearn.cluster import HDBSCAN
from sklearn.datasets import make_blobs

X, _ = make_blobs(n_samples=200, centers=3, cluster_std=1.5, random_state=0)
for algorithm in ["kd_tree", "brute"]:
    for alpha in [1.0, 0.5, 2.0]:
        labels = HDBSCAN(alpha=alpha, algorithm=algorithm, copy=True).fit(X).labels_
        print(f"{algorithm:>8}, alpha={alpha}: n_clusters={labels.max() + 1}, "
              f"n_noise={(labels == -1).sum()}")
```

## Expected Results

The same labels for `"kd_tree"` and `"brute"` for each `alpha` (the docstring says `algorithm` only selects how core distances are computed), and `alpha` changing the result on both paths.

## Actual Results

```
 kd_tree, alpha=1.0: n_clusters=6, n_noise=120
 kd_tree, alpha=0.5: n_clusters=2, n_noise=13
 kd_tree, alpha=2.0: n_clusters=3, n_noise=133
   brute, alpha=1.0: n_clusters=6, n_noise=120
   brute, alpha=0.5: n_clusters=6, n_noise=120
   brute, alpha=2.0: n_clusters=6, n_noise=120
```

## Root cause and possible fix

`sklearn/cluster/_hdbscan/hdbscan.py:254` (`_hdbscan_brute`) does `distance_matrix /= alpha` and then calls `mutual_reachability_graph`, which computes the core distances from the already scaled matrix (`sklearn/cluster/_hdbscan/_reachability.pyx`, `np.partition(distance_matrix, ...)`). The Prim's path (`sklearn/cluster/_hdbscan/_linkage.pyx:189`, `mst_from_data_matrix`) only divides the pairwise distance and uses unscaled core distances.

IIRC the original `hdbscan` library's `mutual_reachability` also computed the core distances before dividing by `alpha`, so this looks like a porting slip.

Fix: compute the core distances on the unscaled matrix, then take `max(core_i, core_j, d_ij / alpha)`. The simplest way is to pass `alpha` down to `mutual_reachability_graph` (dense and sparse kernels) and apply it to the pairwise term only, instead of dividing the matrix up front. For the sparse path, `max_distance` handling should stay as is.

## Versions

```
sklearn: 1.10.dev0, commit 289c4d5cfa (upstream/main; local checkout f570497251 has no change in sklearn/cluster/_hdbscan)
python: 3.13, numpy: 2.5, scipy: 1.18, platform: linux
```
