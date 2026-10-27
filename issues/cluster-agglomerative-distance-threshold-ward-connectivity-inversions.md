# AgglomerativeClustering(distance_threshold=...) returns clusters merged above the threshold with ward linkage and connectivity

```json
{
    "estimators": ["AgglomerativeClustering", "FeatureAgglomeration"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 2,
    "difficulty": 2,
    "decision": 1,
    "decision_tags": ["output-change"]
}
```

## Classification

- **Priority (2)**: silent and plausible-looking wrong result, on the canonical use of connectivity-constrained Ward (image segmentation with `grid_to_graph`, `FeatureAgglomeration` on images). Only with `distance_threshold`, not with `n_clusters`.
- **Difficulty (2)**: the cut must use the subtree maximum of the merge distances instead of the merge order, which needs a small new cutting routine (`_hc_cut` cuts by node id). Easy to test, but the change touches the labelling logic shared by all linkages.
- **Decision (1)**: `output-change` for ward + connectivity + `distance_threshold` fits only; other linkages and unstructured ward produce monotonic trees and would be unchanged.

## Upstream search

Queries: "distance_threshold connectivity ward", "ward connectivity non-monotonic", "distance_threshold monotonic".

Related: #23550 (open) "`distance_threshold` not respected in `AgglomerativeClustering` with sparse `connectivity`". Same symptom, different cause: it is about `complete`/`average` linkage, where only the distances of connected pairs are known, so the linkage distance of merged clusters is underestimated. A commenter notes `single` is fine and `ward` was not tested. Here the merge distances of ward are exact, but not monotonic, and the cut assumes they are. With a random chain connectivity I see inversions only for ward (24 of 99 merges), none for `complete`/`average`/`single`. A fix of the cut as proposed below would not fix #23550, and a fix of #23550 would not fix this.

## Describe the bug and its user-facing impact

The docstring says `distance_threshold` is "the linkage distance threshold at or above which clusters will not be merged". The implementation computes `n_clusters_ = count(distances_ >= distance_threshold) + 1` and then undoes the last `n_clusters_ - 1` merges (in merge order). This is only correct if the merge distances are non-decreasing in merge order.

With `linkage="ward"` and a `connectivity` matrix, they are not: merging two clusters can lower the Ward distance to a neighbor that was not adjacent before (an "inversion"). It happens a lot on grid graphs: on a random 20x20 image with `grid_to_graph`, 83 of the 399 merges have a smaller distance than the previous one. On a random 2D k-NN graph with `n_neighbors=3` I saw 3 inversions, with `n_neighbors=10` none.

As a result, the returned clusters can contain merges at or above the threshold, while some merges below the threshold are undone. Users get a plausible-looking segmentation with no warning, and `n_clusters_` is also off.

## Steps/Code to Reproduce

```python
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.image import grid_to_graph

rng = np.random.RandomState(0)
X = rng.rand(10 * 10, 1)  # a 10x10 random image, one pixel per sample
connectivity = grid_to_graph(10, 10)
threshold = 0.5

agg = AgglomerativeClustering(
    n_clusters=None, distance_threshold=threshold, connectivity=connectivity
).fit(X)
d = agg.distances_
print("merge distances non-decreasing:", np.all(np.diff(d) >= 0))

# Largest merge distance inside the subtree of each node of the tree.
n_samples = X.shape[0]
members = [{i} for i in range(n_samples)]
max_merge = [0.0] * n_samples
for (a, b), dist in zip(agg.children_, d):
    members.append(members[a] | members[b])
    max_merge.append(max(dist, max_merge[a], max_merge[b]))
worst = max(
    max_merge[members.index(set(np.flatnonzero(agg.labels_ == k)))]
    for k in range(agg.n_clusters_)
)
print(f"n_clusters_={agg.n_clusters_}, largest merge distance inside a "
      f"returned cluster: {worst:.3f} (distance_threshold={threshold})")
```

## Expected Results

Every returned cluster only contains merges with a distance below `0.5`. Cutting every node whose subtree contains a merge at or above the threshold gives 18 clusters here.

## Actual Results

```
merge distances non-decreasing: False
n_clusters_=14, largest merge distance inside a returned cluster: 0.650 (distance_threshold=0.5)
```

The last 15 merge distances are `[0.65 0.49 0.67 0.68 0.48 0.4 0.71 0.85 0.91 0.95 0.9 0.88 1.21 1.76 1.68]`: the merge at 0.65 is kept, while the merges at 0.49, 0.48 and 0.40 are undone.

## Root cause and possible fix

`sklearn/cluster/_agglomerative.py`, `AgglomerativeClustering._fit`:

```python
self.n_clusters_ = np.count_nonzero(self.distances_ >= distance_threshold) + 1
...
self.labels_ = _hc_cut(self.n_clusters_, self.children_, self.n_leaves_)
```

`_hc_cut` undoes the merges with the largest node ids, i.e. the last ones in merge order. The inversions come from the structured Ward tree builder (`ward_tree`, which computes the exact Ward distance from the moments for each new adjacent pair); they are inherent to connectivity-constrained Ward and not a bug of the tree itself.

Possible fix: when `distance_threshold` is set, compute for each node the maximum merge distance in its subtree (a cumulative max along `children_`, as in `scipy.cluster.hierarchy.fcluster(criterion="distance")`), and label each sample by its highest ancestor whose subtree maximum is below the threshold. `n_clusters_` is then the number of such roots. For monotonic trees (all other cases) this gives exactly the current result.

## Versions

```
sklearn: 1.10.dev0, commit 289c4d5cfa (upstream/main; local checkout f570497251 has no change in sklearn/cluster/_agglomerative.py)
python: 3.13, numpy: 2.5, scipy: 1.18, platform: linux
```
