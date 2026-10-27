# KMeans can put an empty cluster's center far outside the data (copies an unnormalized sum)

```json
{
    "estimators": ["KMeans", "k_means", "BisectingKMeans"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 1,
    "difficulty": 1,
    "decision": 0,
    "decision_tags": []
}
```

## Classification

- **Priority (1)**: silent, but only the center of a cluster that stays empty after relocation is wrong (labels and inertia are fine in the simple case), the triggers are degenerate (more clusters than distinct points, zero-weight far points) and already raise a `ConvergenceWarning`; same trigger family as #34074, which maintainers called a very edge case. (priority review, 2026-10-23: was 2)
- **Difficulty (1)**: read the biggest cluster's center after it has been averaged (e.g. normalize in a first loop, fill empty clusters in a second one).
- **Decision (0)**: only changes the buggy output, no trade-off.

## Upstream search

Queries: `_average_centers`, `KMeans empty cluster center outside`, `KMeans cluster_centers_ duplicate points wrong`, `kmeans empty clusters relocation`. Related but different:

- #28055 (closed, infinite loop when relocating empty clusters) and its fix #28165 (merged), which introduced the "place empty clusters at the biggest cluster" code that has this bug. Its test `test_relocating_with_duplicates` has the empty cluster at a higher index than the biggest cluster, so it does not hit the bug.
- #34074 (open, stale `labels_` after relocation because strict convergence skips the last E-step). Different root cause, but both can combine (see below).

## Describe the bug and its user-facing impact

After the M-step, `_average_centers` divides each center's weighted sum by its weight. For a cluster with zero weight, it copies the center of the biggest cluster instead ("we avoid setting empty clusters at the origin but place them at the location of the biggest cluster"). But `argmax_weight` is computed before the loop and the copy happens in the same loop as the normalization. If the empty cluster has a lower index than the biggest one, it gets the biggest cluster's *weighted sum* (not yet divided by its weight), i.e. a point roughly `n_samples_in_biggest_cluster` times farther from the mean than the real center.

The resulting center lies far outside the data. No sample is assigned to it, so `labels_` and `inertia_` are fine in the simple case, but `cluster_centers_` and `transform` are wrong. Combined with #34074 (strict convergence right after a relocation), I also saw `labels_` pointing to such a center, with `inertia_` of 1164 where 0 was reachable.

A cluster stays empty after relocation when:

- there are more clusters than distinct points (relocation is skipped since `max(distances) == 0`), which can happen with `init="random"` on data with duplicates;
- the farthest points picked for relocation have `sample_weight=0` (the relocated center gets a zero weight again);
- relocation takes all the points of another cluster.

## Steps/Code to Reproduce

```python
import numpy as np
from sklearn.cluster import KMeans

X = np.array([[0.0], [0.0], [10.0], [10.0], [10.0]])

# cluster 0 gets no sample; the biggest cluster (2) has a higher index
init = np.array([[100.0], [0.0], [10.0]])
for algorithm in ["lloyd", "elkan"]:
    km = KMeans(n_clusters=3, init=init, algorithm=algorithm).fit(X)
    print(algorithm, "cluster_centers_:", km.cluster_centers_.ravel())

# same thing with init="random" when it draws the duplicate point twice
km = KMeans(n_clusters=3, init="random", n_init=1, random_state=21).fit(X)
print("random init cluster_centers_:", km.cluster_centers_.ravel())
print("transform(X[:1]):", km.transform(X[:1]))
```

With a zero-weight outlier and many distinct points:

```python
import numpy as np
from sklearn.cluster import KMeans

rng = np.random.RandomState(0)
X = np.concatenate([rng.randn(20, 1), rng.randn(30, 1) + 10, [[50.0]]])
sw = np.ones(len(X))
sw[-1] = 0  # zero-weight outlier
init = np.array([[-100.0], [0.0], [10.0]])
km = KMeans(3, init=init).fit(X, sample_weight=sw)
print(km.cluster_centers_.ravel(), np.bincount(km.labels_, minlength=3))
```

## Expected Results

Empty clusters placed at the biggest cluster's center, as the code comment says: `[10, 0, 10]` in the first example, and a center at ~9.85 instead of 92.6 in the second (data range is about [-2.6, 50]).

## Actual Results

```
lloyd cluster_centers_: [18.  0. 10.]
elkan cluster_centers_: [18.  0. 10.]
random init cluster_centers_: [ 0. 18. 10.]
transform(X[:1]): [[ 0. 18. 10.]]
```

(plus `ConvergenceWarning: Number of distinct clusters (2) found smaller than n_clusters (3)` for each fit), and for the second snippet:

```
[92.62537088  0.56933459  9.85470906] [ 0 20 31]
```

18 is `mean(X) + 3 * (10 - mean(X))`: the sum of the 3 centered points at 10, not their mean.

## Root cause and possible fix

`sklearn/cluster/_k_means_common.pyx:284-295` (commit below):

```cython
int argmax_weight = np.argmax(weight_in_clusters)

for j in range(n_clusters):
    if weight_in_clusters[j] > 0:
        alpha = 1.0 / weight_in_clusters[j]
        for k in range(n_features):
            centers[j, k] *= alpha
    else:
        for k in range(n_features):
            centers[j, k] = centers[argmax_weight, k]
```

When `j < argmax_weight`, `centers[argmax_weight]` is still the weighted sum. Fix: normalize all non-empty clusters first, then copy into the empty ones (or divide by `weight_in_clusters[argmax_weight]` when copying). Used by both Lloyd and Elkan, dense and sparse, so `KMeans`, `k_means` and `BisectingKMeans` are affected.

Side note: `_relocate_empty_clusters_{dense,sparse}` (`_k_means_common.pyx:210`, `:270`) pick the farthest points regardless of their weight, so a zero-weight point is a useless relocation target. Skipping zero-weight points there would avoid the second trigger, but the `_average_centers` fix is enough to get sensible centers.

## Versions

```
sklearn: 1.10.dev0, commit 289c4d5cfa
python: 3.13, numpy: 2.5.3, scipy: 1.18.1, platform: linux
```
