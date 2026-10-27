# BisectingKMeans(n_clusters=1) ignores sample_weight for the cluster center

```json
{
    "estimators": ["BisectingKMeans"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 1,
    "difficulty": 1,
    "decision": 0,
    "decision_tags": []
}
```

## Classification

- **Priority (1)**: silent wrong center and inertia, but only for `n_clusters=1`, which is mostly hit when sweeping `n_clusters` (elbow plots, grid searches) with sample weights.
- **Difficulty (1)**: use the weighted mean for the root node, one line.
- **Decision (0)**: no trade-off, only `n_clusters=1` fits with non-uniform weights change.

## Upstream search

Queries: "BisectingKMeans sample_weight". Related: #16298 (open meta-issue on `sample_weight`), where `BisectingKMeans` is ticked as handled by a statistical test; that test does not cover `n_clusters=1`. #25752 (merged) made the k-means init weight aware, unrelated to the root center.

## Describe the bug and its user-facing impact

With `n_clusters=1`, `BisectingKMeans` never bisects, so `cluster_centers_[0]` is the center of the root node of the bisecting tree, which is the unweighted mean of `X`. `sample_weight` is ignored, and `inertia_` is computed around this wrong center. `KMeans(n_clusters=1)` and `BisectingKMeans(n_clusters=1)` on the repeated samples both give the weighted mean.

## Steps/Code to Reproduce

```python
import numpy as np
from sklearn.cluster import BisectingKMeans, KMeans

rng = np.random.RandomState(0)
X = rng.rand(20, 2)
sample_weight = rng.randint(0, 5, 20).astype(float)

bkm = BisectingKMeans(n_clusters=1, random_state=0).fit(X, sample_weight=sample_weight)
km = KMeans(n_clusters=1, random_state=0).fit(X, sample_weight=sample_weight)
print("weighted mean:  ", np.average(X, axis=0, weights=sample_weight))
print("BisectingKMeans:", bkm.cluster_centers_[0], "inertia:", bkm.inertia_)
print("KMeans:         ", km.cluster_centers_[0], "inertia:", km.inertia_)

# sample_weight vs repeated samples
X_rep = np.repeat(X, sample_weight.astype(int), axis=0)
print("repeated samples:", BisectingKMeans(n_clusters=1).fit(X_rep).cluster_centers_[0])
```

## Expected Results

`cluster_centers_[0]` equal to the weighted mean `[0.4299 0.6595]` and `inertia_` equal to the `KMeans` one (5.936).

## Actual Results

```
weighted mean:   [0.42991738 0.65949775]
BisectingKMeans: [0.48622879 0.66317155] inertia: 6.082887479159266
KMeans:          [0.42991738 0.65949775] inertia: 5.9364017730080665
repeated samples: [0.42991738 0.65949775]
```

## Root cause and possible fix

`sklearn/cluster/_bisect_k_means.py:421`: the root `_BisectingTree` node is created with `center=X.mean(axis=0)`. For `n_clusters >= 2` the root center is never used for the output (leaf centers come from the weighted k-means of each bisection), so only `n_clusters=1` is affected. Fix: `center=np.average(X, axis=0, weights=sample_weight)` (with a sparse-compatible equivalent, e.g. `sample_weight @ X / sample_weight.sum()`, since `X.mean(axis=0)` is also used for CSR input).

## Versions

```
sklearn: 1.10.dev0, commit 289c4d5cfa (upstream/main; local checkout f570497251 has no change in sklearn/cluster/_bisect_k_means.py)
python: 3.13, numpy: 2.5, scipy: 1.18, platform: linux
```
