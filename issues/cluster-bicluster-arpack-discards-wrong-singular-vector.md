# SpectralCoclustering(svd_method="arpack") discards the smallest singular vector instead of the largest

```json
{
    "estimators": ["SpectralCoclustering", "SpectralBiclustering"],
    "existing_issue": "https://github.com/scikit-learn/scikit-learn/issues/12863",
    "reproduced": true,
    "priority": 2,
    "difficulty": 1,
    "decision": 1,
    "decision_tags": ["output-change"]
}
```

## Classification

- **Priority (2)**: silent wrong results with a documented option (`svd_method="arpack"`, described as "more accurate"), but not the default, and the biclustering estimators are not widely used. For `SpectralCoclustering` it can turn a good clustering into a bad one (consensus score 0.80 -> 0.12 below).
- **Difficulty (1)**: reverse the `svds` outputs to descending order before discarding, as `PCA` and `TruncatedSVD` already do. A couple of lines.
- **Decision (1)**: `output-change`: `svd_method="arpack"` results change (they become consistent with `svd_method="randomized"`).

## Upstream search

Queries: "SpectralCoclustering arpack", "bicluster svds".

- #12863 (open, 2018): exactly this bug, reported from code reading with a tiny example. No maintainer decision since.
- #12898 (open PR, last updated 2026-05-05): reverses the `svds` outputs. Still open, so the bug is still on main.

This file adds a reproducer with a clear quality gap (consensus score 0.80 vs 0.12) and points to the `eigsh` fallback that has the same problem plus a missing `k`.

## Describe the bug and its user-facing impact

`BaseSpectral._svd` returns "first `n_components` left and right singular vectors u and v, discarding the first `n_discard`". The first singular vector of the normalized matrix is the trivial one (constant after the `D^-1/2` scaling), which the Dhillon co-clustering algorithm drops.

`randomized_svd` returns singular vectors in decreasing order, so `u[:, n_discard:]` drops the largest one as intended. `scipy.sparse.linalg.svds` (used for `svd_method="arpack"`) returns them in *increasing* order, so with `"arpack"` the code drops the smallest singular vector and keeps the trivial one. For `SpectralCoclustering`, which only computes `1 + ceil(log2(n_clusters))` vectors, this removes one of the few informative vectors and keeps a useless constant one, so k-means works on fewer informative dimensions.

`SpectralBiclustering` (methods `"scale"` and `"bistochastic"`, `n_discard=1`) has the same off-by-one in which vector is dropped; it is much less visible there because it computes more vectors (`n_components=6`) and then picks the `n_best` most piecewise constant ones. I could not get a worse result for it on checkerboard data.

## Steps/Code to Reproduce

```python
import numpy as np
from sklearn.cluster import SpectralCoclustering
from sklearn.datasets import make_biclusters
from sklearn.metrics import consensus_score

data, rows, cols = make_biclusters(
    (200, 200), n_clusters=4, noise=5, minval=1, maxval=10, shuffle=True, random_state=0
)
data = np.abs(data)
for svd_method in ["randomized", "arpack"]:
    model = SpectralCoclustering(n_clusters=4, svd_method=svd_method, random_state=0)
    model.fit(data)
    score = consensus_score(model.biclusters_, (rows, cols))
    print(f"{svd_method:>10}: consensus score = {score:.3f}, "
          f"bicluster sizes = {np.bincount(model.row_labels_)}")
```

## Expected Results

Both SVD solvers compute the same singular subspace, so the two fits should give similar results (`"arpack"` is documented as the more accurate one).

## Actual Results

```
randomized: consensus score = 0.797, bicluster sizes = [50 50 49 51]
    arpack: consensus score = 0.119, bicluster sizes = [35 73 15 77]
```

Averaged over 5 seeds of the same setup: 0.739 (randomized) vs 0.314 (arpack) with 4 clusters, 0.451 vs 0.310 with 6 clusters. With 3 clusters both solvers give the same score. Monkeypatching `svds` in `sklearn.cluster._bicluster` to return its outputs in decreasing order gives 0.797 for `"arpack"` on the reproducer, the same as `"randomized"`.

## Root cause and possible fix

`sklearn/cluster/_bicluster.py:152`: `u, _, vt = svds(array, k=n_components, ncv=self.n_svd_vecs)` keeps scipy's increasing order, then lines 172-173 drop the first `n_discard` columns/rows. `sklearn/decomposition/_truncated_svd.py:229-232` and `sklearn/decomposition/_pca.py:708-711` reverse the `svds` outputs for this reason ("svds doesn't abide by scipy.linalg.svd/randomized_svd conventions, so reverse its outputs").

Fix: after `svds`, reverse `u` columns and `vt` rows (or sort by the returned singular values). The `eigsh` fallback just below has the same ordering problem (eigenvalues in increasing order) and also calls `eigsh` without `k=n_components`, so it returns 6 vectors whatever `n_components` is; it only runs when `svds` returns NaNs, but it should get the same treatment.

## Versions

```
sklearn: 1.10.dev0, commit 289c4d5cfa (upstream/main; local checkout f570497251 has no change in sklearn/cluster/_bicluster.py)
python: 3.13, numpy: 2.5, scipy: 1.18, platform: linux
```
