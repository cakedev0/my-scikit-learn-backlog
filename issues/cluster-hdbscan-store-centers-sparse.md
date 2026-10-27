# HDBSCAN(store_centers=...) crashes on sparse input

```json
{
    "estimators": ["HDBSCAN"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 1,
    "difficulty": 1,
    "decision": 0,
    "decision_tags": []
}
```

## Classification

- **Priority (1)**: loud crash with an unhelpful numpy error, on sparse input combined with a non-default option. Workaround: densify `X` or compute the centers yourself.
- **Difficulty (1)**: use sparse-aware operations (weighted mean via `strength @ data / strength.sum()`, densify the selected medoid row) or raise a clear error. A few lines.
- **Decision (0)**: it currently raises, any fix is an improvement.

## Upstream search

Queries: "HDBSCAN store_centers", "HDBSCAN centroids medoids", "HDBSCAN sparse store_centers". No match. Related but different: #27893 (closed, medoids with `metric="precomputed"`, fixed by #27898).

## Describe the bug and its user-facing impact

HDBSCAN accepts sparse feature matrices (`accept_sparse=["csr", "lil"]`, and the docstring of `fit` says `X : {array-like, sparse matrix}`), and `store_centers` is not documented as dense only. With a sparse `X`, `store_centers="centroid"` fails with a numpy `AxisError`, and `store_centers="medoid"` fails with `ValueError: setting an array element with a sequence.`. Neither message tells the user that sparse input is the problem.

## Steps/Code to Reproduce

```python
import numpy as np
from scipy import sparse
from sklearn.cluster import HDBSCAN
from sklearn.datasets import make_blobs

X, _ = make_blobs(n_samples=100, centers=[[0, 0], [10, 10]], cluster_std=0.5, random_state=0)
X = sparse.csr_matrix(X)

print(HDBSCAN(copy=True).fit(X).labels_[:3])
for store_centers in ["centroid", "medoid"]:
    try:
        HDBSCAN(store_centers=store_centers, copy=True).fit(X)
    except Exception as e:
        print(f"{store_centers}: {type(e).__name__}: {e}")
```

## Expected Results

`centroids_` / `medoids_` computed as for the dense matrix (or, at least, a clear error saying sparse input is not supported with `store_centers`).

## Actual Results

```
[0 1 0]
centroid: AxisError: axis: axis 0 is out of bounds for array of dimension 0
medoid: ValueError: setting an array element with a sequence.
```

## Root cause and possible fix

`sklearn/cluster/_hdbscan/hdbscan.py`, `_weighted_cluster_center`: `np.average(data, weights=strength, axis=0)` (line 956) does not support a scipy sparse `data`, and `self.medoids_[idx] = data[medoid_index]` (line 964) assigns a sparse row into a dense array.

Fix: for sparse `data`, compute the centroid as `np.asarray(strength @ data).ravel() / strength.sum()` (or `safe_sparse_dot`), and densify the medoid row with `.toarray().ravel()`. `pairwise_distances` already supports sparse input for the medoid search.

## Versions

```
sklearn: 1.10.dev0, commit 289c4d5cfa (upstream/main; local checkout f570497251 has no change in sklearn/cluster/_hdbscan)
python: 3.13, numpy: 2.5, scipy: 1.18, platform: linux
```
