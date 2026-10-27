# HDBSCAN(store_centers=...) crashes when X contains non-finite values

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

- **Priority (1)**: loud crash, needs the combination of non-finite rows (a niche HDBSCAN feature) and the non-default `store_centers`; dropping the non-finite rows before `fit` is a trivial workaround. (priority review, 2026-10-23: was 2)
- **Difficulty (1)**: compute the centers on the full-size labels with the raw data, and exclude label `-3` from the cluster count. A few lines in `_weighted_cluster_center`.
- **Decision (0)**: it currently raises, any fix is an improvement.

## Upstream search

Queries: "HDBSCAN store_centers", "HDBSCAN centroids medoids", "HDBSCAN nan store_centers". No match. Related but different: #27893 (closed, medoids with `metric="precomputed"`, fixed by #27898 which now raises), #27876 (closed, docs of `centroids_`).

## Describe the bug and its user-facing impact

HDBSCAN accepts samples with missing (`np.nan`) or infinite values when `metric != "precomputed"`: they are left out of the clustering and labelled `-3` and `-2`. Combined with `store_centers="centroid"`, `"medoid"` or `"both"`, `fit` crashes with an `IndexError`.

`fit` remaps `labels_` and `probabilities_` back to the full `n_samples`, but then calls `self._weighted_cluster_center(X)` with `X` still reduced to the finite rows, so the boolean masks built from `labels_` have the wrong length.

There is a second bug in the same function, hidden by the crash: `n_clusters = len(set(self.labels_) - {-1, -2})` does not exclude the `-3` label of missing samples, so `centroids_`/`medoids_` would get one extra (uninitialized) row whenever a sample contains `np.nan`. The docstring says "the `-1, -2, -3` labels for the outlier clusters are excluded".

## Steps/Code to Reproduce

```python
import numpy as np
from sklearn.cluster import HDBSCAN
from sklearn.datasets import make_blobs

X, _ = make_blobs(n_samples=100, centers=[[0, 0], [10, 10]], cluster_std=0.5, random_state=0)
X[0, 0] = np.nan  # one sample with a missing value, labelled -3 by HDBSCAN

print(HDBSCAN(copy=True).fit(X).labels_[:3])
HDBSCAN(store_centers="centroid", copy=True).fit(X)
```

## Expected Results

`centroids_` of shape `(2, 2)` with the centroids of the two blobs, computed on the finite samples only.

## Actual Results

```
[-3  0  1]
Traceback (most recent call last):
  File "repro.py", line 9, in <module>
    HDBSCAN(store_centers="centroid", copy=True).fit(X)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^
  File "/home/arthur/scikit-learn/sklearn/base.py", line 1403, in wrapper
    return fit_method(estimator, *args, **kwargs)
  File "/home/arthur/scikit-learn/sklearn/cluster/_hdbscan/hdbscan.py", line 899, in fit
    self._weighted_cluster_center(X)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^
  File "/home/arthur/scikit-learn/sklearn/cluster/_hdbscan/hdbscan.py", line 953, in _weighted_cluster_center
    data = X[mask]
           ~^^^^^^
IndexError: boolean index did not match indexed array along axis 0; size of axis is 99 but size of corresponding boolean axis is 100
```

## Root cause and possible fix

- `sklearn/cluster/_hdbscan/hdbscan.py:899`: `self._weighted_cluster_center(X)` is called with the reduced `X = X[finite_index]` (line 772) while `self.labels_` was remapped to the raw size at lines 884-888.
- `sklearn/cluster/_hdbscan/hdbscan.py:939`: `n_clusters = len(set(self.labels_) - {-1, -2})` misses `-3`.

Fix: pass `self._raw_data` (or index the labels/probabilities with `finite_index`) to `_weighted_cluster_center`, and compute `n_clusters` as `self.labels_.max() + 1` (cluster labels are `0..n_clusters-1`), which ignores all negative labels. A regression test with a `np.nan` and a `np.inf` row and `store_centers="both"` would cover both. Note that `issues/cluster-hdbscan-keeps-training-data.md` proposes to drop `self._raw_data`; if both are fixed, compute the centers in `fit` from the raw `X` before it is reduced, or index the labels with `finite_index`.

## Versions

```
sklearn: 1.10.dev0, commit 289c4d5cfa (upstream/main; local checkout f570497251 has no change in sklearn/cluster/_hdbscan)
python: 3.13, numpy: 2.5, scipy: 1.18, platform: linux
```
