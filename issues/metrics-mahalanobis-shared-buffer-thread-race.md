# Mahalanobis neighbors give wrong distances and NaNs with `n_jobs > 1` (shared work buffer)

```json
{
    "estimators": ["NearestNeighbors", "LocalOutlierFactor", "KNeighborsClassifier", "KNeighborsRegressor", "BallTree", "DistanceMetric"],
    "existing_issue": "https://github.com/scikit-learn/scikit-learn/issues/32753",
    "reproduced": true,
    "priority": 3,
    "difficulty": 2,
    "decision": 1,
    "decision_tags": ["perf-low"]
}
```

## Classification

- **Priority (3)**: silent wrong neighbors, NaN distances and different predictions with a plain `n_jobs=4` on a documented metric, under the default `algorithm="auto"` (which picks `ball_tree` for mahalanobis). Users get no error, just a RuntimeWarning from `sqrt` if the race happens to produce a negative value.
- **Difficulty (2)**: the fix is local to `MahalanobisDistance` (stop using an instance-level scratch buffer in `rdist`/`rdist_csr`), but it is Cython code called in `nogil` hot loops, so the replacement (per-call stack/heap buffer or computing the differences on the fly) needs some care.
- **Decision (1)**: computing `x1 - x2` on the fly costs O(d^2) subtractions instead of O(d), and a per-call `malloc` has its own cost, so a small slowdown of the single-threaded case is possible (`perf-low`). Results are unchanged for `n_jobs=1`.

## Upstream search

Queries: "mahalanobis n_jobs", "mahalanobis thread", "mahalanobis sparse".

- #32753 (open, "LocalOutlierFactor with Mahalanobis distance returns different results based on `n_jobs` parameter"): same root cause, reported through `LocalOutlierFactor`. This file adds that it affects every tree-based neighbors estimator under `algorithm="auto"` (`KNeighborsClassifier` predictions change), not only LOF.
- #34671 (open, draft, by @lesteve): "FIX Fix thread-safety of Mahalanobis distance computation", removes `self.buffer` from the dense and sparse paths. Earlier attempts #34666 and #33188 were closed.
- #12672 (closed, 2018, "KNeighborsRegressor gives different results for different n_jobs values") was about `seuclidean` with brute force (V estimated per chunk), fixed by #12701. #12337 (closed) was about mahalanobis not using several cores, not about wrong results.

## Describe the bug and its user-facing impact

`MahalanobisDistance` stores `x1 - x2` in `self.buffer`, an array owned by the metric object. `BallTree.query` and `query_radius` run their inner loop with the GIL released, and `NearestNeighbors.kneighbors` / `radius_neighbors` dispatch the query chunks to threads (`Parallel(n_jobs, prefer="threads")`) that all share the same tree, hence the same metric object and the same buffer. Concurrent threads overwrite each other's coordinates mid-computation.

With `metric="mahalanobis"` and `n_jobs > 1`, any estimator built on the neighbors trees (`NearestNeighbors`, `KNeighborsClassifier/Regressor`, `RadiusNeighbors*`, `DBSCAN`, ...) silently returns wrong neighbors, wrong (even NaN) distances, and different predictions than with `n_jobs=1`. The race is non-deterministic, so exact numbers change from run to run.

## Steps/Code to Reproduce

```python
import numpy as np
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors

rng = np.random.RandomState(0)
X = rng.randn(4000, 8)
y = (X[:, 0] > 0).astype(int)
VI = np.linalg.inv(np.cov(X.T))

nn = NearestNeighbors(n_neighbors=5, metric="mahalanobis", metric_params={"VI": VI})
d1, i1 = nn.set_params(n_jobs=1).fit(X).kneighbors(X)
d4, i4 = nn.set_params(n_jobs=4).fit(X).kneighbors(X)
print("algorithm:", nn._fit_method)
print("NaN distances with n_jobs=4:", np.isnan(d4).sum())
print("rows with different neighbors:", (i4 != i1).any(axis=1).sum(), "/", len(X))

clf = KNeighborsClassifier(metric="mahalanobis", metric_params={"VI": VI})
p1 = clf.set_params(n_jobs=1).fit(X, y).predict(X)
p4 = clf.set_params(n_jobs=4).fit(X, y).predict(X)
print("different predictions:", (p1 != p4).sum(), "/", len(X))
```

## Expected Results

`n_jobs` only changes the speed: same neighbors, distances and predictions as with `n_jobs=1`, and no NaN.

## Actual Results

```
/home/arthur/scikit-learn/sklearn/utils/parallel.py:184: RuntimeWarning: invalid value encountered in sqrt
(repeated 8 times)
algorithm: ball_tree
NaN distances with n_jobs=4: 4664
rows with different neighbors: 2150 / 4000
different predictions: 713 / 4000
```

## Root cause and possible fix

`sklearn/metrics/_dist_metrics.pyx.tp:1598` declares `cdef float64_t[::1] buffer` on `MahalanobisDistance`, allocated once in `__init__` (line 1614). `rdist` (line 1625) and `rdist_csr` (line 1665) write into `self.buffer` and read it back. The callers in `sklearn/neighbors/_binary_tree.pxi.tp:1241` (`with nogil:` around `_query_single_depthfirst`) and `sklearn/neighbors/_base.py:910` (`Parallel(n_jobs, prefer="threads")` over `self._tree.query`) make this object shared between threads running without the GIL.

The buffer has been an instance attribute for a long time (it used to be `self.vec`, renamed in #25553). Mahalanobis is also excluded from the `pairwise_distances_reduction` backend (`_dispatcher.py`, "is numerically unstable"); this race may be part of why.

Possible fixes: compute `(x1[i] - x2[i]) * mat[i, j] * (x1[j] - x2[j])` without a scratch buffer, or allocate a small buffer per call (stack for small `size`, `malloc` otherwise). The sparse path additionally needs the buffer zeroed per call (see `metrics-mahalanobis-csr-stale-buffer.md`).

## Versions

```
sklearn: 1.10.dev0, commit f570497251 (no change vs upstream/main 289c4d5cfa in the files involved)
python: 3.13, numpy: 2.5, scipy: 1.18, platform: linux
```
