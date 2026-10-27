# BisectingKMeans uses `tol` as an absolute tolerance, so results depend on the scale of X

```json
{
    "estimators": ["BisectingKMeans"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 2,
    "difficulty": 1,
    "decision": 1,
    "decision_tags": ["output-change"]
}
```

## Classification

- **Priority (2)**: silent, default settings: on data with a small scale (feature variance below ~1e-2, e.g. data in meters stored as kilometers, or normalized features with small spread), the inner k-means of each bisection stops after one or a few iterations and the clustering gets noticeably worse, with nothing telling the user. On data with a large scale it only costs extra iterations.
- **Difficulty (1)**: pass `self._tol` (already computed by `_check_params_vs_input`) instead of `self.tol` to `_kmeans_single`. One line.
- **Decision (1)**: `output-change`: results change for most datasets whose mean feature variance is not ~1 (more or fewer inner iterations). The change makes `BisectingKMeans` follow its docstring and `KMeans`.

## Upstream search

Queries: "BisectingKMeans tol", "BisectingKMeans tolerance", "BisectingKMeans sample_weight". Related:

- #22519 (open): the `tol` docstring of `KMeans` says "relative" while `_kmeans_single_*` compare an absolute value; it is about the KMeans docs (and a proposal to change the criterion), not about `BisectingKMeans` skipping the scaling.
- #34879 (open PR, docs only): rewrites the `tol` docstrings, and notices that `BisectingKMeans` passes the raw `tol`. It *documents* the absolute behavior instead of fixing it. I think fixing the code is better: k-means results should not depend on the units of `X`, and `self._tol` is already computed. Whoever handles one should look at the other.

## Describe the bug and its user-facing impact

The docstring of `BisectingKMeans` says `tol` is a "Relative tolerance with regards to Frobenius norm of the difference in the cluster centers of two consecutive iterations to declare convergence", like for `KMeans`. `KMeans` turns it into an absolute tolerance with `_tolerance(X, tol) = mean(var(X, axis=0)) * tol`. `BisectingKMeans` computes the same `self._tol` (in `_BaseKMeans._check_params_vs_input`) but never uses it: it passes the raw `self.tol` to the inner Lloyd/Elkan solver, which compares it to the absolute squared center shift.

So the convergence criterion depends on the units of `X`. Rescaling `X` by a constant changes the labels, which should not happen for k-means. With `X * 1e-3` (mean variance ~1e-5), `tol=1e-4` is larger than the first center shift and each bisection stops after the first iteration.

## Steps/Code to Reproduce

```python
import numpy as np
from sklearn.cluster import BisectingKMeans, KMeans
from sklearn.datasets import make_blobs

X, _ = make_blobs(n_samples=1000, centers=6, cluster_std=2.0, random_state=0)
bkm_ref = BisectingKMeans(n_clusters=6, random_state=0).fit(X)
km_ref = KMeans(n_clusters=6, random_state=0).fit(X)
for scale in [1e2, 1e-1, 1e-2, 1e-3]:
    bkm = BisectingKMeans(n_clusters=6, random_state=0).fit(X * scale)
    km = KMeans(n_clusters=6, random_state=0).fit(X * scale)
    print(f"scale={scale:g}: same labels as unscaled data? "
          f"BisectingKMeans: {np.array_equal(bkm.labels_, bkm_ref.labels_)}, "
          f"KMeans: {np.array_equal(km.labels_, km_ref.labels_)}; "
          f"BisectingKMeans inertia / scale**2 = {bkm.inertia_ / scale**2:.1f} "
          f"(unscaled: {bkm_ref.inertia_:.1f})")
```

## Expected Results

Labels invariant to a global rescaling of `X` (as for `KMeans`), and the normalized inertia staying at ~5163.

## Actual Results

```
scale=100: same labels as unscaled data? BisectingKMeans: True, KMeans: True; BisectingKMeans inertia / scale**2 = 5163.1 (unscaled: 5163.1)
scale=0.1: same labels as unscaled data? BisectingKMeans: False, KMeans: True; BisectingKMeans inertia / scale**2 = 5167.8 (unscaled: 5163.1)
scale=0.01: same labels as unscaled data? BisectingKMeans: False, KMeans: True; BisectingKMeans inertia / scale**2 = 5301.3 (unscaled: 5163.1)
scale=0.001: same labels as unscaled data? BisectingKMeans: False, KMeans: True; BisectingKMeans inertia / scale**2 = 6053.2 (unscaled: 5163.1)
```

At `scale=0.001` the inertia is ~17% worse than on the unscaled data. Passing `tol=np.var(X_scaled, axis=0).mean() * 1e-4` by hand (i.e. what `self._tol` holds) gives the same labels for all four scales.

## Root cause and possible fix

`sklearn/cluster/_bisect_k_means.py:339`, in `_bisect`: `tol=self.tol` should be `tol=self._tol`. `self._tol` is set by `_BaseKMeans._check_params_vs_input` (`sklearn/cluster/_kmeans.py`, `self._tol = _tolerance(X, self.tol)`), which `BisectingKMeans.fit` already calls. The variance is then computed on the whole `X`, as for `KMeans`, which seems the natural choice (a per-subcluster tolerance would also be defensible, but would differ from `KMeans`).

The line dates back to the original implementation (#20031, `0822851f5cb`), so this looks like an oversight rather than a choice.

## Versions

```
sklearn: 1.10.dev0, commit 289c4d5cfa (upstream/main; local checkout f570497251 has no change in sklearn/cluster/_bisect_k_means.py)
python: 3.13, numpy: 2.5, scipy: 1.18, platform: linux
```
