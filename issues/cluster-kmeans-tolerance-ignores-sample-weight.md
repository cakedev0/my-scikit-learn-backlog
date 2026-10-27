# KMeans scales `tol` by the unweighted variance of X, so zero-weight samples change convergence

```json
{
    "estimators": ["KMeans", "k_means", "MiniBatchKMeans"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 1,
    "difficulty": 1,
    "decision": 1,
    "decision_tags": ["output-change"]
}
```

## Classification

- **Priority (1)**: silent early stop, but a large effect needs zero- or low-weight samples with a much larger spread than the weighted data. In the common case only the effective tolerance shifts a bit.
- **Difficulty (1)**: pass `sample_weight` to `_tolerance` and use a weighted variance (`np.average` / `mean_variance_axis(..., weights=...)`).
- **Decision (1)**: `output-change`: weighted fits get a different effective tolerance, so `n_iter_` and possibly the final clustering change for the same `random_state`.

## Upstream search

Queries: `KMeans tolerance sample_weight`, `kmeans _tolerance variance weighted`, `KMeans zero sample_weight`, `KMeans tol definition variance`. Related but different:

- #16298 (open meta-issue on `sample_weight` handling): `KMeans` is ticked there; the common `check_sample_weight_equivalence_on_*_data` checks are xfailed for `KMeans`/`MiniBatchKMeans` because of the stochastic init, which hides this deterministic difference.
- #22519 (open) and #34879 (open PR, docs): the `tol` docstring says "relative to the Frobenius norm" while the code scales by the mean variance; about the definition, not weights.
- `issues/cluster-bisecting-kmeans-tol-not-relative.md`: `BisectingKMeans` does not scale `tol` at all.

## Describe the bug and its user-facing impact

`KMeans` turns `tol` into an absolute threshold on the squared center shift with `_tolerance(X, tol) = tol * mean(var(X, axis=0))`. The variance is unweighted, so samples with `sample_weight=0` (which should be equivalent to removing them) and low-weight samples weigh as much as any other sample in the convergence threshold. With far away zero-weight points the threshold blows up and `fit` stops after the first iteration, returning a much worse clustering without any warning.

Zero weights are a common way to mask samples (e.g. when a meta-estimator or a CV helper passes weights instead of subsetting).

## Steps/Code to Reproduce

```python
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

X, _ = make_blobs(n_samples=300, centers=4, random_state=0)
init = X[np.random.RandomState(0).choice(len(X), 4, replace=False)]

# two extra points with zero weight: they should not change anything
X_extra = np.concatenate([X, [[1e4, 1e4], [-1e4, -1e4]]])
sw = np.r_[np.ones(len(X)), 0.0, 0.0]

ref = KMeans(4, init=init).fit(X)
km = KMeans(4, init=init).fit(X_extra, sample_weight=sw)
print("without the points:      n_iter_", ref.n_iter_, "inertia_", ref.inertia_)
print("with zero-weight points: n_iter_", km.n_iter_, "inertia_", km.inertia_)
```

## Expected Results

Same fit as without the two zero-weight points (`n_iter_=11`, `inertia_=541.45`), since `inertia_` and the centers only depend on the weighted samples.

## Actual Results

```
without the points:      n_iter_ 11 inertia_ 541.4519686901782
with zero-weight points: n_iter_ 1 inertia_ 1224.5614918852273
```

## Root cause and possible fix

`sklearn/cluster/_kmeans.py:283-292` computes `np.var(X, axis=0)` (or `mean_variance_axis(X, axis=0)` for sparse) without weights, and `_check_params_vs_input` (`:873`) calls it before `sample_weight` is even validated. `MiniBatchKMeans` uses the same `self._tol` for its early stopping (`:1992`, `:2172`).

Fix: validate `sample_weight` first and compute a weighted variance (`np.average((X - mean) ** 2, weights=sample_weight, axis=0)` for dense, `mean_variance_axis(X, axis=0, weights=sample_weight)` for sparse). With unit weights the result is unchanged.

## Versions

```
sklearn: 1.10.dev0, commit 289c4d5cfa
python: 3.13, numpy: 2.5.3, scipy: 1.18.1, platform: linux
```
