# MeanShift.predict ignores `cluster_all=False` and never returns the orphan label -1

```json
{
    "estimators": ["MeanShift"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 1,
    "difficulty": 1,
    "decision": 1,
    "decision_tags": ["output-change"]
}
```

## Classification

- **Priority (1)**: silent inconsistency between `fit(X).labels_` and `predict(X)`, but only with the non-default `cluster_all=False`, and users can apply the orphan rule themselves (`pairwise_distances_argmin_min` against `cluster_centers_` plus the bandwidth).
- **Difficulty (1)**: store the bandwidth used during `fit` (it is estimated when `bandwidth=None`) and apply the same `distance <= bandwidth` rule in `predict`.
- **Decision (1)**: `output-change`: `predict` starts returning `-1` with `cluster_all=False`. Storing the estimated bandwidth also means adding a fitted attribute (e.g. `bandwidth_`), a small API addition.

## Upstream search

Queries: "MeanShift cluster_all predict". No match (#22672, #2062, #6943 unrelated).

## Describe the bug and its user-facing impact

With `cluster_all=False`, the docstring says "orphans are given cluster label -1", where orphans are points that are not within `bandwidth` of any kernel. `fit` applies this rule to `labels_`, but `predict` always returns the nearest center (`pairwise_distances_argmin`) and ignores `cluster_all`. So `MeanShift(cluster_all=False).fit(X).predict(X)` differs from `fit_predict(X)`, and new points far from every cluster are silently assigned to a cluster instead of being flagged as orphans.

## Steps/Code to Reproduce

```python
import numpy as np
from sklearn.cluster import MeanShift
from sklearn.datasets import make_blobs

X, _ = make_blobs(n_samples=50, centers=[[0, 0], [10, 10]], cluster_std=0.3, random_state=0)
X = np.vstack([X, [[5, 5]]])  # an isolated point, far from both kernels

ms = MeanShift(bandwidth=2, bin_seeding=True, min_bin_freq=3, cluster_all=False).fit(X)
print("labels_ of the isolated point:  ", ms.labels_[-1])
print("predict for the same point:     ", ms.predict(X[-1:])[0])
print("fit_predict(X) == fit(X).predict(X):", np.array_equal(ms.labels_, ms.predict(X)))
```

## Expected Results

`predict` returns `-1` for the isolated point, as `labels_` does, so that `fit(X).predict(X)` equals `fit_predict(X)`.

## Actual Results

```
labels_ of the isolated point:   -1
predict for the same point:      1
fit_predict(X) == fit(X).predict(X): False
```

## Root cause and possible fix

`sklearn/cluster/_mean_shift.py`, `MeanShift.fit` applies `bool_selector = distances.flatten() <= bandwidth` when `cluster_all=False`, while `MeanShift.predict` only does `pairwise_distances_argmin(X, self.cluster_centers_)`. The bandwidth used in `fit` is a local variable (estimated with `estimate_bandwidth` when `bandwidth=None`), so `predict` cannot apply the rule today.

Fix: store the bandwidth used in `fit` as a fitted attribute (e.g. `bandwidth_`), and in `predict` use `pairwise_distances_argmin_min` and set the label to `-1` where the distance is larger than `bandwidth_` when `cluster_all=False`. If changing `predict` is not wanted, the docstring of `predict` should at least say that it ignores `cluster_all`.

## Versions

```
sklearn: 1.10.dev0, commit 289c4d5cfa (upstream/main; local checkout f570497251 has no change in sklearn/cluster/_mean_shift.py)
python: 3.13, numpy: 2.5, scipy: 1.18, platform: linux
```
