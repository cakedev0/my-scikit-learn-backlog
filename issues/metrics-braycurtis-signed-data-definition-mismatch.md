# `metric="braycurtis"` gives different distances in `pairwise_distances` and `NearestNeighbors` for negative values

```json
{
    "estimators": ["pairwise_distances", "NearestNeighbors", "DistanceMetric", "pairwise_distances_argmin_min"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 1,
    "difficulty": 1,
    "decision": 2,
    "decision_tags": ["unclear-spec", "output-change"]
}
```

## Classification

- **Priority (1)**: silent inconsistency between two public entry points for the same metric name, but Bray-Curtis is meant for non-negative data (counts, abundances), where both definitions agree.
- **Difficulty (1)**: a one-line change in `BrayCurtisDistance.dist`/`dist_csr` (or in the docs).
- **Decision (2)**: someone has to pick a definition. Following scipy (`sum|x - y| / sum|x + y|`) makes `pairwise_distances` and the neighbors code agree, but changes `NearestNeighbors`/`BallTree` results on signed data and can divide by 0 or exceed 1; keeping sklearn's `sum(|x| + |y|)` (bounded in [0, 1], no division by 0 for non-zero rows) means documenting the difference or not routing `pairwise_distances` to scipy (`unclear-spec`, `output-change`).

## Upstream search

Queries: "braycurtis", "braycurtis sparse".

No match. #25532 (closed, "`pairwise_distances` is inconsistent with `scipy.spatial.distance` when using `metric=\"matching\"`") is the same kind of sklearn vs scipy definition mismatch for another metric; #22427 (open) is the general metrics API discussion.

## Describe the bug and its user-facing impact

`pairwise_distances(metric="braycurtis")` calls `scipy.spatial.distance.cdist`, which computes `sum|x_i - y_i| / sum|x_i + y_i|`. `DistanceMetric` (used by `NearestNeighbors`, `BallTree`, the `ArgKmin`/`RadiusNeighbors` reductions, `pairwise_distances_argmin(_min)`) computes `sum|x_i - y_i| / sum(|x_i| + |y_i|)`, as its docstring says. Both agree for non-negative data and differ as soon as a feature is negative (e.g. standardized data), so the same metric name gives different distances and neighbor rankings depending on the function used.

## Steps/Code to Reproduce

```python
import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors

X = np.array([[1.0, -1.0]])
Y = np.array([[2.0, 1.0], [-1.0, -1.0]])
print("pairwise_distances:", pairwise_distances(X, Y, metric="braycurtis"))
print("NearestNeighbors  :", NearestNeighbors(n_neighbors=2, metric="braycurtis").fit(Y).kneighbors(X))
print("NearestNeighbors(algorithm='ball_tree'):",
      NearestNeighbors(n_neighbors=2, metric="braycurtis", algorithm="ball_tree").fit(Y).kneighbors(X))
```

## Expected Results

The same distances from both functions.

## Actual Results

```
pairwise_distances: [[1. 1.]]
NearestNeighbors  : (array([[0.5, 0.6]]), array([[1, 0]]))
NearestNeighbors(algorithm='ball_tree'): (array([[0.5, 0.6]]), array([[1, 0]]))
```

(Distances `[0.6, 0.5]` for rows `[0, 1]` with sklearn's definition, `[1, 1]` with scipy's.)

## Root cause and possible fix

`sklearn/metrics/_dist_metrics.pyx.tp:1897-1920` (`BrayCurtisDistance`, `denom += fabs(x1[j]) + fabs(x2[j])`) vs scipy's `braycurtis` (`l1_sum = abs(u + v)`), reached from `sklearn/metrics/pairwise.py:2451-2453`. `test_dist_metrics.py` compares the two on `random_sample` data, which is non-negative, so the difference never shows. Options are listed under Decision.

## Versions

```
sklearn: 1.10.dev0, commit f570497251 (no change vs upstream/main 289c4d5cfa in the files involved)
python: 3.13, numpy: 2.5, scipy: 1.18, platform: linux
```
