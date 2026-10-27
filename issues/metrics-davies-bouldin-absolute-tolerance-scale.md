# `davies_bouldin_score` returns the perfect score 0.0 for small-scale data (absolute tolerance)

```json
{
    "estimators": ["davies_bouldin_score"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 1,
    "difficulty": 1,
    "decision": 1,
    "decision_tags": ["output-change"]
}
```

## Classification

- **Priority (1)**: silent and misleading (the best possible score), but only for data whose within-cluster distances are all below ~1e-8, i.e. features in tiny units. Rescaling the data is a workaround once you know.
- **Difficulty (1)**: replace the `isclose(..., 0)` shortcut by an exact-zero check, or a tolerance relative to the data scale.
- **Decision (1)**: changes the returned value for inputs that currently hit the shortcut by accident (`output-change`); the exactly-degenerate cases tested in `test_davies_bouldin_score` keep returning 0.0.

## Upstream search

Queries: "davies_bouldin_score", "davies_bouldin centroid", "davies_bouldin zero".

No match (#12611/#12760 and #28528 deal with other degenerate cases, see `metrics-davies-bouldin-coincident-centroids.md`). The shortcut comes from the original PR #10827 (merged 2018) and was ported in #32693 (merged, array API support) without change.

## Describe the bug and its user-facing impact

The Davies-Bouldin index is a ratio of distances, so it is invariant to rescaling `X`. `davies_bouldin_score` short-circuits to `0.0` when all intra-cluster distances, or all centroid distances, are "close to 0" according to `isclose` with its default absolute tolerance `atol=1e-8`. Data at a scale of ~1e-9 then gets the best possible score whatever the clustering, while `silhouette_score` and `calinski_harabasz_score` are unaffected.

## Steps/Code to Reproduce

```python
from sklearn.datasets import make_blobs
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score

X, y = make_blobs(200, random_state=0)
for scale in [1, 1e-6, 1e-9]:
    Xs = X * scale
    print(scale, davies_bouldin_score(Xs, y), silhouette_score(Xs, y),
          calinski_harabasz_score(Xs, y))
```

## Expected Results

`davies_bouldin_score` is the same (~0.730) for the three scales, like the two other scores.

## Actual Results

```
1 0.7301143325484819 0.47898225760335833 248.8702394851624
1e-06 0.7301143325484817 0.47898225760335833 248.8702394851624
1e-09 0.0 0.47898225760335833 248.87023948516233
```

## Root cause and possible fix

`sklearn/metrics/cluster/_unsupervised.py:477-480`:

```python
if xp.all(xpx.isclose(intra_dists, zero)) or xp.all(
    xpx.isclose(centroid_distances, zero)
):
    return 0.0
```

`isclose(a, 0)` reduces to `|a| <= atol = 1e-8`, an absolute threshold. The code before the array API port used `np.allclose(..., 0)`, with the same default `atol`, so this dates back to the original implementation (#10827). Checking for exact zeros (`xp.all(intra_dists == 0)`) keeps the two degenerate cases covered by `test_davies_bouldin_score` and makes the score scale-invariant. The centroid branch has a separate problem when centroids coincide, see `metrics-davies-bouldin-coincident-centroids.md`.

## Versions

```
sklearn: 1.10.dev0, commit f570497251 (no change vs upstream/main 289c4d5cfa in the files involved)
python: 3.13, numpy: 2.5, scipy: 1.18, platform: linux
```
