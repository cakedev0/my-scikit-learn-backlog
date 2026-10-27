# `haversine_distances` on sparse input swaps latitude and longitude for some rows

```json
{
    "estimators": ["haversine_distances", "pairwise_distances", "DistanceMetric"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 1,
    "difficulty": 1,
    "decision": 0,
    "decision_tags": []
}
```

## Classification

- **Priority (1)**: silent wrong distances, but sparse latitude/longitude data is unusual and the bug needs a point with an exact 0 coordinate right before the affected row.
- **Difficulty (1)**: use the column index directly (`x1_component = ix1`); the modulo is meaningless.
- **Decision (0)**: no trade-off.

## Upstream search

Queries: "haversine sparse", "DistanceMetric sparse".

No match. #12568 (merged, "Support Haversine distance in pairwise_distances") and #23604 (merged, CSR support for all `DistanceMetric`s) are the related changes.

## Describe the bug and its user-facing impact

`haversine_distances` documents `X`/`Y` as `{array-like, sparse matrix}`, and `pairwise_distances(..., metric="haversine")` forwards sparse input to it. On CSR input, `HaversineDistance.rdist_csr` decides whether a stored value is the latitude or the longitude with `ix1 % x1_start`, where `ix1` is the column index and `x1_start` is the row's offset in the `data` array. The column index alone is the answer (0 or 1); the modulo is wrong when `x1_start == 1`, i.e. for the second row when the first row stores a single value (a point with latitude or longitude exactly 0). For that row, every stored value is read as the latitude.

## Steps/Code to Reproduce

```python
import numpy as np
import scipy.sparse as sp
from sklearn.metrics.pairwise import haversine_distances

# (latitude, longitude) in radians. The first point lies on the prime meridian,
# so its CSR row stores a single value and the second row starts at offset 1.
X = np.array([[0.5, 0.0], [0.0, 1.0]])
Y = np.array([[0.2, 0.3]])
print("dense :", haversine_distances(X, Y).ravel())
print("sparse:", haversine_distances(sp.csr_matrix(X), sp.csr_matrix(Y)).ravel())
```

## Expected Results

The same distances for dense and sparse input: `[0.41049881 0.72334443]`.

## Actual Results

```
dense : [0.41049881 0.72334443]
sparse: [0.41049881 0.83246334]
```

## Root cause and possible fix

`sklearn/metrics/_dist_metrics.pyx.tp:2707-2708` (and the same expression in both tail loops, lines 2726 and 2735):

```cython
x1_component = ix1 if (x1_start == 0) else ix1 % x1_start
x2_component = ix2 if (x2_start == 0) else ix2 % x2_start
```

`x{1,2}_indices` already hold column indices, so the component is `ix1` / `ix2`. With `x_start >= 2` the modulo happens to return `ix` unchanged, which is why only `x_start == 1` is affected (the dense side of a sparse/dense pair is stored with offsets that are multiples of 2, so it is never hit).

## Versions

```
sklearn: 1.10.dev0, commit f570497251 (no change vs upstream/main 289c4d5cfa in the files involved)
python: 3.13, numpy: 2.5, scipy: 1.18, platform: linux
```
