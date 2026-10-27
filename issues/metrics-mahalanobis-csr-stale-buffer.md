# `DistanceMetric("mahalanobis").pairwise` is wrong on CSR input (stale coordinates from previous pairs)

```json
{
    "estimators": ["DistanceMetric"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 1,
    "difficulty": 1,
    "decision": 0,
    "decision_tags": []
}
```

## Classification

- **Priority (1)**: silent wrong distances, but only reachable through `DistanceMetric.pairwise` with sparse input: `pairwise_distances` routes mahalanobis to scipy (which rejects sparse), the neighbors trees reject sparse input, and the reduction backend excludes mahalanobis.
- **Difficulty (1)**: reset the touched buffer entries at the start of `rdist_csr` (or the whole buffer, it has `n_features` entries).
- **Decision (0)**: no trade-off.

## Upstream search

Queries: "mahalanobis sparse", "mahalanobis thread", "mahalanobis n_jobs".

No issue reports the sparse/dense mismatch. Related: #32753 (open, thread race on the same `self.buffer`, see `metrics-mahalanobis-shared-buffer-thread-race.md`) and its draft fix #34671, which rewrites `rdist_csr` without the buffer and would likely fix this as a side effect (I haven't checked its sparse path, its description says it was not reviewed closely yet). #23604 (merged) added CSR support to all `DistanceMetric`s.

## Describe the bug and its user-facing impact

For CSR inputs, `MahalanobisDistance.rdist_csr` fills `self.buffer` only at the indices stored in either row, then computes `buffer.T @ VI @ buffer` over all `n_features` entries. Coordinates that are zero in both rows are never written, so they keep the value left by a previous pair. The returned distance mixes in coordinates of other samples.

## Steps/Code to Reproduce

```python
import numpy as np
import scipy.sparse as sp
from sklearn.metrics import DistanceMetric

dm = DistanceMetric.get_metric("mahalanobis", VI=np.eye(3))  # plain euclidean
X = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
Y = np.array([[0.0, 0.0, 1.0]])
print("dense  :", dm.pairwise(X, Y).ravel())
print("csr/csr:", dm.pairwise(sp.csr_matrix(X), sp.csr_matrix(Y)).ravel())
```

## Expected Results

The same distances as for dense input: `[1.41421356 1.41421356]`.

## Actual Results

```
dense  : [1.41421356 1.41421356]
csr/csr: [1.73205081 1.73205081]
```

## Root cause and possible fix

`sklearn/metrics/_dist_metrics.pyx.tp:1665-1716` (`MahalanobisDistance.rdist_csr`): the merge loop writes `self.buffer[ix]` only for stored indices, and the final double loop reads `self.buffer[0:size]`. Zeroing `self.buffer` at the start of `rdist_csr` fixes it. The buffer is also shared between threads, see `metrics-mahalanobis-shared-buffer-thread-race.md`; a fix that moves to a per-call buffer must zero it as well.

## Versions

```
sklearn: 1.10.dev0, commit f570497251 (no change vs upstream/main 289c4d5cfa in the files involved)
python: 3.13, numpy: 2.5, scipy: 1.18, platform: linux
```
