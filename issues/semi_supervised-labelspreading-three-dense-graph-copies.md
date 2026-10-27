# LabelSpreading(kernel="rbf").fit holds three (n_samples, n_samples) matrices at once

```json
{
    "estimators": ["LabelSpreading"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 1,
    "difficulty": 1,
    "decision": 0,
    "decision_tags": []
}
```

## Classification

- **Priority (1)**: memory only, no wrong results, on a rarely used estimator whose dense RBF graph already restricts it to small data: 3x the graph instead of 1x, and `kernel="knn"` is the usual way out for larger data. (priority review, 2026-10-23: was 2)
- **Difficulty (1)**: two lines in `LabelSpreading._build_graph`: ask scipy for an in-place Laplacian and negate in place.
- **Decision (0)**: identical results (checked, max abs difference 0.0), strictly less memory. `copy=` is available in `scipy.sparse.csgraph.laplacian` for all supported scipy versions.

## Upstream search

Queries: "LabelSpreading memory", "LabelSpreading laplacian", "label spreading MemoryError", "LabelSpreading". Related but different:

- #34328 (closed, DOC): add a scalability warning for the RBF kernel. It documents the inherent O(n_samples^2) cost, not the extra copies.
- #15922 (closed, WIP): a `sparse-rbf` kernel option. Feature, not this.
- #34350, #34835 (open): zero-affinity rows in `predict_proba`/`fit`. #34835 does not touch `_build_graph`.

`git log upstream/main -- sklearn/semi_supervised/_label_propagation.py`: no related change.

## Describe the bug and its user-facing impact

`LabelSpreading._build_graph` computes the RBF affinity matrix, then builds its normalized Laplacian as a new array, then negates it into a third array, while the two previous ones are still referenced. With the dense `kernel="rbf"` (the default), each of them is an `(n_samples, n_samples)` float64 matrix, so `fit` peaks at 3 of them. `LabelPropagation` with the same kernel peaks at 1.

Users hit a MemoryError (or swap) at ~1.7x fewer samples than the algorithm needs.

## Steps/Code to Reproduce

```python
import tracemalloc

import numpy as np
from sklearn.datasets import make_classification
from sklearn.semi_supervised import LabelPropagation, LabelSpreading

for n_samples in [1_000, 2_000, 4_000]:
    X, y = make_classification(n_samples=n_samples, random_state=0)
    y[::2] = -1  # half of the samples are unlabeled
    n2 = n_samples**2 * 8  # one n_samples x n_samples float64 matrix
    for Est in [LabelPropagation, LabelSpreading]:
        tracemalloc.start()
        Est(kernel="rbf", max_iter=5).fit(X, y)
        peak = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
        print(f"n_samples={n_samples}, {Est.__name__:16s}: fit peak = {peak / n2:.2f} x (n_samples, n_samples) float64")
```

## Expected Results

A peak of ~1 dense graph for both estimators: the affinity matrix can be turned into the (negated, zero-diagonal) normalized Laplacian in place.

## Actual Results

```
n_samples=1000, LabelPropagation: fit peak = 1.01 x (n_samples, n_samples) float64
n_samples=1000, LabelSpreading  : fit peak = 3.00 x (n_samples, n_samples) float64
n_samples=2000, LabelPropagation: fit peak = 1.01 x (n_samples, n_samples) float64
n_samples=2000, LabelSpreading  : fit peak = 3.00 x (n_samples, n_samples) float64
n_samples=4000, LabelPropagation: fit peak = 1.00 x (n_samples, n_samples) float64
n_samples=4000, LabelSpreading  : fit peak = 3.00 x (n_samples, n_samples) float64
```

(`ConvergenceWarning`s from `max_iter=5` removed from the output.)

## Root cause and possible fix

`sklearn/semi_supervised/_label_propagation.py:625-627` (commit below):

```python
affinity_matrix = self._get_kernel(self.X_)                    # 1st n x n
laplacian = csgraph_laplacian(affinity_matrix, normed=True)    # 2nd n x n (copy=True by default)
laplacian = -laplacian                                         # 3rd n x n
```

Possible fix, for the dense case:

```python
laplacian = csgraph_laplacian(affinity_matrix, normed=True, copy=False)
np.negative(laplacian, out=laplacian)
laplacian.flat[:: n_samples + 1] = 0.0
```

The sparse (`kernel="knn"`) branch can keep `-laplacian`, since it is O(n_samples * n_neighbors). I checked this patch by monkeypatching `_build_graph`: the peak drops from 3.00x to 1.01x at `n_samples=2000`, and `label_distributions_` is identical.

## Versions

```
sklearn: 1.10.dev0, commit 289c4d5cfa (upstream/main; local checkout f570497251, no diff on this file)
python: 3.13.14, numpy: 2.5.3, scipy: 1.18.1, platform: linux
```
