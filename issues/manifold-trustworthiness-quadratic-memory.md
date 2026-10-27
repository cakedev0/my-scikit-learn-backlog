# trustworthiness allocates three (n_samples, n_samples) matrices, two of them int64

```json
{
    "estimators": ["trustworthiness"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 1,
    "difficulty": 2,
    "decision": 0,
    "decision_tags": []
}
```

## Classification

- **Priority (1)**: `trustworthiness` needs 24 bytes per pair of samples: ~2.4 GB for 10k samples, ~60 GB for 50k samples, so it can't evaluate embeddings of medium-sized datasets (e.g. full MNIST) even though the O(n_samples^2) distance computations would only take seconds to minutes. Loud failure (MemoryError), niche function, hence 1.
- **Difficulty (2)**: rewrite the rank computation per block of rows with `pairwise_distances_chunked`; ~20 lines, plus care with ties so the result stays identical.
- **Decision (0)**: same result (checked on random data), memory goes from O(n_samples^2) to O(chunk_size * n_samples). Ties between equal distances are already broken arbitrarily by `np.argsort`, so a chunked version that ranks with `<` is as valid (and deterministic).

## Upstream search

Queries: "trustworthiness", "trustworthiness memory". Hits are about the `n_neighbors < n_samples / 2` constraint (#18567 closed, #18832 closed, #23033 merged), parameter validation (#26276) and the docstring example (#28203). Nothing about memory.

## Describe the bug and its user-facing impact

`sklearn.manifold.trustworthiness` materializes:

1. `dist_X = pairwise_distances(X)`: (n, n) float64;
2. `ind_X = np.argsort(dist_X, axis=1)`: (n, n) int64;
3. `inverted_index = np.zeros((n_samples, n_samples), dtype=int)`: (n, n) int64,

all alive at the same time, to read only `n_neighbors` entries per row of `inverted_index`. The peak is exactly 3 dense n x n matrices of 8 bytes, while the score only needs, for each sample, the rank in the input space of its `n_neighbors` nearest neighbors in the embedding.

## Steps/Code to Reproduce

```python
import tracemalloc

import numpy as np
from sklearn.manifold import trustworthiness

rng = np.random.RandomState(0)
for n in [1000, 2000, 4000]:
    X = rng.rand(n, 10)
    X_emb = X[:, :2].copy()
    tracemalloc.start()
    trustworthiness(X, X_emb, n_neighbors=5)
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    print(f"n={n}: peak {peak / 2**20:7.1f} MiB = {peak / (8 * n * n):.1f} x one n x n float64 matrix")
```

## Expected Results

Memory bounded by a block of rows of the distance matrix, as for the other pairwise reductions in scikit-learn (`pairwise_distances_chunked`, `silhouette_samples`, `KNeighborsMixin.kneighbors`, ...).

## Actual Results

```
n=1000: peak    23.2 MiB = 3.0 x one n x n float64 matrix
n=2000: peak    91.8 MiB = 3.0 x one n x n float64 matrix
n=4000: peak   366.7 MiB = 3.0 x one n x n float64 matrix
```

## Root cause and possible fix

`sklearn/manifold/_t_sne.py:526-548` (function starts at line 454, commit below): the full `dist_X`, `ind_X` and `inverted_index` arrays.

Possible fix: iterate over `pairwise_distances_chunked(X, metric=metric)`, set the self-distances to `inf` in each block, and compute the rank of each embedded neighbor `j` of sample `i` as `1 + (D[i] < D[i, j]).sum()`. A quick prototype of this (with `working_memory=16`) returns the same value as `trustworthiness` (0.6595122994652407 on a 3000 x 10 random `X`) with a peak of 32 MiB instead of 206 MiB, and its memory no longer grows quadratically.

`metric="precomputed"` keeps needing the user's (n, n) matrix, but the `dist_X.copy()` and the two int64 matrices go away there too.

## Versions

```
sklearn: 1.10.dev0, commit 289c4d5cfa (upstream/main; the local checkout f570497251 only differs in TSNE.fit, not in trustworthiness)
python: 3.13.14, numpy: 2.5.3, scipy: 1.18.1, platform: linux
```

See also: `issues/manifold-trustworthiness-tied-distances.md` (tied distances), which a chunked rewrite could fix at the same time.
