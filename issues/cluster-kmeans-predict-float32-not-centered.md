# KMeans.predict and score disagree with labels_ on float32 data far from the origin

```json
{
    "estimators": ["KMeans", "MiniBatchKMeans", "BisectingKMeans"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 3,
    "difficulty": 2,
    "decision": 1,
    "decision_tags": ["perf-low", "float-order"]
}
```

## Classification

- **Priority (3)**: silent wrong predictions that look plausible. It needs float32 input whose offset from the origin is large compared to the cluster spread (~1e3-1e4), but common data sits there: float32 GPS coordinates within a city give 34% of the training samples a wrong `predict` label and 4x the inertia on 1.7.2 (see below). Casting to float64 is a cheap workaround once you know. Maintainer calibration, 2026-10-26.
- **Difficulty (2)**: centering in `predict`/`score` is a few lines, but it is shared by `_labels_inertia` callers in `KMeans`, `MiniBatchKMeans` and `BisectingKMeans`, and sparse input cannot be centered the same way.
- **Decision (1)**: `perf-low`: centering a dense `X` in `predict` naively needs a copy (or a float64 path for float32), but centering each chunk inside the Lloyd kernel avoids it (see the fix below). Measured on a prototype (2026-10-28, `kmeans_learnings/experiments.md` E42): the E-step is unchanged for n_features <= 32 and up to ~20% slower at 1 thread and ~30% slower at 16 threads (intel-laptop) on wide data (d = 180, 1000), still much cheaper than copying X. `float-order`: labels change at the level of rounding for well-conditioned data.

## Upstream search

Queries: `KMeans float32 predict labels_`, `KMeans float32 precision`, `KMeans predict center data mean`, `KMeans labels_ predict mismatch`, `KMeans predict float32 wrong cluster`, `KMeans score inertia_ different`, `KMeans predict centered float32` (comments). No match. Related but different:

- #9354 (closed): float32 precision of `euclidean_distances`, fixed there by upcasting chunks; that is why `transform` is right here.
- #34074 (open): `labels_` vs `predict` mismatch after empty-cluster relocation, other cause.
- #6156 (closed, 2018): old inertia inconsistency, unrelated code.

## Describe the bug and its user-facing impact

`KMeans.fit` subtracts the mean of `X` before running Lloyd/Elkan ("subtract of mean of x for more accurate distance computations"), so the training labels are accurate. `predict` and `score` do not: they call the Lloyd E-step on the raw data, which picks the closest center with `||c||² - 2 x·c`. In float32, when `x` and `c` are far from the origin, this difference cancels catastrophically (e.g. `||c||² ~ 3e8` with a float32 resolution of ~30, while squared distances between clusters are ~10-100).

So on the training data itself, `predict(X)` differs from `labels_` and `-score(X)` differs from `inertia_`, while `transform(X).argmin(axis=1)` (which goes through `euclidean_distances` with float64 upcasting) agrees with `labels_`. On new data, `predict` silently assigns many samples to the wrong cluster. Coordinates in meters, timestamps or prices stored as float32 can easily hit this.

`MiniBatchKMeans` and `BisectingKMeans` share `predict`/`score` (and `MiniBatchKMeans` never centers, so its fit is affected too).

Realistic case, on scikit-learn 1.7.2: 200k float32 GPS points around 50 hot spots in Paris (~300 m spread each, hot spots within ~10 km), `KMeans(n_clusters=50, n_init=1)`:

| dtype | `labels_ != predict(X)` | inertia of `predict` labels / `inertia_` | `-score(X) / inertia_` |
| --- | ---: | ---: | ---: |
| float64 | 0 | 1.000 | 1.000 |
| float32 | 68591 / 200000 | 4.08 | 4.08 |

The error grows with `n_features` too: on blobs with d=50, offsets of 100x and 1000x the data range flip 2% and 85% of the `predict` labels.

## Steps/Code to Reproduce

```python
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

X, _ = make_blobs(n_samples=5000, centers=5, n_features=3, random_state=0)
X = (X + 10_000).astype(np.float32)  # e.g. coordinates in meters, prices, ...

km = KMeans(n_clusters=5, random_state=0).fit(X)
print("labels_ != predict(X):         ", (km.labels_ != km.predict(X)).sum())
print("labels_ != transform(X).argmin:", (km.labels_ != km.transform(X).argmin(axis=1)).sum())
print("inertia_:  ", km.inertia_)
print("-score(X): ", -km.score(X))
```

GPS reproducer:

```python
import numpy as np
from sklearn.cluster import KMeans

rng = np.random.default_rng(0)
hubs = np.array([48.8566, 2.3522]) + rng.normal(scale=0.05, size=(50, 2))
X = hubs[rng.integers(50, size=200_000)] + rng.normal(scale=0.003, size=(200_000, 2))
X = X.astype(np.float32)

km = KMeans(n_clusters=50, random_state=0, n_init=1).fit(X)
print((km.labels_ != km.predict(X)).sum(), -km.score(X) / km.inertia_)
```

## Expected Results

`predict(X)` equal to `labels_` (the `predict` docstring describes it as the index of the closest center, and `fit_predict` is documented as equivalent to `fit(X).predict(X)`), and `-score(X)` close to `inertia_`.

## Actual Results

```
labels_ != predict(X):          1053
labels_ != transform(X).argmin: 0
inertia_:   14724.3525390625
-score(X):  51740.9921875
```

## Root cause and possible fix

Fix opened on the fork: cakedev0/scikit-learn#41 (2026-10-28), the per-chunk `shift` option below.

- `sklearn/cluster/_kmeans.py:1474`: `fit` centers dense `X` (and the init) with `X -= X_mean`.
- `sklearn/cluster/_kmeans.py:1090` (`predict`) and `:1175` (`score`) call `_labels_inertia_threadpool_limit` on uncentered `X`.
- `sklearn/cluster/_k_means_lloyd.pyx:200-201`: the E-step ranks centers by `||c||² - 2 x·c` (GEMM), which loses all precision in float32 for large offsets.

Possible fix: in `predict`/`score`, for dense input, subtract a reference point (e.g. the mean of `cluster_centers_`, or the training mean stored at fit) from both `X` and the centers before the E-step. Done on `X` up front it costs one copy of `X`. Without the copy: give `lloyd_iter_chunked_dense` a `shift` argument and subtract it from each chunk of 256 rows into a thread-local buffer before the GEMM, which gives the same precision as `fit`. The same kernel change would let `fit` stop copying and centering `X` (implicit centering, follow-up of cakedev0/scikit-learn#39). Shifting only the centers (GEMM on raw `X` against `c - s`, plus a per-center `2 s.(c - s)` term) is not enough: the error then grows linearly with the offset instead of quadratically, which still mislabels ~3e-4 of the samples at offset/spread 1e3 and ~4% at 1e4 (float32, d=50, k=20). An alternative is to run the E-step in float64 for float32 input. Sparse input cannot be centered without densifying; there the same cancellation affects `fit` too (sparse float32 with a 1e4 offset gave `inertia_` of 36544 with Lloyd and 7392 with Elkan, vs 14724 dense), but sparse data far from the origin is much less common.

## Versions

```
sklearn: 1.10.dev0, commit 289c4d5cfa (GPS reproducer also checked on 1.7.2 from PyPI, 2026-10-26)
python: 3.13, numpy: 2.5.3, scipy: 1.18.1, platform: linux
```
