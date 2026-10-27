# NearestCentroid densifies sparse X in fit (3x the dense size) and in predict_proba

```json
{
    "estimators": ["NearestCentroid"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 2,
    "difficulty": 2,
    "decision": 0,
    "decision_tags": ["float-order"]
}
```

## Classification

- **Priority (2)**: a regression since 1.6 on a documented use case: `NearestCentroid` (Rocchio) is shown on sparse TF-IDF data in `examples/text/plot_document_classification_20newsgroups.py`. Every `fit` on sparse input now needs ~3x the memory of the densified matrix, which makes it unusable on real text corpora (e.g. 10k documents x 100k terms: ~24 GB instead of a few MB). Results are correct and the failure is loud (MemoryError), hence not 3.
- **Difficulty (2)**: the pooled within-class variance must be computed without materializing `X - centroids_[y]`, e.g. per class with `sklearn.utils.sparsefuncs.mean_variance_axis`. A few lines in `fit`, plus a sparse-aware rescaling in `_decision_function`, both need care to keep the same numbers.
- **Decision (0)**: no API or output change beyond floating-point noise (`float-order`).

## Upstream search

Queries: "NearestCentroid sparse", "NearestCentroid memory" (no results), "NearestCentroid" (30 results). Related:

- #26689 (merged, 1.6): added `decision_function`/`predict_proba` and moved the within-class std computation out of `if self.shrink_threshold:` (which used to be rejected for sparse input), which introduced this.
- #33477 (open, `sample_weight` support): keeps the densification, and even makes it explicit (`X_dense - self.centroids_[y_ind]`).
- #33335 (open, `partial_fit`): computes per-class variances with `mean_variance_axis`, which would fix `fit` as a side effect, but it's a feature PR that has been open for months.

No issue reports the memory blow-up.

## Describe the bug and its user-facing impact

Since 1.6, `NearestCentroid.fit` always computes the within-class standard deviation as `(X - self.centroids_[y_ind]) ** 2`. With sparse `X`, `X - dense` is a dense `(n_samples, n_features)` matrix, `self.centroids_[y_ind]` is another one and `** 2` a third. So `fit` on sparse data peaks at ~3x the size of the densified input, while the input itself and the fitted model (`centroids_`, `n_classes x n_features`) are tiny.

`predict_proba`, `predict_log_proba`, `decision_function` (and `predict` when `priors` is not uniform) also densify transiently, through `X_normalized[:, mask] /= self.within_class_std_dev_[mask]` on a CSR matrix (~1.4x the dense size).

Before 1.6 this code only ran with `shrink_threshold`, which raised for sparse input, so sparse `fit` was O(nnz).

## Steps/Code to Reproduce

```python
import tracemalloc

import numpy as np
import scipy.sparse as sp
from sklearn.neighbors import NearestCentroid

rng = np.random.RandomState(0)
n_samples = 1_000
for n_features in [5_000, 10_000, 20_000]:
    # TF-IDF-like data: 0.2% non-zeros
    X = sp.random(n_samples, n_features, density=0.002, format="csr", random_state=rng)
    y = rng.randint(0, 3, n_samples)
    dense_mib = n_samples * n_features * 8 / 2**20
    sparse_mib = (X.data.nbytes + X.indices.nbytes + X.indptr.nbytes) / 2**20

    tracemalloc.start()
    clf = NearestCentroid().fit(X, y)
    fit_peak = tracemalloc.get_traced_memory()[1] / 2**20
    tracemalloc.reset_peak()
    clf.predict_proba(X)
    proba_peak = tracemalloc.get_traced_memory()[1] / 2**20
    tracemalloc.stop()
    print(
        f"n_features={n_features:6d}: X is {sparse_mib:.2f} MiB sparse ({dense_mib:.0f} MiB if dense), "
        f"fit peak {fit_peak:.0f} MiB, predict_proba peak {proba_peak:.0f} MiB"
    )
```

## Expected Results

Peak memory in `fit` of the order of `nnz` plus `n_classes * n_features` (a few MiB here), as before 1.6 and as for the other estimators that accept sparse input.

## Actual Results

```
n_features=  5000: X is 0.12 MiB sparse (38 MiB if dense), fit peak 115 MiB, predict_proba peak 55 MiB
n_features= 10000: X is 0.23 MiB sparse (76 MiB if dense), fit peak 229 MiB, predict_proba peak 108 MiB
n_features= 20000: X is 0.46 MiB sparse (153 MiB if dense), fit peak 458 MiB, predict_proba peak 218 MiB
```

(The `within_class_std_dev_ has at least 1 zero standard deviation` UserWarning is also raised on each `fit`, which is almost unavoidable on sparse text data; removed from the output.)

## Root cause and possible fix

`sklearn/neighbors/_nearest_centroid.py:231` (commit below):

```python
variance = np.array(X - self.centroids_[y_ind], copy=False) ** 2
```

The pooled within-class sum of squares can be computed per class without forming the residuals, e.g. `n_k * mean_variance_axis(X[y_ind == k], axis=0)[1]` summed over classes (the centroids are the class means for the euclidean metric; for `metric="manhattan"` the centroids are medians, so the sum of squares around the median needs `var + (mean - median) ** 2`).

In `_decision_function` (`sklearn/neighbors/_nearest_centroid.py:316-317`), rescaling the columns of a sparse `X` can be done with `sklearn.utils.sparsefuncs.inplace_column_scale` (or `X @ diags(1 / std)`) instead of a masked fancy-index assignment.

## Versions

```
sklearn: 1.10.dev0, commit 289c4d5cfa (upstream/main; local checkout f570497251, no diff on this file)
python: 3.13.14, numpy: 2.5.3, scipy: 1.18.1, platform: linux
```
