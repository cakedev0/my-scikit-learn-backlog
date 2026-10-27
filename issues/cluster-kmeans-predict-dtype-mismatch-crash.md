# KMeans.predict and score crash when X's float dtype differs from the one seen in fit

```json
{
    "estimators": ["KMeans", "MiniBatchKMeans", "BisectingKMeans"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 3,
    "difficulty": 1,
    "decision": 0,
    "decision_tags": []
}
```

## Classification

- **Priority (3)**: a crash on valid input in the basic fit/predict workflow of three widely used estimators: fit on float32 features, then predict on a plain list or a float64 array (or the reverse). The error (a Cython buffer dtype mismatch) does not tell users what to do, and nothing in the docs hints that dtypes must match. Maintainer calibration, 2026-10-23.
- **Difficulty (1)**: cast `X` to `self.cluster_centers_.dtype` in `predict`/`score` (or validate with `dtype=self.cluster_centers_.dtype`). `transform` already works because it goes through `euclidean_distances`.
- **Decision (0)**: no trade-off.

## Upstream search

Queries: `KMeans Buffer dtype mismatch`, `KMeans predict float32 fitted float64`, `KMeans predict dtype mismatch float`, the exact error message, and `gh pr list --search "KMeans predict dtype"` / `gh issue list --search "KMeans predict float32 float64 ValueError"` (these two don't search comments). No match. Related but different: #6846 (merged, float32 support via fused types, which introduced the dtype-specific Cython paths), #17796 (closed, float32 precision test failure on ARM).

## Describe the bug and its user-facing impact

`predict` and `score` validate `X` with `dtype=[np.float64, np.float32]`, which keeps float32 as float32 and converts everything else to float64. Then they pass `X` and `self.cluster_centers_` to the fused-type Cython E-step, which requires both to have the same dtype. So:

- a model fitted on float32 data crashes on float64 (or int, or list) input;
- a model fitted on float64 data crashes on float32 input.

`transform` works in both cases. Reproduced on 1.9.0 too, so it is not a recent regression.

## Steps/Code to Reproduce

```python
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans, BisectingKMeans

X = np.random.RandomState(0).rand(100, 2)  # float64
for Est in [KMeans, MiniBatchKMeans, BisectingKMeans]:
    for fit_dtype, pred_dtype in [(np.float64, np.float32), (np.float32, np.float64)]:
        est = Est(n_clusters=3, random_state=0).fit(X.astype(fit_dtype))
        for method in ["predict", "score", "transform"]:
            try:
                getattr(est, method)(X.astype(pred_dtype))
                res = "ok"
            except ValueError as e:
                res = f"ValueError: {e}"
            print(f"{Est.__name__} fit {fit_dtype.__name__} {method} {pred_dtype.__name__}: {res}")
```

## Expected Results

`predict` and `score` work for any numeric input dtype, like `transform` and like other estimators.

## Actual Results

```
KMeans fit float64 predict float32: ValueError: Buffer dtype mismatch, expected 'const float' but got 'double'
KMeans fit float64 score float32: ValueError: Buffer dtype mismatch, expected 'const float' but got 'double'
KMeans fit float64 transform float32: ok
KMeans fit float32 predict float64: ValueError: Buffer dtype mismatch, expected 'const double' but got 'float'
KMeans fit float32 score float64: ValueError: Buffer dtype mismatch, expected 'const double' but got 'float'
KMeans fit float32 transform float64: ok
MiniBatchKMeans fit float64 predict float32: ValueError: Buffer dtype mismatch, expected 'const float' but got 'double'
MiniBatchKMeans fit float64 score float32: ValueError: Buffer dtype mismatch, expected 'const float' but got 'double'
MiniBatchKMeans fit float64 transform float32: ok
MiniBatchKMeans fit float32 predict float64: ValueError: Buffer dtype mismatch, expected 'const double' but got 'float'
MiniBatchKMeans fit float32 score float64: ValueError: Buffer dtype mismatch, expected 'const double' but got 'float'
MiniBatchKMeans fit float32 transform float64: ok
BisectingKMeans fit float64 predict float32: ValueError: Buffer dtype mismatch, expected 'const float' but got 'double'
BisectingKMeans fit float64 score float32: ValueError: Buffer dtype mismatch, expected 'const float' but got 'double'
BisectingKMeans fit float64 transform float32: ok
BisectingKMeans fit float32 predict float64: ValueError: Buffer dtype mismatch, expected 'const double' but got 'float'
BisectingKMeans fit float32 score float64: ValueError: Buffer dtype mismatch, expected 'const double' but got 'float'
BisectingKMeans fit float32 transform float64: ok
```

## Root cause and possible fix

`_BaseKMeans._check_test_data` (`sklearn/cluster/_kmeans.py:943-953`) validates with `dtype=[np.float64, np.float32]` instead of the dtype of the fitted centers. `predict` (`:1090`) and `score` (`:1175`) then call `_labels_inertia_threadpool_limit(X, sample_weight, self.cluster_centers_, ...)`, which dispatches to `lloyd_iter_chunked_dense`/`_sparse` (`sklearn/cluster/_k_means_lloyd.pyx`), fused on a single `floating` type for `X` and `centers`. `BisectingKMeans.predict` (`sklearn/cluster/_bisect_k_means.py:480`) uses the same helper.

Fix: in `_check_test_data`, use `dtype=self.cluster_centers_.dtype`. For float32 centers and float64 input this downcasts `X`, the same thing `fit` would have done.

## Versions

```
sklearn: 1.10.dev0, commit 289c4d5cfa (also reproduced on 1.9.0)
python: 3.13, numpy: 2.5.3, scipy: 1.18.1, platform: linux
```
