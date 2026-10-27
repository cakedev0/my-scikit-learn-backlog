# IncrementalPCA sets noise_variance_ to 0 when the last batch has n_components samples

```json
{
    "estimators": ["IncrementalPCA"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 1,
    "difficulty": 1,
    "decision": 0,
    "decision_tags": []
}
```

## Classification

- **Priority (1)**: silent wrong `noise_variance_` (and a near-singular `get_covariance()`, a huge `get_precision()`), but only when the last batch has exactly `n_components` samples, which needs `n_samples % batch_size == n_components` in `fit` or a matching `partial_fit` call.
- **Difficulty (1)**: one condition to change in `partial_fit`.
- **Decision (0)**: no trade-off, the zero value is plainly wrong.

## Upstream search

- `"IncrementalPCA noise_variance_"`, `"IncrementalPCA noise"`: no match.
- `git log -L` on the condition points to #23264 (merged, 2022, "Avoid spurious warning in IncrementalPCA when n_samples == n_components_"), which introduced the `n_samples` check to avoid a mean of an empty slice on the first batch. The new condition is too broad for later batches.

## Describe the bug and its user-facing impact

At the end of each `partial_fit` call (also used by `fit` on each batch), `IncrementalPCA` sets `noise_variance_ = 0.0` whenever `n_components_` equals the number of samples of the current batch. That shortcut is only valid for the first batch. On later batches the SVD is computed on the stacked matrix of the previous components, the new batch and the mean correction, so it has more than `n_components_` singular values and the noise variance is not zero.

So `fit` silently returns `noise_variance_ == 0` when the last batch produced by `gen_batches` has exactly `n_components` samples (e.g. `n_samples=105`, `batch_size=100` or `50`, `n_components=5`). `get_covariance()` then drops the isotropic noise term and becomes rank `n_components` (condition number ~1e16), and `get_precision()` returns garbage.

## Steps/Code to Reproduce

```python
import numpy as np
from sklearn.decomposition import PCA, IncrementalPCA

rng = np.random.RandomState(0)
X = rng.randn(105, 10)

print("PCA:                     ", PCA(n_components=5).fit(X).noise_variance_)
print("IncrementalPCA bs=40:    ", IncrementalPCA(n_components=5, batch_size=40).fit(X).noise_variance_)
print("IncrementalPCA bs=50:    ", IncrementalPCA(n_components=5, batch_size=50).fit(X).noise_variance_)
# 105 = 2 * 50 + 5 = 100 + 5: the last batch has exactly n_components samples
print("IncrementalPCA bs=100:   ", IncrementalPCA(n_components=5, batch_size=100).fit(X).noise_variance_)

ipca = IncrementalPCA(n_components=5).partial_fit(X[:100]).partial_fit(X[100:])
print("partial_fit 100 + 5:     ", ipca.noise_variance_)
print("cond(get_covariance()):  ", np.linalg.cond(ipca.get_covariance()))
```

## Expected Results

A positive `noise_variance_` in all cases. `IncrementalPCA` only approximates the trailing eigenvalues (hence 0.23 instead of 0.74 with `batch_size=40`), but it should not drop to exactly 0 depending on the size of the last batch.

## Actual Results

```
PCA:                      0.7378206057885535
IncrementalPCA bs=40:     0.23285212500245484
IncrementalPCA bs=50:     0.0
IncrementalPCA bs=100:    0.0
partial_fit 100 + 5:      0.0
cond(get_covariance()):   7.22412182169185e+16
```

## Root cause and possible fix

`sklearn/decomposition/_incremental_pca.py:368` (commit 289c4d5cfa):

```python
# we already checked `self.n_components <= n_samples` above
if self.n_components_ not in (n_samples, n_features):
    self.noise_variance_ = explained_variance[self.n_components_ :].mean()
else:
    self.noise_variance_ = 0.0
```

`n_samples` is the size of the current batch. The intent of #23264 was to skip the mean when `explained_variance[self.n_components_:]` is empty. Checking that directly fixes both cases:

```python
if explained_variance.shape[0] > self.n_components_:
```

## Versions

```
sklearn: 1.10.dev0, commit 289c4d5cfa (upstream/main)
python: 3.13, numpy: 2.5.3, scipy: 1.18.1, platform: linux
```
