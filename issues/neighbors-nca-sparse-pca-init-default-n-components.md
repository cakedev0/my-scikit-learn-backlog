# NeighborhoodComponentsAnalysis(init="pca") fails on sparse input unless n_components < n_features

```json
{
    "estimators": ["NeighborhoodComponentsAnalysis"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 2,
    "difficulty": 1,
    "decision": 0,
    "decision_tags": []
}
```

## Classification

- **Priority (2)**: sparse support with `init="pca"` is a new, unreleased feature (#34122, in `upcoming_changes`), and it fails with the default `n_components=None`, with an error about `svd_solver='arpack'` that the user never chose. Worth fixing before the release ships the feature.
- **Difficulty (1)**: pass `svd_solver="covariance_eigh"` (which supports sparse input and `n_components` up to `n_features`) to the internal `PCA` when `X` is sparse and `n_components >= min(X.shape)`.
- **Decision (0)**: no trade-off for released behavior.

## Upstream search

- `NeighborhoodComponentsAnalysis sparse`: #34122 (merged, added sparse support for `init` in `["pca", "random", "identity"]`; its test uses `n_components=2`), #21815 (closed, older request for sparse NCA), #35161 (closed, unrelated string labels problem). No report of this failure.

## Describe the bug and its user-facing impact

#34122 added sparse input support to `NeighborhoodComponentsAnalysis` for `init` in `["pca", "random", "identity"]`. With `init="pca"`, NCA fits `PCA(n_components=n_components)` where `n_components = self.n_components or n_features`. On sparse input, `PCA(svd_solver="auto")` switches to `"arpack"`, which requires `n_components < min(n_samples, n_features)`. So with the default `n_components=None` (or any `n_components >= min(n_samples, n_features)`), fitting on sparse data always fails, while the same call on dense data works. The test added in #34122 uses `n_components=2`, so it doesn't cover the default.

## Steps/Code to Reproduce

```python
import numpy as np
import scipy.sparse as sp
from sklearn.datasets import make_classification
from sklearn.neighbors import NeighborhoodComponentsAnalysis

X, y = make_classification(n_samples=60, n_features=6, random_state=0)
nca = NeighborhoodComponentsAnalysis(init="pca", max_iter=5, random_state=0)
print(nca.fit(X, y).components_.shape)
nca.fit(sp.csr_matrix(X), y)
```

## Expected Results

The sparse fit works and gives the same `components_` as the dense one (shape `(6, 6)`).

## Actual Results

```
(6, 6)
Traceback (most recent call last):
  ...
  File "/home/arthur/scikit-learn/sklearn/decomposition/_pca.py", line 684, in _fit_truncated
    raise ValueError(
    ...<4 lines>...
    )
ValueError: n_components=6 must be strictly less than min(n_samples, n_features)=6 with svd_solver='arpack'
```

## Root cause and possible fix

`sklearn/neighbors/_nca.py:433-435` builds `PCA(n_components=n_components, random_state=self.random_state_)` with the default `svd_solver="auto"`, which `PCA` resolves to `"arpack"` for sparse input (`sklearn/decomposition/_pca.py:499-500`). Passing `svd_solver="covariance_eigh"` when `X` is sparse and `n_components >= min(X.shape)` fixes it: it accepts sparse input and `n_components` up to `n_features` (checked: same components as dense `PCA` up to sign, max abs diff 5e-15 on this data). Keeping `"arpack"` otherwise avoids materializing an `(n_features, n_features)` covariance for wide data with few components. A test with the default `n_components` on sparse input would cover it.

## Versions

```
sklearn: 1.10.dev0, commit f570497251 (no diff in sklearn/neighbors vs upstream/main 289c4d5cfa)
python: 3.13, numpy: 2.5.3, scipy: 1.18.1, platform: linux
```
