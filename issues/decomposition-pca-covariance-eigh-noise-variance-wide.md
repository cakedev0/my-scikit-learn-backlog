# PCA(svd_solver="covariance_eigh") computes a different noise_variance_ and score when n_samples < n_features

```json
{
    "estimators": ["PCA"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 1,
    "difficulty": 1,
    "decision": 1,
    "decision_tags": ["output-change"]
}
```

## Classification

- **Priority (1)**: silent inconsistency between solvers in `noise_variance_`, `get_covariance`, `get_precision` and `score`, but only with an explicit `svd_solver="covariance_eigh"` on wide data (`"auto"` only picks it for tall data, where it is a poor choice for wide data anyway).
- **Difficulty (1)**: one line in `_fit_full`: average over `min(n_samples, n_features) - n_components` eigenvalues.
- **Decision (1)**: `output-change` for the few users of `covariance_eigh` on wide data. One could argue that `covariance_eigh` matches the textbook PPCA formula (average over `n_features - n_components` eigenvalues), but the docstring and the other three solvers use `min(n_features, n_samples) - n_components`, a convention chosen in #9108.

## Upstream search

- `"noise_variance_"`: #9108 (merged, 2017) aligned `_fit_truncated` with `_fit_full` on the `min(n_features, n_samples) - n_components` convention; #22300 (merged) division by zero in `get_precision`; #34395 (closed) pseudo-inverse in `get_precision`. None mentions `covariance_eigh`, which was added later in #27491.
- `"covariance_eigh noise_variance"` hit the rate limit; the broader `"noise_variance_"` query above returns nothing about `covariance_eigh`.

## Describe the bug and its user-facing impact

The `noise_variance_` docstring says it is "equal to the average of (min(n_features, n_samples) - n_components) smallest eigenvalues of the covariance matrix of X". The `full`, `arpack` and `randomized` solvers do this. With `svd_solver="covariance_eigh"`, the eigendecomposition of the `n_features x n_features` covariance returns `n_features` eigenvalues, and `noise_variance_` is the mean of the last `n_features - n_components` of them. When `n_samples < n_features`, this includes `n_features - n_samples` extra zero eigenvalues, so `noise_variance_` is smaller, and `get_covariance`, `get_precision`, `score_samples` and `score` differ from the other solvers.

`test_pca_solver_equivalence` covers wide data but only with `n_components=None` (noise variance 0), so it does not catch this.

## Steps/Code to Reproduce

```python
import numpy as np
from sklearn.decomposition import PCA

rng = np.random.RandomState(0)
X = rng.randn(20, 50)  # n_samples < n_features

for solver in ["full", "arpack", "randomized", "covariance_eigh"]:
    pca = PCA(n_components=5, svd_solver=solver, random_state=0).fit(X)
    print(f"{solver:16} noise_variance_={pca.noise_variance_:.4f} score={pca.score(X):.3f}")
```

## Expected Results

The same `noise_variance_` and `score` for all solvers, as documented.

## Actual Results

```
full             noise_variance_=1.6791 score=-70.946
arpack           noise_variance_=1.6791 score=-70.946
randomized       noise_variance_=1.6791 score=-70.947
covariance_eigh  noise_variance_=0.5597 score=-60.478
```

## Root cause and possible fix

`sklearn/decomposition/_pca.py:638-639` (commit 289c4d5cfa):

```python
if n_components < min(n_features, n_samples):
    self.noise_variance_ = xp.mean(explained_variance_[n_components:])
```

For `full`, `explained_variance_` has `min(n_samples, n_features)` entries; for `covariance_eigh` it has `n_features` entries. Slicing `explained_variance_[n_components:min(n_features, n_samples)]` before the mean fixes it. `explained_variance_ratio_` and the returned components are not affected (the extra eigenvalues are zero and get truncated).

## Versions

```
sklearn: 1.10.dev0, commit 289c4d5cfa (upstream/main)
python: 3.13, numpy: 2.5.3, scipy: 1.18.1, platform: linux
```
