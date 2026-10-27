# PCA(whiten=True) returns a wrong get_covariance, get_precision and score

```json
{
    "estimators": ["PCA", "IncrementalPCA"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 2,
    "difficulty": 1,
    "decision": 1,
    "decision_tags": ["output-change"]
}
```

## Classification

- **Priority (2)**: silent wrong results in `get_covariance`, `get_precision`, `score` and `score_samples` for a common option of a widely used estimator. The values are off by a factor of the explained variance, so they look plausible; only users of the probabilistic PCA methods with `whiten=True` are hit (e.g. selecting `n_components` by cross-validated `score`).
- **Difficulty (1)**: remove two `if self.whiten:` blocks in `_BasePCA` and adapt one test assertion.
- **Decision (1)**: `output-change`: `score` and `get_covariance` of whitened models change (to the same values as `whiten=False`). The test `test_pca_score` asserts that the whitened model scores lower on the training data, which encodes the bug and must be updated.

## Upstream search

- `"get_covariance"`: found #13898 "PCA implementation of score_samples calculates the wrong precision matrix when whiten is True" (2019). It describes this exact bug, got no maintainer answer on the bug itself, and was closed by its author in 2020 without a fix. The linked PR #13922 (a different `score_samples` implementation) was closed unmerged.
- `"PCA whiten get_covariance"`, `"whiten score PCA"`, `"get_precision whiten"`: no other match.
- #34395 "FIX use pseudo-inverse for singular PCA precision" (closed) is about singular precision matrices, not whitening.

## Describe the bug and its user-facing impact

With `whiten=True`, `PCA.get_covariance()` multiplies each component's contribution by its explained variance twice, so it returns roughly `components_.T @ diag(explained_variance_**2) @ components_` instead of the probabilistic PCA covariance. `get_precision()`, `score_samples()` and `score()` inherit the error.

Whitening only rescales the output of `transform`: `components_`, `explained_variance_` and `noise_variance_` are identical with and without whitening, so the generative model (and its covariance and likelihood on `X`) should be identical too. Users who pick `n_components` or compare models with `PCA(whiten=True).score` (e.g. in `GridSearchCV`) get a wrong log-likelihood. `IncrementalPCA` shares the code and has the same bug.

## Steps/Code to Reproduce

```python
import numpy as np
from sklearn.decomposition import PCA

rng = np.random.RandomState(0)
X = rng.randn(500, 4) * [5.0, 3.0, 1.0, 0.1]

pca = PCA(n_components=2).fit(X)
pca_w = PCA(n_components=2, whiten=True).fit(X)

print("empirical variances:   ", np.round(np.var(X, axis=0, ddof=1), 3))
print("whiten=False diag(cov):", np.round(np.diag(pca.get_covariance()), 3))
print("whiten=True  diag(cov):", np.round(np.diag(pca_w.get_covariance()), 3))
print("score whiten=False:", pca.score(X))
print("score whiten=True: ", pca_w.score(X))
```

## Expected Results

The same covariance and score for `whiten=True` and `whiten=False`: the first two diagonal entries close to the empirical variances 23.98 and 8.22.

## Actual Results

```
empirical variances:    [2.3983e+01 8.2240e+00 9.4300e-01 1.0000e-02]
whiten=False diag(cov): [23.983  8.224  0.477  0.476]
whiten=True  diag(cov): [5.64268e+02 6.42020e+01 4.88000e-01 4.77000e-01]
score whiten=False: -7.572254771886196
score whiten=True:  -9.26586877855104
```

## Root cause and possible fix

`sklearn/decomposition/_base.py:50-51` (`get_covariance`) and `:92-93` (`get_precision`) do:

```python
if self.whiten:
    components_ = components_ * xp.sqrt(exp_var[:, np.newaxis])
```

This is a leftover from the time `PCA` stored whitened components (`V / S * sqrt(n_samples)`), where this line undid the scaling. Commit 3f6c61fed7 (2014) made `components_` unit norm and moved the whitening to `transform`, and removed this block from `PCA.get_precision`, but the block survived in the `IncrementalPCA`/`_BasePCA` code that later became the shared base class.

Fix: delete both blocks. `test_pca_score` in `sklearn/decomposition/tests/test_pca.py` asserts `ll1 > ll2` for the whitened model on the same data; it should assert equality instead.

## Versions

```
sklearn: 1.10.dev0, commit 289c4d5cfa (upstream/main)
python: 3.13, numpy: 2.5.3, scipy: 1.18.1, platform: linux
```
