# FastICA(whiten_solver="eigh") builds the covariance from uncentered X and loses accuracy on data with a large mean

```json
{
    "estimators": ["FastICA", "fastica"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 1,
    "difficulty": 1,
    "decision": 0,
    "decision_tags": []
}
```

## Classification

- **Priority (1)**: silent loss of accuracy (wrong sources) with the non-default `whiten_solver="eigh"`, only when the feature means are large compared to their spread (~1e6 times or more), e.g. raw timestamps or sensor counts.
- **Difficulty (1)**: replace `XT.dot(X)` with `XT.dot(XT.T)`.
- **Decision (0)**: no trade-off, the fix computes the same matrix in exact arithmetic.

## Upstream search

- `"FastICA eigh"`, `"FastICA eigh whiten_solver"`: no match. `git log -L` shows the line was introduced in #22527 ("ENH Add `eigh` solver to `FastICA`") and has not changed since.

## Describe the bug and its user-facing impact

With `whiten_solver="eigh"`, `FastICA` computes the whitening matrix from `linalg.eigh(XT.dot(X))`, where `XT` is the centered (validated, copied) data and `X` is the raw input as passed by the user. In exact arithmetic `Xc.T @ X == Xc.T @ Xc` because the centered columns sum to zero. In floating point, the residual column sums of `Xc` (~eps * mean * n_samples) get multiplied by the mean, so the "covariance" is off by a term proportional to `mean**2`. It is also not symmetric, and `eigh` only reads one triangle.

When the mean is large relative to the standard deviation, the whitening is wrong and the recovered sources are much worse than with `whiten_solver="svd"` on the same data. On the data below with a 1e8 offset, the eigenvalues of `XT.dot(X)` are `[113.3, 10592.8, 29906.4]` instead of `[54.7, 8364.3, 30663.2]` for `XT.dot(XT.T)`.

## Steps/Code to Reproduce

```python
import numpy as np
from sklearn.decomposition import FastICA

rng = np.random.RandomState(0)
S = rng.laplace(size=(2000, 3))
X = S @ rng.randn(3, 3).T

for offset in [0, 1e7]:
    for solver in ["svd", "eigh"]:
        ica = FastICA(whiten_solver=solver, random_state=0, max_iter=1000)
        S_est = ica.fit_transform(X + offset)
        # |correlation| between each true source and its best match
        corr = np.abs(np.corrcoef(S.T, S_est.T)[:3, 3:]).max(axis=1)
        print(f"offset={offset:g} whiten_solver={solver!r}: {np.round(corr, 4)}")
```

## Expected Results

Adding a constant to `X` should not change the recovered sources, and both whitening solvers should agree.

## Actual Results

```
sklearn/decomposition/_fastica.py:132: ConvergenceWarning: FastICA did not converge. Consider increasing tolerance or the maximum number of iterations.
  warnings.warn(
offset=0 whiten_solver='svd': [0.9991 0.9992 0.9999]
offset=0 whiten_solver='eigh': [0.9991 0.9992 0.9999]
offset=1e+07 whiten_solver='svd': [0.9991 0.9992 0.9999]
offset=1e+07 whiten_solver='eigh': [0.8913 0.6133 0.8591]
```

## Root cause and possible fix

`sklearn/decomposition/_fastica.py:616` (commit 289c4d5cfa):

```python
d, u = linalg.eigh(XT.dot(X))
```

`X` here is the raw argument of `_fit_transform`, not the centered `XT`. Use `XT.dot(XT.T)` (or `XT @ XT.T`), which is the actual scatter matrix of the centered data. This also avoids relying on `ndarray.dot` accepting whatever `X` was (list, DataFrame, other dtype).

## Versions

```
sklearn: 1.10.dev0, commit 289c4d5cfa (upstream/main)
python: 3.13, numpy: 2.5.3, scipy: 1.18.1, platform: linux
```
