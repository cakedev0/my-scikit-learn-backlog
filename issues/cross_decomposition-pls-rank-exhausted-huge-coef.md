# `PLSRegression` returns ~1e14 coefficients when `n_components` exceeds the rank of collinear `X`

```json
{
    "estimators": ["PLSRegression", "PLSCanonical", "CCA"],
    "existing_issue": "https://github.com/scikit-learn/scikit-learn/issues/31971",
    "reproduced": true,
    "priority": 2,
    "difficulty": 2,
    "decision": 1,
    "decision_tags": ["unclear-spec", "output-change"]
}
```

## Classification

- **Priority (2)**: silent garbage (coefficients ~1e14, worse training fit than with fewer components, no warning) for collinear features, e.g. one-hot encoded categories or derived features, when `n_components` reaches `n_features` (allowed by the `min(n_samples, n_features)` bound). Users sweeping `n_components` by CV get a bad last candidate that CV will usually discard, hence not 3.
- **Difficulty (2)**: a few lines in the NIPALS loop, but choosing the threshold for "numerically zero residual" and what to do next (stop early as for a constant `y` residual, or raise) needs care.
- **Decision (1)**: `unclear-spec`: stop with fewer effective components (as for the constant `y` residual) or raise, see the discussion in #31971. `output-change`: the affected fits change, which is the point.

## Upstream search

#31971 (open, "ValueError in PLSRegression.fit() with zero-variance predictor"): when all of `X` has zero variance, the first `x_scores` is exactly zero and the deflation divides 0 by 0 (NaN predictions or a LAPACK error). @ogrisel noted there: "if the problem only happens when all features have 0 variance, then I think it would be fine to raise a `ValueError`... if you can reproduce the problem on non-trivial training sets, then maybe it might be interesting to [...]". This file is that non-trivial case: same root cause (no guard when the `X` residual is exhausted), but the residual is round-off instead of exactly 0, so instead of NaNs the model silently gets huge coefficients. #31972 (open PR) adds a guard on `x_scores @ x_scores < eps` that would also catch this case. Other queries ("PLSRegression n_components rank", "PLSRegression collinear coefficients"): #29662 (closed, rank bounds), #25978 / #26204 (docs of the `n_components` range), #2089 (closed, old NIPALS failures).

## Describe the bug and its user-facing impact

The NIPALS loop deflates `X` after each component. When `X` has rank `r < n_features` (collinear columns), the residual `Xk` is zero up to round-off after `r` components. The next iteration normalizes a weight vector computed from that noise, gets `x_scores` of norm ~1e-15, and divides by `x_scores @ x_scores` (~1e-30) to compute `x_loadings` and `y_loadings`. The last `y_loadings` is then ~1e14 and so are the coefficients. No warning is raised.

The model with `n_components=4` fits the training data worse than the one with `n_components=3`, which is impossible for PLS with exact arithmetic (with `n_components = rank`, PLS regression equals OLS).

## Steps/Code to Reproduce

```python
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LinearRegression

rng = np.random.RandomState(0)
X = rng.randn(40, 3)
X = np.c_[X, X[:, 0] + X[:, 1]]  # 4 collinear features, rank 3
y = X @ rng.randn(4) + 0.5 * rng.randn(40)

print(f"OLS train R2: {LinearRegression().fit(X, y).score(X, y):.4f}")
for k in [3, 4]:
    pls = PLSRegression(n_components=k).fit(X, y)
    print(f"n_components={k}: train R2 {pls.score(X, y):.4f}, coef_ {pls.coef_.ravel()}")
```

## Expected Results

With `n_components=4` the same predictions as with `n_components=3` (the 4th component carries no information), or at least a warning/error that the `X` residual is exhausted, as is done for a constant `y` residual (`UserWarning: y residual is constant at iteration k`, then the loop stops).

## Actual Results

```
OLS train R2: 0.9473
n_components=3: train R2 0.9473, coef_ [ 1.15128457 -0.24122308  0.36820203  0.50774222]
n_components=4: train R2 0.8896, coef_ [ 2.95080358e+14  2.95080358e+14  7.13789158e-01 -2.95080358e+14]
```

No warning is emitted (run with `-W always`).

## Root cause and possible fix

`sklearn/cross_decomposition/_pls.py:328` and `:337`:

```python
x_loadings = np.dot(x_scores, Xk) / np.dot(x_scores, x_scores)
...
y_loadings = np.dot(x_scores, yk) / np.dot(x_scores, x_scores)
```

There is no check that `Xk` (or `x_scores`) is numerically nonzero. The `y` side already has such a guard (columns of `yk` below `10 * eps` are zeroed, and an all-zero `yk` stops the loop with a warning). A symmetric guard on `Xk`, relative to the norm of the initial centered/scaled `X` (not an absolute `eps`, since `X` may have any scale), would stop the loop with a warning and leave the remaining components at zero, which `pinv` in the rotation computation already handles. The same loop is used by `PLSCanonical` and `CCA` (bounded by `min(n_samples, n_features, n_targets)`, so they hit it less often).

## Versions

```
sklearn: 1.10.dev0, commit 289c4d5cfa
python: 3.13.14, numpy: 2.5.3, scipy: 1.18.1, platform: linux
```
