# `LocallyLinearEmbedding(method="hessian")` computes LTSA, not Hessian eigenmaps

```json
{
    "estimators": ["LocallyLinearEmbedding", "locally_linear_embedding"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 2,
    "difficulty": 1,
    "decision": 1,
    "decision_tags": ["output-change"]
}
```

## Classification

- **Priority (2)**: silent wrong algorithm: `method="hessian"` returns exactly the `method="ltsa"` embedding, so users comparing the LLE variants (as in the manifold examples) compare LTSA with itself. The output is still a sensible embedding, and HLLE is a niche method, hence not 3.
- **Difficulty (1)**: one-line fix (`qr(Yi, mode="economic")`, or slice the `dp` Hessian columns), easy to check against Donoho & Grimes' reference code.
- **Decision (1)**: `output-change`: `method="hessian"` results change for every user of that option. The `n_neighbors > n_components * (n_components + 3) / 2` requirement becomes meaningful again.

## Upstream search

Queries: "hessian LLE", "hessian ltsa", "hessian eigenmapping qr", and "hessian ltsa same" in comments. Hits: #211 (merged, the original HLLE/MLLE implementation from 2011), #28096 (merged, sparse M memory optimization), #29792 / #29798 (open, `transform` vs `fit_transform` discrepancy, unrelated), #24353 (closed, MLLE example `modified_tol`). Nothing about HLLE being equal to LTSA.

## Describe the bug and its user-facing impact

In Hessian eigenmaps (Donoho & Grimes 2003), each neighborhood contributes the `dp = d (d + 1) / 2` columns of the orthonormalized quadratic terms (the local Hessian estimator). scikit-learn builds `Yi = [1, U, quadratic terms]` of shape `(k, 1 + d + dp)` but orthonormalizes it with `scipy.linalg.qr(Yi)`, whose default `mode="full"` returns a `(k, k)` `Q`. Taking `Q[:, d + 1:]` then keeps `k - d - 1` columns: the `dp` Hessian columns plus the whole orthogonal complement of `Yi`.

These columns span the orthogonal complement of `span(1, U)`, so `w @ w.T = I - P_{1,U}`, which is exactly LTSA's local matrix `I - Gi @ Gi.T`. The Hessian quadratic form disappears and `method="hessian"` returns the LTSA embedding (same eigenvectors and same `reconstruction_error_` up to round-off). It has been like this since the 2011 implementation.

Users choosing `method="hessian"` (or comparing methods, as in `plot_lle_digits.py` and `plot_compare_methods.py`) silently get LTSA.

## Steps/Code to Reproduce

```python
import numpy as np
from scipy.linalg import qr, svd
from sklearn.datasets import make_swiss_roll
from sklearn.manifold import LocallyLinearEmbedding

X, _ = make_swiss_roll(n_samples=300, random_state=0)
kw = dict(n_neighbors=10, n_components=2, eigen_solver="dense")
hessian = LocallyLinearEmbedding(method="hessian", **kw).fit(X)
ltsa = LocallyLinearEmbedding(method="ltsa", **kw).fit(X)
print("max |hessian - ltsa| embedding:",
      np.abs(np.abs(hessian.embedding_) - np.abs(ltsa.embedding_)).max())
print("reconstruction_error_:", hessian.reconstruction_error_, ltsa.reconstruction_error_)

# Local weights for one neighborhood, as built by the "hessian" branch
k, d = 10, 2
dp = d * (d + 1) // 2
Gi = X[:k] - X[:k].mean(0)
U = svd(Gi, full_matrices=False)[0]
Yi = np.c_[np.ones(k), U[:, :d], U[:, 0] ** 2, U[:, 0] * U[:, 1], U[:, 1] ** 2]
print("columns kept by sklearn (full QR):", qr(Yi)[0][:, d + 1:].shape[1])
print("columns of the Hessian estimator (economic QR):",
      qr(Yi, mode="economic")[0][:, d + 1:].shape[1], "= dp =", dp)
```

## Expected Results

`method="hessian"` implements the Hessian eigenmap of reference [2] in the docstring: per neighborhood, only the `dp` orthonormalized quadratic columns (Donoho & Grimes' `Pii = Yt(:, d+2:end)'` after a thin Gram-Schmidt). Its embedding differs from LTSA's. A standalone reimplementation of the same loop with `qr(Yi, mode="economic")` gives a different embedding (on an 800-sample swiss roll with `n_neighbors=12`, the R² of the true 2-D coordinates regressed on the embedding is `[0.831, 0.972]` for both current methods and `[0.832, 0.976]` for that HLLE).

## Actual Results

```
max |hessian - ltsa| embedding: 4.863817681943772e-14
reconstruction_error_: 0.011291345767251625 0.011291345767248632
columns kept by sklearn (full QR): 7
columns of the Hessian estimator (economic QR): 3 = dp = 3
```

## Root cause and possible fix

`sklearn/manifold/_locally_linear.py:290`:

```python
Q, R = qr(Yi)
w = Q[:, n_components + 1 :]
```

`scipy.linalg.qr` defaults to `mode="full"`. Fix: `Q = qr(Yi, mode="economic")[0]` so that `w` has exactly `dp` columns. The `S = w.sum(0)` normalization below is a no-op in both versions (all columns are orthogonal to the constant column, so their sums are ~0 and fall under `hessian_tol`), as in the reference Matlab code. The fix changes the output of `method="hessian"`.

## Versions

```
sklearn: 1.10.dev0, commit 289c4d5cfa
python: 3.13.14, numpy: 2.5.3, scipy: 1.18.1, platform: linux
```
