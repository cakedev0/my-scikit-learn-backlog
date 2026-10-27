# newton-cg keeps two weighted copies of X for binary LogisticRegression and GLMs

```json
{
    "estimators": ["LogisticRegression", "LogisticRegressionCV", "PoissonRegressor", "GammaRegressor", "TweedieRegressor"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 1,
    "difficulty": 1,
    "decision": 1,
    "decision_tags": ["float-order"]
}
```

## Classification

- **Priority (1)**: performance only (2x X in temporary memory, ~2x slower than needed), with the non-default `newton-cg` solver, binary targets or GLMs; long-standing without complaints, and `lbfgs` is a cheap workaround. (priority review, 2026-10-23: was 2)
- **Difficulty (1)**: compute `hessp(s)` as `X.T @ (h * (X @ s))` instead of `X.T @ (hX @ s)` with a precomputed `hX = h[:, None] * X`. A dozen lines in one function.
- **Decision (1)**: the Hessian-vector product is computed in a different order, so coefficients can change at the floating-point noise level (`float-order`). I saw identical coefficients on the example below.

## Upstream search

- `gh search issues ... newton-cg memory logistic`, `... hessp memory copy X logistic`: no match. Related but different: #32162 (open, multi-threading of `LogisticRegression`), #25462 (open, newton-lsmr solver).
- `git log -S` shows the precompute was moved into `_linear_loss.py` by #21808 (loss module for LogisticRegression) and touched by #34412 (array API for newton-cg). `_logistic.py` had the same `d[:, np.newaxis] * X` precompute since the first `logistic_regression_path` implementation, so this is long-standing.

## Describe the bug and its user-facing impact

For a non-multiclass loss (binary `LogisticRegression`, `PoissonRegressor`, `GammaRegressor`, `TweedieRegressor`), `LinearModelLoss.gradient_hessian_product` builds `hX = hess_pointwise[:, None] * X`, a full n_samples x n_features copy, once per Newton iteration, only to compute `hessp(s) = X.T @ (hX @ s)`. In `_newton_cg`, the previous iteration's `fhess_p` closure (and its `hX`) is still referenced while the new one is built, so two copies of X are alive at the peak.

The precompute doesn't save any work: `X.T @ (h * (X @ s))` has the same two matrix-vector products plus an O(n_samples) multiply, and building `hX` costs an extra pass over X per Newton iteration. The multiclass branch of the same function already computes the product on the fly, which is why its memory is ~0.13x X.

Measured with `tracemalloc` on a 40 MB float64 X (50000 x 100):

| estimator | peak temporary memory |
|---|---|
| `LogisticRegression(solver="newton-cg")`, binary | 2.05 x X |
| `LogisticRegression(solver="newton-cg")`, 3 classes | 0.13 x X |
| `LogisticRegression(solver="lbfgs")`, binary | 0.04 x X |
| `PoissonRegressor(solver="newton-cg")` | 2.04 x X |
| `PoissonRegressor(solver="lbfgs")` | 0.04 x X |

I monkeypatched `gradient_hessian_product` (binary branch only, scratch script, private API) to compute `X.T @ (h * (X @ s))` instead: the peak dropped from 2.05x to 0.05x X, binary `LogisticRegression(solver="newton-cg").fit` went from 0.115 s to 0.056 s (loaded laptop, `OMP_NUM_THREADS=2`), and the first coefficients were identical to 8 digits.

## Steps/Code to Reproduce

```python
import tracemalloc

import numpy as np
from sklearn.linear_model import LogisticRegression, PoissonRegressor

rng = np.random.RandomState(0)
X = rng.randn(50_000, 100)  # 40 MB
y = rng.randint(3, size=X.shape[0])
for name, est, yy in [
    ("LogisticRegression newton-cg, binary", LogisticRegression(solver="newton-cg"), y % 2),
    ("LogisticRegression newton-cg, 3 classes", LogisticRegression(solver="newton-cg"), y),
    ("LogisticRegression lbfgs, binary", LogisticRegression(solver="lbfgs"), y % 2),
    ("PoissonRegressor newton-cg", PoissonRegressor(solver="newton-cg"), y),
    ("PoissonRegressor lbfgs", PoissonRegressor(solver="lbfgs"), y),
]:
    tracemalloc.start()
    est.fit(X, yy)
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    print(f"{name:<42} peak temporary memory = {peak / X.nbytes:.2f} x X.nbytes")
```

## Expected Results

The temporary memory of `newton-cg` should be O(n_samples + n_features) on top of X, as for its multiclass path and for `lbfgs`, not 2x X.

## Actual Results

```
LogisticRegression newton-cg, binary       peak temporary memory = 2.05 x X.nbytes
LogisticRegression newton-cg, 3 classes    peak temporary memory = 0.13 x X.nbytes
LogisticRegression lbfgs, binary           peak temporary memory = 0.04 x X.nbytes
PoissonRegressor newton-cg                 peak temporary memory = 2.04 x X.nbytes
PoissonRegressor lbfgs                     peak temporary memory = 0.04 x X.nbytes
```

## Root cause and possible fix

`sklearn/linear_model/_linear_loss.py:838-875`, binary branch of `gradient_hessian_product`:

```python
            if sparse.issparse(X):
                hX = (
                    sparse.dia_array((hess_pointwise, 0), shape=(n_samples, n_samples))
                    @ X
                )
            else:
                hX = hess_pointwise[:, None] * X
            ...
            def hessp(s):
                ...
                        ret[:n_features] = np.linalg.multi_dot(
                            [X.T, hX, s[:n_features]]
                        )
```

`hX` is only used through `hX @ s` and `hX_sum = hX.sum(axis=0)`. Both can be computed without it: `hess_pointwise * (X @ s)` and `X.T @ hess_pointwise`. The sparse branch has the same issue (a reweighted sparse copy of X).

`sklearn/utils/optimize.py:381` (`fgrad, fhess_p = grad_hess(xk, *args)`) keeps the previous `fhess_p` alive while the new one is built, which doubles the peak. With the fix above the closure only holds O(n_samples) arrays, so this no longer matters.

## Versions

```
sklearn: 1.10.dev0, commit 289c4d5cfa
python: 3.13.14, numpy: 2.5.3, scipy: 1.18.1, platform: linux
```
