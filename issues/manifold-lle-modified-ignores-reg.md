# `LocallyLinearEmbedding(method="modified")` ignores `reg` and hardcodes `1e-3`

```json
{
    "estimators": ["LocallyLinearEmbedding", "locally_linear_embedding"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 1,
    "difficulty": 1,
    "decision": 0,
    "decision_tags": []
}
```

## Classification

- **Priority (1)**: a documented parameter is silently ignored, but only for MLLE and only when it differs from the default (the hardcoded value equals the default `reg=1e-3`).
- **Difficulty (1)**: replace the literal `1e-3` by the `reg` argument (plus the same `trace == 0` fallback as `barycenter_weights`).
- **Decision (0)**: no change at the default value; users who set `reg` get what they asked for.

## Upstream search

Queries: "LocallyLinearEmbedding reg ignored", "locally linear embedding modified reg parameter". Only #29715 (closed, `n_neighbors <= n_samples` check), unrelated. The line dates back to the original MLLE implementation (#211, 2011).

## Describe the bug and its user-facing impact

`reg` is documented as "Regularization constant, multiplies the trace of the local covariance matrix of the distances", without restriction on `method`. In the `method="modified"` branch, the regularized LLE weights `w_reg` are computed with a local variable that overwrites the argument:

```python
reg = 1e-3 * evals.sum(1)
```

So `reg` has no effect on `fit` / `fit_transform` for MLLE, while it is still used by `transform` (which always uses `barycenter_kneighbors_graph(..., reg=self.reg)`). A user tuning `reg` for MLLE (e.g. in a grid search) sees identical embeddings for every value.

## Steps/Code to Reproduce

```python
import numpy as np
from sklearn.datasets import make_swiss_roll
from sklearn.manifold import LocallyLinearEmbedding

X, _ = make_swiss_roll(n_samples=300, random_state=0)
for method in ["standard", "modified"]:
    Y1, Y2 = [
        LocallyLinearEmbedding(
            n_neighbors=10, n_components=2, method=method, reg=reg, eigen_solver="dense"
        ).fit_transform(X)
        for reg in [1e-3, 1.0]
    ]
    print(f"{method}: max diff between reg=1e-3 and reg=1.0: "
          f"{np.abs(np.abs(Y1) - np.abs(Y2)).max():.3g}")
```

## Expected Results

The MLLE embedding depends on `reg`, as the standard LLE one does. The `trace(G)` of the local Gram matrix used by standard LLE is `evals.sum(1)` here, so `reg * evals.sum(1)` is the same regularization.

## Actual Results

```
standard: max diff between reg=1e-3 and reg=1.0: 0.182
modified: max diff between reg=1e-3 and reg=1.0: 0
```

## Root cause and possible fix

`sklearn/manifold/_locally_linear.py:336` shadows the `reg` argument with `1e-3 * evals.sum(1)`. Rename the local variable and use `reg * evals.sum(1)` (falling back to `reg` where the trace is 0, as `barycenter_weights` does). Identical results at the default `reg=1e-3`.

The docstring could also say that `reg` is not used by `method="hessian"` and `method="ltsa"` in `fit` (they have no regularized least-squares step).

## Versions

```
sklearn: 1.10.dev0, commit 289c4d5cfa
python: 3.13.14, numpy: 2.5.3, scipy: 1.18.1, platform: linux
```
