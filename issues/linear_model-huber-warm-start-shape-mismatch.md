# HuberRegressor with `warm_start=True` silently returns garbage when `n_features` or `fit_intercept` changed

```json
{
    "estimators": ["HuberRegressor"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 1,
    "difficulty": 1,
    "decision": 0,
    "decision_tags": []
}
```

## Classification

- **Priority (1)**: silent wrong model, but only when refitting a warm-started `HuberRegressor` on a different number of features or after flipping `fit_intercept`, which is rare (and the first case is a user error that should raise).
- **Difficulty (1)**: build the initial parameter vector from the current `fit_intercept` and check `coef_.shape` against `X.shape[1]`, a few lines.
- **Decision (0)**: the fixed cases are all wrong today; raising for a feature-count mismatch is what `SGDRegressor` already does.

## Upstream search

- `gh search issues ... HuberRegressor warm_start`: nothing relevant (#8230, #25522 are general warm-start discussions).
- `gh search issues ... HuberRegressor` (30 results): #27777 (open) and PR #34891 (open) are about lbfgs line search failures raising `ValueError`; #32949 (closed) boolean `X`. PR #34891 touches `_huber.py` but not the warm-start initialization.

## Describe the bug and its user-facing impact

The private loss `_huber_loss_and_gradient` decides whether there is an intercept from the length of the parameter vector (`fit_intercept = n_features + 2 == w.shape[0]`). With `warm_start=True`, `fit` always builds the initial vector as `[coef_, intercept_, scale_]` from the previous fit, without looking at the current `fit_intercept` or at the current number of features. Two consequences:

1. Refit on one more feature: the old vector has length `n_old + 2 = n_new + 1`, so the loss treats it as "no intercept". `coef_` is then the no-intercept solution and `intercept_` is set to `parameters[-2]`, which is the last coefficient. The model is meaningless and no error is raised (`SGDRegressor` raises `X has 3 features, but SGDRegressor is expecting 2 features`).
2. Flip `fit_intercept` from `True` to `False`: the vector still has length `n + 2`, so L-BFGS optimizes an intercept, then `fit` sets `intercept_ = 0.0` and keeps coefficients that were fitted assuming an intercept. `predict` is then worse than a cold `fit_intercept=False` fit.

## Steps/Code to Reproduce

```python
import numpy as np
from sklearn.linear_model import HuberRegressor

rng = np.random.RandomState(0)
X = rng.randn(100, 3)
y = X @ [1.0, 2.0, 3.0] + 10.0 + 0.1 * rng.randn(100)

# Case 1: refit on more features with warm_start=True
warm = HuberRegressor(warm_start=True).fit(X[:, :2], y).fit(X, y)
cold = HuberRegressor().fit(X, y)
print("warm coef:", warm.coef_, "intercept:", warm.intercept_)
print("cold coef:", cold.coef_, "intercept:", cold.intercept_)

# Case 2: switch fit_intercept from True to False with warm_start=True
warm = HuberRegressor(warm_start=True).fit(X, y)
warm.set_params(fit_intercept=False).fit(X, y)
cold = HuberRegressor(fit_intercept=False).fit(X, y)
print("warm coef:", warm.coef_, "MSE:", np.mean((warm.predict(X) - y) ** 2))
print("cold coef:", cold.coef_, "MSE:", np.mean((cold.predict(X) - y) ** 2))
```

## Expected Results

Case 1 should raise a clear error about the feature mismatch (or ignore the stale `coef_`). Case 2 should give the same model as the cold fit, since `warm_start` only changes the initialization of a convex problem.

## Actual Results

```
warm coef: [2.13653503 3.17988185 1.58746245] intercept: 1.5874624524408876
cold coef: [0.99404854 1.98913832 3.00779195] intercept: 9.981712025820592
warm coef: [0.99404857 1.98913829 3.00779192] MSE: 99.6697231544668
cold coef: [2.13653283 3.17987469 1.58749388] MSE: 95.65006719199143
```

In case 1 the "intercept" equals the third coefficient. In case 2 the warm coefficients are those of the model with intercept.

## Root cause and possible fix

`sklearn/linear_model/_huber.py:302`:

```python
if self.warm_start and hasattr(self, "coef_"):
    parameters = np.concatenate((self.coef_, [self.intercept_, self.scale_]))
```

and the shape-based dispatch at `sklearn/linear_model/_huber.py:55`. Fix: when warm starting, check `self.coef_.shape[0] == X.shape[1]` (raise otherwise, as SGD does) and only append `intercept_` when `self.fit_intercept` is true. Passing `fit_intercept` explicitly to the loss instead of inferring it from `w.shape` would make the helper less fragile too.

For comparison, the GLMs (`PoissonRegressor` etc., `_glm/glm.py:372`) handle the `fit_intercept` flip correctly and fail on a feature-count change with a cryptic `matmul` shape error rather than silently.

## Versions

```
sklearn: 1.10.dev0, commit 289c4d5cfa
python: 3.13, numpy: 2.5.3, scipy: 1.18.1, platform: linux
```
