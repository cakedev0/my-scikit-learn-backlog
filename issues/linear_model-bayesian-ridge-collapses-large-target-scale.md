# BayesianRidge and ARDRegression silently predict the mean when `y` has a large scale

```json
{
    "estimators": ["BayesianRidge", "ARDRegression"],
    "existing_issue": "https://github.com/scikit-learn/scikit-learn/issues/17624",
    "reproduced": true,
    "priority": 2,
    "difficulty": 2,
    "decision": 1,
    "decision_tags": ["output-change"]
}
```

## Classification

- **Priority (2)**: silent wrong model (all coefficients ~0, R2 = 0, no warning) under default parameters, for targets in "large" units relative to the features (e.g. prices in dollars with standardized features and a few hundred samples). The workaround (scale `y`, or set `lambda_init`) is cheap once you know, but nothing hints at it.
- **Difficulty (2)**: the fix is a one-line change of the default initialization, but its effect on the fixed-point iteration needs some care to argue and test (e.g. that it never lands in a worse stationary point on regular data).
- **Decision (1)**: `output-change`: a different default `lambda_init` (documented as 1) changes results in the collapsed cases and possibly the number of iterations elsewhere. The docstring of `lambda_init` must be updated too.

## Upstream search

- `gh issue list --search "BayesianRidge"`: #17624 (open, labels Documentation and module:linear_model) "BayesianRidge fails when input and output data are of very different sizes" reports this exact symptom (with `y` ~1e6, `fit_intercept=False`). In the discussion, a contributor explains it as the effect of the default Gamma hyperpriors (`lambda_2=1e-6` "prefers low weights"), and @agramfort suggested improving the docstring. A docs-only PR, #34712 ("DOC Document BayesianRidge/ARD scale sensitivity"), was closed unmerged.
- #33757 (closed, `predict(return_std=True)` centering), #31093 (closed, covariance matrix), #24313 (closed, std with uniform weights), #13852 (open, `scores_` formula): different bugs.

**New argument** compared to #17624: the collapse is not caused by the prior, but by the scale-unaware initialization `lambda_ = 1`. With the same default hyperpriors, starting from `lambda_init=1 / var(y)` converges to the proper solution, whose log marginal likelihood is much higher than the collapsed one (-1176.0 vs -1306.5 in the reproducer below; -30.5 vs -32.3 on the example of #17624, which then gives `coef_ = [1e6, 1e6]` as expected). So the default fit stops at a worse stationary point of the objective it is supposed to maximize, which a docstring note can't fix.

## Describe the bug and its user-facing impact

`BayesianRidge` initializes the noise precision with `alpha_ = 1 / var(y)` but the weight precision with `lambda_ = 1`, whatever the scale of `y`. The initial penalty ratio `lambda_ / alpha_ = var(y)` is huge when `y` is large, so the first posterior mean is ~0. The MacKay updates then stay at a degenerate stationary point where the coefficients are ~0 and all the variance is attributed to noise. The fit "converges" (in 2 iterations by default, since the stopping criterion is an absolute change of `coef_` below `tol=1e-3`) and returns a model that predicts the mean.

That point has a much lower log marginal likelihood than the proper solution, so it's not a modeling choice, the optimizer is just stuck. `ARDRegression` has the same initialization (`lambda_ = np.ones(n_features)`) and the same failure.

With standardized features, the collapse happens from `std(y)` of about 1e4-1e5 for 30-100 samples, ~1e5-1e6 for 300-1000 samples (growing roughly linearly with `n_samples`). Monetary targets in dollars or cents easily reach that.

## Steps/Code to Reproduce

```python
import numpy as np
from sklearn.linear_model import ARDRegression, BayesianRidge, LinearRegression

rng = np.random.RandomState(0)
X = rng.randn(100, 3)
y = X @ [3.0, 2.0, 1.0] + rng.randn(100)

for scale in [1, 1e4, 3e4, 1e5]:
    for model in [LinearRegression(), BayesianRidge(), ARDRegression()]:
        model.fit(X, y * scale)
        print(f"y * {scale:<6g} {type(model).__name__:16} R2 = {model.score(X, y * scale):.3f}, "
              f"coef_ / scale = {np.round(model.coef_ / scale, 2)}")

# The collapsed solution is a worse optimum of the evidence than the proper one
y_large = y * 3e4
default = BayesianRidge(compute_score=True).fit(X, y_large)
rescaled = BayesianRidge(compute_score=True, lambda_init=1 / np.var(y_large)).fit(X, y_large)
for name, m in [("default lambda_init=1", default), ("lambda_init=1/var(y)", rescaled)]:
    print(f"{name:22} R2 = {m.score(X, y_large):.3f}  "
          f"log marginal likelihood = {m.scores_[-1]:.1f}")
```

## Expected Results

`BayesianRidge` and `ARDRegression` are (up to the very weak default Gamma hyperpriors) equivariant to the scale of `y`: multiplying `y` by a constant multiplies `coef_` by the same constant and doesn't change R2, as for `LinearRegression`. At least, the fit should not stop at a stationary point with a much lower marginal likelihood than a nearby one.

## Actual Results

```
y * 1      LinearRegression R2 = 0.941, coef_ / scale = [2.96 1.95 1.04]
y * 1      BayesianRidge    R2 = 0.941, coef_ / scale = [2.95 1.94 1.04]
y * 1      ARDRegression    R2 = 0.941, coef_ / scale = [2.96 1.94 1.03]
y * 10000  LinearRegression R2 = 0.941, coef_ / scale = [2.96 1.95 1.04]
y * 10000  BayesianRidge    R2 = 0.941, coef_ / scale = [2.95 1.94 1.04]
y * 10000  ARDRegression    R2 = 0.941, coef_ / scale = [2.96 1.94 1.03]
y * 30000  LinearRegression R2 = 0.941, coef_ / scale = [2.96 1.95 1.04]
y * 30000  BayesianRidge    R2 = 0.000, coef_ / scale = [0. 0. 0.]
y * 30000  ARDRegression    R2 = 0.000, coef_ / scale = [0. 0. 0.]
y * 100000 LinearRegression R2 = 0.941, coef_ / scale = [2.96 1.95 1.04]
y * 100000 BayesianRidge    R2 = 0.000, coef_ / scale = [0. 0. 0.]
y * 100000 ARDRegression    R2 = 0.000, coef_ / scale = [0. 0. 0.]
default lambda_init=1  R2 = 0.000  log marginal likelihood = -1306.5
lambda_init=1/var(y)   R2 = 0.941  log marginal likelihood = -1176.0
```

No warning is raised in any of these fits. With `tol=1e-12` the default fit runs 75 iterations and still ends at `coef_ ~ 0`, so it's not only the absolute stopping criterion.

## Root cause and possible fix

`sklearn/linear_model/_bayes.py:265-269` (upstream main 289c4d5cfa):

```python
if alpha_ is None:
    alpha_ = 1.0 / (y_var + eps)
if lambda_ is None:
    lambda_ = 1.0
```

and `sklearn/linear_model/_bayes.py:684` for `ARDRegression` (`lambda_ = np.ones(n_features)`). The first update of `lambda_` is `(gamma + 2 * lambda_1) / (||coef||^2 + 2 * lambda_2)`. When the initial coefficients are tiny (in units of `y`), `gamma ~ 0` and `||coef||^2` is comparable to `2 * lambda_2 = 2e-6`, so `lambda_` stays ~1 and the iteration is stuck.

A scale-aware default fixes the reproducer, e.g. `lambda_init = 1 / var(y)` (initial penalty ratio of 1, i.e. starting close to least squares). Another option is to fit on `y / std(y)` and rescale `coef_`, `alpha_`, `lambda_` and `sigma_` afterwards, which makes the estimator exactly scale-equivariant but changes the meaning of the Gamma hyperpriors (`alpha_1`, `alpha_2`, `lambda_1`, `lambda_2`) with respect to the units of `y`. Either way the documented default "If not set, lambda_init is 1" changes. The convergence check (absolute change of `coef_` below `tol`) has the same unit problem and could be made relative at the same time.

## Versions

```
sklearn: 1.10.dev0, commit f570497251 (sklearn/linear_model/_bayes.py identical on upstream/main 289c4d5cfa)
python: 3.13.14, numpy: 2.5.3, scipy: 1.18.1, platform: linux
```
