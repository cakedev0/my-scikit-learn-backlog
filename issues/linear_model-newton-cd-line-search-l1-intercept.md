# Newton-CD solvers count the intercept in the L1 term of the line search and crash with `ConvergenceWarning` raised as an exception

```json
{
    "estimators": ["LogisticRegression", "LogisticRegressionCV", "PoissonRegressor", "GammaRegressor", "TweedieRegressor"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 3,
    "difficulty": 1,
    "decision": 0,
    "decision_tags": []
}
```

## Classification

- **Priority (3)**: on main, `solver="auto"` now picks `"newton-cd-gram"` or `"newton-cd"` for any `l1_ratio > 0`, so a plain `LogisticRegression(l1_ratio=0.5).fit(X, y)` raises on small, unscaled, multiclass data (~20-40% of random seeds in the setting below). Not released yet, so it's worth fixing before 1.10.
- **Difficulty (1)**: the Armijo term must use the penalized weights only (`linear_loss.weight_intercept(coef)`), the same way `LinearModelLoss.loss` and `min_norm_subgradient` already do. Two lines.
- **Decision (0)**: the fix makes the line search consistent with the objective it minimizes. Results only change for fits where the wrong Armijo term changed which step was accepted.

## Upstream search

Queries (`gh search issues --include-prs`): "newton-cd line search", "NewtonCDGramSolver", "newton-cd intercept", "newton-cd ConvergenceWarning fallback". Only the PRs that added the solvers come up, no issue about the line search or the L1 fallback.

Related merged PRs: #34377 (NewtonCDGramSolver), #34561 (NewtonCDSolver), #34681 (multinomial NewtonCDSolver), #35043 (divide by zero in the line search), #35110 (`solver="auto"`). None of them touches the L1 term of the Armijo condition.

## Describe the bug and its user-facing impact

`NewtonSolver.line_search` adds `l1_reg_strength * (||coef + coef_newton||_1 - ||coef_old||_1)` to the Armijo term, where `coef` includes the intercept(s). The intercept is not penalized in the objective (`LinearModelLoss.loss` only applies `l1_penalty` to `weights`). So whenever the Newton step moves an intercept towards 0, the expected decrease is overestimated by `l1_reg_strength * |change of |intercept||`, the sufficient decrease condition can't be met, the line search gives up and the solver falls back to lbfgs. With a non-zero L1 penalty there is no fallback, and `fallback_lbfgs_solve` does `raise ConvergenceWarning(msg)`, so `fit` crashes with a warning class raised as an exception (and right after a warning saying "It will now resort to lbfgs instead").

The opposite case (intercept moving away from 0) makes the Armijo condition too loose, so steps that don't decrease the objective enough can be accepted.

It hits both `"newton-cd"` and `"newton-cd-gram"`, binary and multinomial `LogisticRegression` and the GLMs with `l1_ratio > 0`. It's much more frequent with unscaled features (large intercepts relative to the coefficients) and for multinomial problems. With the data below, over 100 seeds, `newton-cd` fails 20 times and `newton-cd-gram` 37 times. Since #35110, `solver="auto"` (the default) selects these solvers as soon as `l1_ratio > 0`.

## Steps/Code to Reproduce

```python
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

X, y = make_classification(
    n_samples=20, n_features=5, n_informative=5, n_redundant=0, n_classes=3,
    class_sep=2.0, random_state=0,
)
X = 100 * X  # unscaled features

saga = LogisticRegression(l1_ratio=0.5, solver="saga", max_iter=10_000).fit(X, y)
print("saga converged in", saga.n_iter_, "epochs")

LogisticRegression(l1_ratio=0.5).fit(X, y)  # solver="auto" -> "newton-cd-gram"
```

## Expected Results

The problem is well posed (saga converges to the solution) and the default solver should fit it, or at worst return with a `ConvergenceWarning` like the other solvers do.

## Actual Results

```
sklearn/linear_model/_glm/_newton_solver.py:505: ConvergenceWarning: Line search of Newton solver NewtonCDGramSolver at iteration #12 did not converge after 21 line search refinement iterations. It will now resort to lbfgs instead.
  warnings.warn(
saga converged in [2292] epochs
Traceback (most recent call last):
  ...
  File "sklearn/linear_model/_logistic.py", line 741, in _logistic_regression_path
    w0 = sol.solve(X=X, y=y, sample_weight=sample_weight)
  File "sklearn/linear_model/_glm/_newton_solver.py", line 664, in solve
    self.fallback_lbfgs_solve(X=X, y=y, sample_weight=sample_weight)
  File "sklearn/linear_model/_glm/_newton_solver.py", line 1027, in fallback_lbfgs_solve
    raise ConvergenceWarning(msg)
sklearn.exceptions.ConvergenceWarning: This solver, NewtonCDGramSolver, does not have a fallback for non-zero L1 penalties.
```

With `verbose=2`, the failing line search shows a loss decrease that stays at ~0.55x the Armijo term for every step size, while the Armijo term uses `sigma = 1/2**11`: the predicted decrease is ~3600x too steep, which is impossible for a convex L1 term unless the term itself is wrong.

## Root cause and possible fix

`sklearn/linear_model/_glm/_newton_solver.py:365-372` (commit 289c4d5cfa):

```python
if self.l1_reg_strength > 0:
    L1 = self.linear_loss.l1_penalty(
        self.coef + self.coef_newton, self.l1_reg_strength
    )
    L1_old = self.linear_loss.l1_penalty(self.coef_old, self.l1_reg_strength)
    armijo_term += sigma * (L1 - L1_old)
```

`l1_penalty` expects `weights` (it is called with `weights` in `LinearModelLoss.loss`), but here it gets the full `coef`, intercepts included. Fix: strip the intercepts first, e.g.

```python
weights_new, _ = self.linear_loss.weight_intercept(self.coef + self.coef_newton)
weights_old, _ = self.linear_loss.weight_intercept(self.coef_old)
L1 = self.linear_loss.l1_penalty(weights_new, self.l1_reg_strength)
L1_old = self.linear_loss.l1_penalty(weights_old, self.l1_reg_strength)
```

I checked this by patching `l1_penalty` to drop the intercepts when called from `line_search`: failures go from 124/400 to 0/400 (multinomial, data above with `class_sep` in {2, 5}), from 11/200 to 0/200 (binary), and from 1/200 to 0/200 for `PoissonRegressor(alpha=0.1, l1_ratio=0.5)` on `100 * randn(20, 5)` features (the GLMs are much less affected).

Separately, `fallback_lbfgs_solve` raising `ConvergenceWarning` (a `UserWarning` subclass) as an exception is surprising: once the line search is fixed it should be rare, but returning the current iterate with a warning would be friendlier than crashing.

## Versions

```
sklearn: 1.10.dev0, commit 289c4d5cfa (upstream/main, pure-Python files overlaid on the local build: no compiled file differs in sklearn/linear_model)
python: 3.13.14, numpy: 2.5.3, scipy: 1.18.1, platform: linux
```
