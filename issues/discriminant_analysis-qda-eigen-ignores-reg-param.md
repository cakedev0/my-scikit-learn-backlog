# `QuadraticDiscriminantAnalysis(solver="eigen")` silently ignores `reg_param`

```json
{
    "estimators": ["QuadraticDiscriminantAnalysis"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 2,
    "difficulty": 1,
    "decision": 1,
    "decision_tags": ["new-error", "output-change"]
}
```

## Classification

- **Priority (2)**: a regularization parameter is silently ignored for one solver, so a grid search over `reg_param` with `solver="eigen"` returns identical models without any hint. The `eigen` solver is new in 1.8.0 and not the default, which limits the number of affected users.
- **Difficulty (1)**: either apply the same transformation to the eigenvalues in `_solve_eigen`, or raise when `reg_param != 0` with `solver="eigen"`, as is done for `shrinkage` with `solver="svd"`.
- **Decision (1)**: applying `reg_param` changes the predictions of current `eigen` fits with `reg_param > 0` (`output-change`); raising instead breaks code that sets both (`new-error`). Both are fine since the current behavior is a no-op.

## Upstream search

GraphQL searches "QuadraticDiscriminantAnalysis reg_param eigen", "QuadraticDiscriminantAnalysis tol", "QuadraticDiscriminantAnalysis LinAlgError not full rank".

- #32108 (merged, added `solver="eigen"`, `shrinkage` and `covariance_estimator` to QDA in 1.8.0): `reg_param` is only discussed for the `svd` solver.
- #32590 (open) / #33738 (open PR): add `shrinkage` to the `svd` solver, and mention `reg_param` could be deprecated in favor of `shrinkage`. Neither mentions that `reg_param` is a no-op with `eigen`.

## Describe the bug and its user-facing impact

`reg_param` is documented without any solver restriction: "Regularizes the per-class covariance estimates by transforming S2 as `S2 = (1 - reg_param) * S2 + reg_param * np.eye(n_features)`". It is only applied in `_solve_svd`. With `solver="eigen"`, any `reg_param` value gives exactly the same model as `reg_param=0`, without warning or error.

The opposite combination (`solver="svd"` with `shrinkage`) raises `NotImplementedError`, and `covariance_estimator` with `svd` raises `ValueError`, so the silent no-op is inconsistent with the rest of the class.

## Steps/Code to Reproduce

```python
import numpy as np
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

rng = np.random.RandomState(0)
X = rng.normal(size=(100, 3))
y = np.repeat([0, 1], 50)
X[y == 1] += 1

for solver in ["svd", "eigen"]:
    p0 = QuadraticDiscriminantAnalysis(solver=solver).fit(X, y).predict_proba(X)
    p1 = QuadraticDiscriminantAnalysis(solver=solver, reg_param=0.9).fit(X, y).predict_proba(X)
    print(f"{solver:5s} max |proba(reg_param=0.9) - proba(reg_param=0)| = {np.abs(p1 - p0).max():.3f}")
```

## Expected Results

`reg_param=0.9` changes the predictions with both solvers, or `solver="eigen"` with `reg_param != 0` raises like `solver="svd"` with `shrinkage` does.

## Actual Results

```
svd   max |proba(reg_param=0.9) - proba(reg_param=0)| = 0.175
eigen max |proba(reg_param=0.9) - proba(reg_param=0)| = 0.000
```

## Root cause and possible fix

`sklearn/discriminant_analysis.py:1049` applies `scaling = (1 - reg_param) * scaling + reg_param` in `QuadraticDiscriminantAnalysis._solve_svd`; `_solve_eigen` (line 1014) never reads `self.reg_param`, and `fit` does not validate the combination.

Options:

1. Apply the same transformation to the eigenvalues in `_solve_eigen` (and to the returned `cov` so that `covariance_` stays consistent). It then composes with `shrinkage` / `covariance_estimator`.
2. Raise in `fit` when `solver="eigen"` and `reg_param != 0`, pointing to `shrinkage`, and document `reg_param` as `svd`-only.

Option 1 is the least surprising for users; option 2 fits the plan in #32590 to eventually replace `reg_param` by `shrinkage`.

## Versions

```
sklearn: 1.10.dev0, commit 289c4d5cfa (upstream/main; local HEAD f570497251 has no diff in sklearn/discriminant_analysis.py)
python: 3.13, numpy: 2.5.3, scipy: 1.18.1, platform: linux
```
