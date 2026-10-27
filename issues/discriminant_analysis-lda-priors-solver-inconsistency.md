# `LinearDiscriminantAnalysis` solvers give different models when `priors` is set

```json
{
    "estimators": ["LinearDiscriminantAnalysis"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 2,
    "difficulty": 2,
    "decision": 2,
    "decision_tags": ["unclear-spec", "output-change"]
}
```

## Classification

- **Priority (2)**: with user-set `priors` (a common way to handle class imbalance), `solver="svd"` (default) and `solver="lsqr"`/`"eigen"` fit different covariance matrices and so different decision boundaries, silently. The difference is large when classes have different shapes and the priors are far from the class frequencies.
- **Difficulty (2)**: once the convention is chosen, either `_class_cov` weights by class frequencies (a few lines) or `_solve_svd` weights the centered samples by `priors_ / (n_k / n)` (a few lines, but the effect on the between-class scaling and `transform` must be checked).
- **Decision (2)**: there are two defensible conventions (`unclear-spec`). The textbook pooled within-class MLE (ESL eq. 4.10) does not depend on the priors, which is what `svd` does; `lsqr`/`eigen` and the `covariance_` docstring weight class covariances by the priors. Whichever is picked, one solver's output changes for users who set `priors` (`output-change`).

## Upstream search

GraphQL searches "LinearDiscriminantAnalysis priors covariance", "LinearDiscriminantAnalysis priors svd lsqr different", "LDA priors within-class covariance weighted", "LinearDiscriminantAnalysis svd priors inconsistent".

- #33626 (merged, "FIX Inconsistencies between solvers in LDA and QDA") explicitly lists this as an **unsolved issue** in its description: "`_solve_eigen()` weights the class covariances by the class priors, while `_solve_svd()` does not [...] they still disagree otherwise (e.g. if the priors are set manually)". The author offered to follow up but no issue or PR tracks it.
- #15640 (closed): LDA/QDA biased vs unbiased covariance, fixed by #33626, different cause.
- #12662 (open): `shrinkage="auto"` computes one Ledoit-Wolf shrinkage per class in `_class_cov`; related code, different symptom.
- #11727 (closed): eigen solver implementation questions, unrelated to priors.

## Describe the bug and its user-facing impact

LDA assumes one covariance matrix shared by all classes. With `priors=None` all three solvers agree (since #33626). When `priors` is set, they disagree:

- `solver="svd"` pools all class-centered samples, so the within-class covariance is weighted by the class *frequencies* `n_k / n`; `priors` only shifts the intercept.
- `solver="lsqr"` and `solver="eigen"` compute `covariance_ = sum_k priors_[k] * C_k`, so `priors` also changes the covariance estimate and the direction of `coef_`.

A user who sets `priors=[0.5, 0.5]` on imbalanced data gets a different classifier depending on the solver, with no warning. In the example below, the predicted probability of the minority class at the same point is 0.08 with `svd` and 0.59 with `lsqr`/`eigen`. Also, with `solver="svd", store_covariance=True`, the stored `covariance_` is prior-weighted but the coefficients are computed from the frequency-weighted one, so the fitted model is internally inconsistent.

## Steps/Code to Reproduce

```python
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

rng = np.random.RandomState(0)
X = np.vstack([rng.normal(scale=[1.0, 0.2], size=(300, 2)),
               rng.normal(scale=[0.2, 1.0], size=(30, 2)) + 1])
y = np.r_[np.zeros(300), np.ones(30)]

for priors in [None, [0.5, 0.5]]:
    print(f"priors={priors}")
    for solver in ["svd", "lsqr", "eigen"]:
        clf = LinearDiscriminantAnalysis(solver=solver, priors=priors).fit(X, y)
        print(f"  {solver:5s} coef_={clf.coef_.ravel()}  "
              f"P(y=1 | x=[1, 0])={clf.predict_proba([[1, 0]])[0, 1]:.3f}")
```

## Expected Results

The solver is an implementation detail: all solvers should fit the same model for the same `priors`, as they do with `priors=None`.

## Actual Results

```
priors=None
  svd   coef_=[1.13524908 7.11139685]  P(y=1 | x=[1, 0])=0.009
  lsqr  coef_=[1.13524908 7.11139685]  P(y=1 | x=[1, 0])=0.009
  eigen coef_=[1.13524908 7.11139685]  P(y=1 | x=[1, 0])=0.009
priors=[0.5, 0.5]
  svd   coef_=[1.13524908 7.11139685]  P(y=1 | x=[1, 0])=0.080
  lsqr  coef_=[2.02997084 1.75749652]  P(y=1 | x=[1, 0])=0.591
  eigen coef_=[2.02997084 1.75749652]  P(y=1 | x=[1, 0])=0.591
```

## Root cause and possible fix

- `sklearn/discriminant_analysis.py:177` (`_class_cov`, used by `lsqr`, `eigen` and `svd` with `store_covariance=True`): `cov += priors[idx] * _cov(Xg, ...)`.
- `sklearn/discriminant_analysis.py:596-614` (`_solve_svd`): the within-class scaling comes from the SVD of all class-centered samples concatenated (`Xc`), i.e. implicit `n_k / n` weights.

Two options:

1. Make `_class_cov` weight by `n_k / n` regardless of `priors` (the pooled MLE of the shared covariance, ESL eq. 4.10; `priors` then only affects the intercepts). This matches `svd` and changes `lsqr`/`eigen` plus the documented meaning of `covariance_`.
2. Make `_solve_svd` weight each centered sample of class k by `sqrt(priors_[k] * n / n_k)` before the SVD, matching `lsqr`/`eigen` and the current `covariance_` docstring (the direction suggested by the #33626 author).

I'd lean to option 1 since it is the textbook estimator and `priors` is usually meant as "deployment class distribution", but this needs a maintainer decision.

## Versions

```
sklearn: 1.10.dev0, commit 289c4d5cfa (upstream/main; local HEAD f570497251 has no diff in sklearn/discriminant_analysis.py)
python: 3.13, numpy: 2.5.3, scipy: 1.18.1, platform: linux
```
