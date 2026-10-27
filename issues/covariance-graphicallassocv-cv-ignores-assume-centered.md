# `GraphicalLassoCV` ignores `assume_centered` when scoring the alpha grid

```json
{
    "estimators": ["GraphicalLassoCV"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 1,
    "difficulty": 1,
    "decision": 1,
    "decision_tags": ["output-change"]
}
```

## Classification

- **Priority (1)**: silent, but only matters with `assume_centered=True` on data whose mean is not close to zero (with a mean of exactly zero the difference vanishes). The selected `alpha_` and `cv_results_` then come from a different model than the one refitted.
- **Difficulty (1)**: pass `assume_centered` down to `graphical_lasso_path` (a module-level helper, not exported from `sklearn.covariance`) and use it for both the train and test empirical covariances.
- **Decision (1)**: the selected `alpha_` and reported scores change for users of `assume_centered=True` (`output-change`). Centering each test fold on the training mean, like `EmpiricalCovariance.score` does, would also change the scores with the default `assume_centered=False` (slightly).

## Upstream search

GraphQL searches "GraphicalLassoCV assume_centered", "graphical_lasso_path assume_centered", "GraphicalLassoCV centering cross-validation", "GraphicalLassoCV alpha selection wrong". No match (#13456, merged, only fixed the `assume_centered` docstring wording; #34888 is an unrelated array API PR).

## Describe the bug and its user-facing impact

`GraphicalLassoCV(assume_centered=True)` uses the non-centered second moment matrix to build the default alpha grid (`alpha_max(emp_cov)`) and for the final refit. But the cross-validation that picks `alpha_` calls `graphical_lasso_path`, which has no `assume_centered` parameter and always centers both the training fold and the test fold (`empirical_covariance(X)` and `empirical_covariance(X_test)`). The `alpha = 0` entry of `cv_results_` is computed with `cross_val_score(EmpiricalCovariance(), ...)`, also ignoring `assume_centered`.

So with `assume_centered=True`, `cv_results_` and `alpha_` describe centered models while `covariance_` / `precision_` are fitted without centering. In the example, `GraphicalLassoCV` picks `alpha_ = 0.174` while cross-validating `GraphicalLasso(assume_centered=True)` on the same grid and folds picks `0.284`, and the reported scores are off by ~2 nats.

## Steps/Code to Reproduce

```python
import warnings
import numpy as np
from sklearn.covariance import GraphicalLasso, GraphicalLassoCV
from sklearn.datasets import make_sparse_spd_matrix
from sklearn.model_selection import KFold

warnings.simplefilter("ignore")  # ConvergenceWarning noise
rng = np.random.RandomState(0)
cov = np.linalg.inv(make_sparse_spd_matrix(5, alpha=0.8, random_state=0))
X = rng.multivariate_normal(np.zeros(5), cov, size=60) + 3.0

cv = KFold(3)
gl_cv = GraphicalLassoCV(assume_centered=True, cv=cv).fit(X)
alphas = gl_cv.cv_results_["alphas"][:-1]  # last entry is alpha=0
manual = [
    np.mean([GraphicalLasso(alpha=a, assume_centered=True).fit(X[tr]).score(X[te])
             for tr, te in cv.split(X)])
    for a in alphas
]
print("GraphicalLassoCV scores:", gl_cv.cv_results_["mean_test_score"][:5].round(2))
print("manual CV scores       :", np.round(manual[:5], 2))
print("alpha_                 :", gl_cv.alpha_)
print("best alpha, manual CV  :", alphas[np.argmax(manual)])
```

## Expected Results

`cv_results_["mean_test_score"]` matches the cross-validated score of `GraphicalLasso(alpha=a, assume_centered=True)`, and `alpha_` is the best alpha of that cross-validation.

## Actual Results

```
GraphicalLassoCV scores: [-7.27 -7.27 -7.21 -7.15 -7.14]
manual CV scores       : [-12.86  -9.96  -9.15  -9.09  -9.1 ]
alpha_                 : 0.17378071369231057
best alpha, manual CV  : 0.2840108357399399
```

## Root cause and possible fix

- `sklearn/covariance/_graph_lasso.py:666`: `emp_cov = empirical_covariance(X)` in `graphical_lasso_path` (always centered).
- `sklearn/covariance/_graph_lasso.py:675`: `test_emp_cov = empirical_covariance(X_test)` centers the test fold on its own mean. Even with the default `assume_centered=False` this differs from `GraphicalLasso.score`, which centers the test data on the training `location_` (with the default the selected alpha is barely affected in the example: 0.183 vs 0.185).
- `sklearn/covariance/_graph_lasso.py:1019`: `GraphicalLassoCV.fit` does not forward `self.assume_centered`.
- `sklearn/covariance/_graph_lasso.py:1091`: the `alpha = 0` score uses `EmpiricalCovariance()` without `assume_centered`.

Fix: add `assume_centered=False` to `graphical_lasso_path`, use it for the training covariance, and score the test fold with `empirical_covariance(X_test - location, assume_centered=True)` where `location` is the training mean (or zero), as `EmpiricalCovariance.score` does. Forward the parameter from `GraphicalLassoCV.fit`, including to the `EmpiricalCovariance` used for `alpha = 0`.

## Versions

```
sklearn: 1.10.dev0, commit 289c4d5cfa (upstream/main; local HEAD f570497251 has no diff in sklearn/covariance/)
python: 3.13, numpy: 2.5.3, scipy: 1.18.1, platform: linux
```
