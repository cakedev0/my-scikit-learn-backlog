# LogisticRegression with sag/saga on sparse X stops early with a wrong intercept and no warning

```json
{
    "estimators": ["LogisticRegression", "LogisticRegressionCV"],
    "existing_issue": "https://github.com/scikit-learn/scikit-learn/issues/25198",
    "reproduced": true,
    "priority": 2,
    "difficulty": 2,
    "decision": 1,
    "decision_tags": ["output-change"]
}
```

## Classification

- **Priority (2)**: silent wrong results, on default parameters (`tol`, `max_iter`), with the solver the docs recommend for large sparse data. Sparse features are usually non-negative (counts, one-hot, TF-IDF), so the intercept matters, and the fit returns a near-zero intercept that looks plausible. It has been known and open since 2022 (#25198), which is why I don't put it higher.
- **Difficulty (2)**: removing the `0.01` intercept decay for SAG/SAGA (dense and sparse then give the same iterates) is a one-line change in `sag_solver`, but one has to check why the decay was there (it comes from the SGD solver, where it stabilizes the intercept of sparse data) and that it is not needed with SAG's averaged gradients. Adding the intercept to the stopping criterion is a small Cython change.
- **Decision (1)**: `output-change` for every sparse sag/saga fit with an intercept (the results move to the actual solution, the one the dense path already finds).

## Upstream search

Queries: "saga sparse intercept", "sag sparse intercept convergence", "sag intercept_decay".

- #25198 (open, 2022): "Convergence issues of SAGA solver in LogisticRegression with sparse data - SPARSE_INTERCEPT_DECAY". Same root cause. @TomDLT explained it and listed options: make the decay a parameter (A), change or remove the heuristic (B), make the stopping criterion more conservative about the intercept (C). Nobody picked it up. Related issues listed there: #13995, #11178, #13246.
- #23180 (open) "Investigate SAG/SAGA solver", #31536 / #31675 (open) on sample_weight in SAG: different topics.

New arguments for #25198:

1. With default `tol` and `max_iter`, the fit stops early and silently (no `ConvergenceWarning`), because the stopping criterion ignores the intercept. The report in #25198 used `max_iter=4000`; here the defaults are enough to get an intercept of ~0.02 instead of ~2.
2. The decay comes from SGD (Bottou), where a sparse feature is only updated when it is non-zero in the drawn sample. In SAG/SAGA, every coordinate receives the averaged gradient step (just in time, through the lagged updates), so the argument for damping the intercept doesn't carry over. Removing the decay gives exactly the dense iterates (checked below), which is what users expect from a change of input format.

## Describe the bug and its user-facing impact

For sparse `X`, `sag_solver` gets `intercept_decay = SPARSE_INTERCEPT_DECAY = 0.01` from `make_dataset`, so each intercept update is 100x smaller than with dense `X`. On top of that, the stopping criterion of the SAG/SAGA loop only looks at the coefficients (`max |w - w_prev| / max |w| <= tol`), not at the intercept. When the features are not tiny (e.g. counts), the coefficients settle quickly while the intercept is still crawling from 0, the criterion is met after a few dozen epochs, and the solver returns a far-off intercept without any `ConvergenceWarning`.

Below, the true intercept is 2 and lbfgs finds 1.976. Dense sag/saga find 1.94/1.90 (still slowly converging, as the docs warn for unscaled features). Sparse sag/saga on the same data stop after ~30 epochs with an intercept of ~0.02 and a lower training accuracy.

## Steps/Code to Reproduce

```python
import warnings

import numpy as np
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression

# Sparse count-like features (values 10, 20, 30, ...) and a true intercept of 2.
rng = np.random.RandomState(0)
X = sp.random(1000, 20, density=0.1, format="csr", random_state=rng,
              data_rvs=lambda k: 10 * (1 + rng.poisson(1, k)))
y = rng.uniform(size=1000) < 1 / (1 + np.exp(-(2 + X @ rng.randn(20) / 10)))

for solver in ["lbfgs", "sag", "saga"]:
    for X_ in [X.toarray(), X]:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            clf = LogisticRegression(solver=solver, max_iter=1000, random_state=0).fit(X_, y)
        print(f"{solver:5} {type(X_).__name__:10} n_iter_={clf.n_iter_[0]:4d} "
              f"intercept_={clf.intercept_[0]:.3f} accuracy={clf.score(X_, y):.3f} "
              f"warnings={[type(x.message).__name__ for x in w]}")
```

## Expected Results

Sparse and dense inputs give the same fit (as for lbfgs), or at least sparse sag/saga keep iterating (or warn) until the intercept has converged.

## Actual Results

```
lbfgs ndarray    n_iter_=  20 intercept_=1.976 accuracy=0.870 warnings=[]
lbfgs csr_matrix n_iter_=  20 intercept_=1.976 accuracy=0.870 warnings=[]
sag   ndarray    n_iter_= 210 intercept_=1.942 accuracy=0.870 warnings=[]
sag   csr_matrix n_iter_=  28 intercept_=0.014 accuracy=0.836 warnings=[]
saga  ndarray    n_iter_= 338 intercept_=1.903 accuracy=0.870 warnings=[]
saga  csr_matrix n_iter_=  33 intercept_=0.033 accuracy=0.837 warnings=[]
```

## Root cause and possible fix

Commit 289c4d5cfa:

- `sklearn/linear_model/_base.py:104` returns `intercept_decay = SPARSE_INTERCEPT_DECAY` (0.01) for sparse `X`, and `sklearn/linear_model/_sag.py:299` passes it to the SAG loop.
- `sklearn/linear_model/_sag_fast.pyx.tp:317,321` multiply the averaged intercept gradient step by `intercept_decay`.
- `sklearn/linear_model/_sag_fast.pyx.tp:400-404`: the convergence check loops over `n_features * n_classes` weights only and ignores `intercept_array`.

I checked the first point by monkeypatching `sklearn.linear_model._sag.make_dataset` to return `intercept_decay=1.0`: sparse sag/saga then reproduce the dense results exactly (210 and 338 epochs, intercepts 1.942 and 1.903).

The decay comes from the SGD solver, where sparse features are updated rarely and the intercept every step. SAG/SAGA apply the averaged gradient to every coordinate just in time, so the fixed point is the same with or without the decay and the decay only slows the intercept down. Possible fix: use `intercept_decay=1.0` in `sag_solver`, and include the intercept in the stopping criterion so that a slow intercept can never be hidden.

`Ridge(solver="sag")` is not affected on the same kind of data (with sparse input and an intercept it doesn't run the SAG loop, `n_iter_` is `None`, and finds the right intercept), and `Ridge(solver="saga")` rejects sparse input with `fit_intercept=True`.

## Versions

```
sklearn: 1.10.dev0, commit 289c4d5cfa (upstream/main, pure-Python files overlaid on the local build: no compiled file differs in sklearn/linear_model)
python: 3.13.14, numpy: 2.5.3, scipy: 1.18.1, platform: linux
```
