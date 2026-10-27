# ElasticNet(l1_ratio=0, positive=True) never converges: the ridge dual gap ignores the positivity constraint

```json
{
    "estimators": ["ElasticNet", "ElasticNetCV", "enet_path"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 1,
    "difficulty": 1,
    "decision": 0,
    "decision_tags": []
}
```

## Classification

- **Priority (1)**: the coefficients are correct, but every fit runs `max_iter` epochs (1000 instead of ~6 here), raises a `ConvergenceWarning`, and reports a wrong `dual_gap_`. Only for the pure L2 case with `positive=True`, which the docs already discourage (`l1_ratio <= 0.01` "is not reliable").
- **Difficulty (1)**: replace `||X^T R||^2` by `||max(X^T R, 0)||^2` in formulation B when `positive` is set, in the dense, sparse and Gram gap functions.
- **Decision (0)**: the gap becomes correct; no change to the coefficients beyond stopping earlier.

## Upstream search

- `gh search issues ... ElasticNet positive l1_ratio converge`, `positive ridge dual gap coordinate descent`, `ElasticNet l1_ratio=0 ConvergenceWarning`: no matching issue. Related: #32845 (merged, "make coordinate descent solver converge for ridge regression") introduced formulation B for `alpha == 0` but did not handle `positive`. #19620 (merged) compared Ridge and `ElasticNet(l1_ratio=0)`.
- `git log -8 -- sklearn/linear_model/_cd_fast.pyx`: no later change to this branch.

## Describe the bug and its user-facing impact

Since #32845, the coordinate descent solvers use a dedicated dual gap ("formulation B") when there is no L1 penalty (`l1_ratio=0`). That dual is the one of the unconstrained ridge problem. With `positive=True`, the optimum has `X^T R - beta * w < 0` on the clipped coordinates, so this gap stays strictly positive at the solution. The solver then never meets `tol`: it always runs `max_iter` epochs, warns, and stores a meaningless `dual_gap_`.

The coefficients themselves are right (they match a bounded least-squares reference), so the impact is wasted time and a spurious warning that tells the user to increase `max_iter` or the regularization. Dense, sparse and `precompute=True` paths all behave the same.

## Steps/Code to Reproduce

```python
import warnings
import numpy as np
import scipy.sparse as sp
from scipy.optimize import lsq_linear
from sklearn.linear_model import ElasticNet

rng = np.random.RandomState(0)
n_samples, n_features = 50, 5
X = rng.randn(n_samples, n_features)
y = X @ [1.0, -2.0, 3.0, -1.0, 0.5] + 0.1 * rng.randn(n_samples)
alpha = 0.1

# Reference: min 1/(2n) ||y - Xw||^2 + alpha/2 ||w||^2  s.t. w >= 0
A = np.vstack([X, np.sqrt(n_samples * alpha) * np.eye(n_features)])
b = np.concatenate([y, np.zeros(n_features)])
print("reference coef:", lsq_linear(A, b, bounds=(0, np.inf)).x.round(6))

for name, Xfit, kw in [("dense", X, {}), ("sparse", sp.csr_matrix(X), {}), ("gram", X, {"precompute": True})]:
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        m = ElasticNet(alpha=alpha, l1_ratio=0.0, positive=True, fit_intercept=False, **kw).fit(Xfit, y)
    print(f"{name:6s} coef: {m.coef_.round(6)}  n_iter_={m.n_iter_}  dual_gap_={m.dual_gap_:.3g}  ConvergenceWarning={len(rec) > 0}")

m = ElasticNet(alpha=alpha, l1_ratio=0.0, positive=False, fit_intercept=False).fit(X, y)
print("positive=False:  n_iter_ =", m.n_iter_, " dual_gap_ =", m.dual_gap_)
```

## Expected Results

The solver stops after a few epochs with a gap below `tol` and no warning, as it does with `positive=False`.

## Actual Results

```
reference coef: [0.649588 0.       2.544209 0.       0.192162]
dense  coef: [0.649588 0.       2.544209 0.       0.192162]  n_iter_=1000  dual_gap_=21.4  ConvergenceWarning=True
sparse coef: [0.649588 0.       2.544209 0.       0.192162]  n_iter_=1000  dual_gap_=21.4  ConvergenceWarning=True
gram   coef: [0.649588 0.       2.544209 0.       0.192162]  n_iter_=1000  dual_gap_=21.4  ConvergenceWarning=True
positive=False:  n_iter_ = 6  dual_gap_ = 8.934080142353196e-11
```

## Root cause and possible fix

Formulation B in `gap_enet` (`sklearn/linear_model/_cd_fast.pyx:275`), `gap_enet_sparse` (`:800`) and `gap_enet_gram` (`:1259`):

```python
dual = -0.5 * R_norm2 + Ry - 1 / (2 * beta) * dual_norm_XtA
```

with `dual_norm_XtA = ||X^T R||_2^2`. For `min 0.5 ||y - Xw||^2 + 0.5 beta ||w||^2` s.t. `w >= 0`, the Lagrangian minimized over `w >= 0` gives `w = max(X^T theta, 0) / beta`, hence the dual

```
D(theta) = -0.5 ||theta||^2 + theta @ y - 1 / (2 beta) * ||max(X^T theta, 0)||^2
```

So the fix is to sum only the positive entries of `XtA` when `positive` is true (the `positive` flag is already passed to these functions). The formulation A branch (`alpha > 0`) already handles `positive` by using `max(XtA)` instead of `abs_max(XtA)`.

## Versions

```
sklearn: 1.10.dev0, commit 289c4d5cfa
python: 3.13, numpy: 2.5.3, scipy: 1.18.1, platform: linux
```
