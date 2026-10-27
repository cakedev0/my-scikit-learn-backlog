# LogisticRegression with newton-cd crashes on float32 multiclass data (also reached by solver="auto")

```json
{
    "estimators": ["LogisticRegression", "LogisticRegressionCV"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 2,
    "difficulty": 1,
    "decision": 0,
    "decision_tags": []
}
```

## Classification

- **Priority (2)**: loud crash with a cryptic Cython error, on a narrow combination: float32 X, 3+ classes and `l1_ratio > 0` with `n_samples < n_features` (or an explicit `solver="newton-cd"`); casting to float64 is a cheap workaround. Unreleased code (#34681), so worth fixing before the release, but not a silent wrong result. (priority review, 2026-10-23: was 3)
- **Difficulty (1)**: cast to float64 before calling the multinomial Cython solver (or validate X as float64 for that case), plus a float32 multiclass case in the dtype tests.
- **Decision (0)**: float32 users get a float64 copy of X, which is what they have to do by hand today. Making the Cython solver fused-typed is a possible follow-up.

## Upstream search

- `gh search issues ... newton-cd float32`, `... "Buffer dtype mismatch" newton-cd`: no match. #35029 (open) is about off-centered data, unrelated.
- The multinomial path was added by #34681 ("ENH Newton-CD part 8: add multinomial NewtonCDSolver", merged, unreleased). `solver="auto"` was added by #35110 (merged) and picks `"newton-cd"` for `l1_ratio > 0` and `n_samples < n_features`.

## Describe the bug and its user-facing impact

`LogisticRegression` validates X with `dtype=[float64, float32]` for all solvers, so float32 input stays float32. The binary `newton-cd` path and `newton-cd-gram` handle that, but the multinomial `newton-cd` path passes float32 arrays to `enet_coordinate_descent_multinomial`, which only accepts float64 memoryviews. `fit` then fails with `ValueError: Buffer dtype mismatch, expected 'float64_t' but got 'float'`.

Configurations I checked (dense and CSR give the same results):

| solver | classes | n_samples >= n_features | n_samples < n_features |
|---|---|---|---|
| `"auto"`, `l1_ratio=1` | 2 | ok | ok |
| `"auto"`, `l1_ratio=1` | 3 | ok (picks newton-cd-gram) | **ValueError** |
| `"newton-cd"` | 3 | **ValueError** | **ValueError** |
| `"newton-cd-gram"` | 3 | ok | ok |

float32 is a common input dtype (e.g. from deep learning embeddings or `astype(np.float32)` to save memory), and wide multiclass data is exactly the case where `"auto"` picks `"newton-cd"`. `LogisticRegressionCV` goes through the same path.

## Steps/Code to Reproduce

```python
import numpy as np
from sklearn.linear_model import LogisticRegression

rng = np.random.RandomState(0)
X = rng.randn(20, 50).astype(np.float32)  # n_samples < n_features: "auto" picks "newton-cd"
y = np.arange(20) % 3

LogisticRegression(l1_ratio=1.0).fit(X.astype(np.float64), y)  # works
LogisticRegression(l1_ratio=1.0).fit(X, y)  # fails
```

## Expected Results

The fit succeeds, as with float64 input and as with float32 input for the binary case, `newton-cd-gram` and all other solvers (`test_dtype_match` checks that float32 is supported, but only on a binary target).

## Actual Results

```
Traceback (most recent call last):
  File ".../ncd_repro.py", line 9, in <module>
    LogisticRegression(l1_ratio=1.0).fit(X, y)  # fails
  File "/home/arthur/scikit-learn/sklearn/base.py", line 1403, in wrapper
    return fit_method(estimator, *args, **kwargs)
  File "/home/arthur/scikit-learn/sklearn/linear_model/_logistic.py", line 1829, in fit
    coefs, _, _, n_iter = _logistic_regression_path(
        X,
    ...<20 lines>...
        estimator=self,
    )
  File "/home/arthur/scikit-learn/sklearn/linear_model/_logistic.py", line 741, in _logistic_regression_path
    w0 = sol.solve(X=X, y=y, sample_weight=sample_weight)
  File "/home/arthur/scikit-learn/sklearn/linear_model/_glm/_newton_solver.py", line 640, in solve
    self.inner_solve(X=X, y=y, sample_weight=sample_weight)
  File "/home/arthur/scikit-learn/sklearn/linear_model/_glm/_newton_solver.py", line 1542, in inner_solve
    _, gap, inner_tol, n_inner_iter = enet_coordinate_descent_multinomial(
        W=w,
    ...<12 lines>...
        verbose=self.verbose >= 4,
    )
  File "sklearn/linear_model/_cd_fast.pyx", line 2772, in sklearn.linear_model._cd_fast.enet_coordinate_descent_multinomial
ValueError: Buffer dtype mismatch, expected 'float64_t' but got 'float'
```

## Root cause and possible fix

- `sklearn/linear_model/_logistic.py:1713-1721`: `validate_data(..., dtype=[xp.float64, xp.float32], ...)` keeps float32 for every solver.
- `sklearn/linear_model/_glm/_newton_solver.py:1536-1557`: the multinomial branch of `NewtonCDSolver.inner_solve` passes `w = np.copy(self.coef, order="F")` (float32), `X`, `raw_prediction`, `grad_pointwise` and `proba` as is.
- `sklearn/linear_model/_cd_fast.pyx:2772` onwards: `enet_coordinate_descent_multinomial` declares all its memoryviews as `float64_t` (`cdef float64_t[::1, :] W_ = W`, `cdef const float64_t[::1, :] X_`, ...), unlike `enet_coordinate_descent` which is fused-typed.

Simplest fix: in `LogisticRegression.fit` (and `LogisticRegressionCV.fit`), validate X as float64 when `solver_ == "newton-cd"` and the target has more than 2 classes, like `liblinear` already ends up doing. Alternatively, make `enet_coordinate_descent_multinomial` fused over `floating`. Either way, `test_dtype_match` should get a multiclass target.

## Versions

```
sklearn: 1.10.dev0, commit 289c4d5cfa
python: 3.13.14, numpy: 2.5.3, scipy: 1.18.1, platform: linux
```
