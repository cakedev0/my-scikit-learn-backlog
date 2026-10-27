# QuantileRegressor crashes with `TypeError: 'NoneType' object is not subscriptable` when linprog does not succeed

```json
{
    "estimators": ["QuantileRegressor"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 1,
    "difficulty": 1,
    "decision": 1,
    "decision_tags": ["new-error"]
}
```

## Classification

- **Priority (1)**: a crash with an unhelpful error, right after a `ConvergenceWarning` that suggests the fit will go on. It needs non-default `solver_options` (iteration or time limit) or invalid input such as a negative sample weight, so few users hit it.
- **Difficulty (1)**: check `result.x is None` and raise a clear error (or fall back), a few lines.
- **Decision (1)**: `new-error` only in the sense that the warning path becomes an explicit error when HiGHS returns no point; today it already raises, just with a confusing `TypeError`. Whether to raise, or to keep the partial solution when HiGHS gives one, needs a small decision.

## Upstream search

- `gh search issues ... QuantileRegressor NoneType`, `QuantileRegressor maxiter`, `QuantileRegressor highs fails`, `linprog result.x None`: no match. Hits were #20132 and #31708 (feature requests for other solvers) and #25838 (closed, test tolerance on Windows).
- `gh search issues ... QuantileRegressor negative sample_weight`: only #15531 (open, general "enforce positive sample_weight").
- `git log -5 -- sklearn/linear_model/_quantile.py`: no related change.

## Describe the bug and its user-facing impact

When `scipy.optimize.linprog` does not succeed, `QuantileRegressor.fit` emits a `ConvergenceWarning` ("Linear programming for QuantileRegressor did not succeed") and then reads `result.x`. The HiGHS solvers (the default `"highs"` and 3 of the 4 allowed values, `"revised simplex"` being the other one) return `x=None` when they stop on an iteration limit, a time limit, or detect an unbounded problem. The next line slices `None` and the user gets `TypeError: 'NoneType' object is not subscriptable` from inside `fit`.

Triggers I found:

- `solver_options={"maxiter": ...}` or `{"time_limit": ...}` that is too small;
- a negative `sample_weight` (accepted by `_check_sample_weight`), which makes the LP unbounded.

The warning-then-continue code dates from the `interior-point` default (#9978), where `result.x` was always an array; the default switched to `"highs"` in #27943.

## Steps/Code to Reproduce

```python
import numpy as np
from sklearn.linear_model import QuantileRegressor

rng = np.random.RandomState(0)
X = rng.randn(20, 2)
y = X @ [1.0, 2.0] + rng.randn(20)

for kwargs in [{"solver_options": {"maxiter": 1}}, {"solver_options": {"time_limit": 1e-9}}]:
    try:
        QuantileRegressor(alpha=0.0, **kwargs).fit(X, y)
    except Exception as e:
        print(kwargs, "->", type(e).__name__, e)

sample_weight = np.ones(20)
sample_weight[0] = -1.0
QuantileRegressor(alpha=0.0).fit(X, y, sample_weight=sample_weight)
```

## Expected Results

Either a fitted model with the `ConvergenceWarning` (as the warning implies), or a clear error saying that the LP solver returned no solution and why (iteration limit, unbounded problem / negative weights, ...).

## Actual Results

```
/home/arthur/scikit-learn/sklearn/linear_model/_quantile.py:244: ConvergenceWarning: Linear programming for QuantileRegressor did not succeed.
Status is 1: Iteration limit reached.
Result message of linprog:
Iteration limit reached. (HiGHS Status 14: model_status is Iteration limit reached; primal_status is None)
  warnings.warn(
/home/arthur/scikit-learn/sklearn/linear_model/_quantile.py:244: ConvergenceWarning: Linear programming for QuantileRegressor did not succeed.
Status is 1: Iteration limit reached.
Result message of linprog:
Time limit reached. (HiGHS Status 13: model_status is Time limit reached; primal_status is None)
  warnings.warn(
/home/arthur/scikit-learn/sklearn/linear_model/_quantile.py:244: ConvergenceWarning: Linear programming for QuantileRegressor did not succeed.
Status is 3: Problem appears to be unbounded.
Result message of linprog:
The problem is unbounded. (HiGHS Status 10: model_status is Unbounded; primal_status is Feasible)
  warnings.warn(
{'solver_options': {'maxiter': 1}} -> TypeError 'NoneType' object is not subscriptable
{'solver_options': {'time_limit': 1e-09}} -> TypeError 'NoneType' object is not subscriptable
Traceback (most recent call last):
  File ".../q4.py", line 16, in <module>
    QuantileRegressor(alpha=0.0).fit(X, y, sample_weight=sample_weight)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/arthur/scikit-learn/sklearn/base.py", line 1403, in wrapper
    return fit_method(estimator, *args, **kwargs)
  File "/home/arthur/scikit-learn/sklearn/linear_model/_quantile.py", line 256, in fit
    params = solution[:n_params] - solution[n_params : 2 * n_params]
             ~~~~~~~~^^^^^^^^^^^
TypeError: 'NoneType' object is not subscriptable
```

## Root cause and possible fix

`sklearn/linear_model/_quantile.py:236` stores `solution = result.x`, the failure branch at lines 237-252 only warns, and line 256 slices `solution`. Fix: when `result.x is None`, raise a `ValueError` (or `ConvergenceWarning` turned error) that includes the linprog message. Negative weights could also be rejected upfront with `_check_sample_weight(..., ensure_non_negative=True)`, since the pinball loss LP is unbounded with them.

## Versions

```
sklearn: 1.10.dev0, commit 289c4d5cfa
python: 3.13, numpy: 2.5.3, scipy: 1.18.1, platform: linux
```
