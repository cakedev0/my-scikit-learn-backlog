# MLPRegressor's squared error loss averages over outputs but its gradient sums over them

```json
{
    "estimators": ["MLPRegressor"],
    "existing_issue": "https://github.com/scikit-learn/scikit-learn/issues/8349",
    "reproduced": true,
    "priority": 1,
    "difficulty": 1,
    "decision": 1,
    "decision_tags": ["output-change"]
}
```

## Classification

- **Priority (1)**: only multi-output regression; `loss_`/`loss_curve_` are on a different scale than documented and L-BFGS/`tol`-based stopping is slightly off, but the measured effect on the fitted model is tiny (coef error 2e-4 vs 7e-7 against the exact solution, no measurable test MSE change). (priority review, 2026-10-23: was 2)
- **Difficulty (1)**: replace `.mean()` by `.sum()` in `squared_loss` so the loss matches the gradient and the user guide, and extend `test_gradient` to multi-output regression.
- **Decision (1)**: `output-change`: `loss_`, `loss_curve_` and the iterations where sgd/adam/lbfgs stop change for multi-output regression.

## Upstream search

Queries: "MLPRegressor multioutput loss", "MLP squared_loss mean", "MLPRegressor multi-output loss".

- #8349 (open since 2017, "Bug in bfgs gradient computation of MLPRegressor with multiple output neurons"): this exact bug. A recent comment compares two fixes (`.sum()` in the loss vs scaling the deltas) and notes the fix does not measurably improve test MSE; the argument is the L-BFGS loss/gradient contract.
- #34830 (open PR, "FIX MLP squared_error loss now matches the gradient for multi-output targets"): implements the `.sum()` fix with a parametrized gradient test; not merged at the commit below.

## Describe the bug and its user-facing impact

For `MLPRegressor` with `n_outputs > 1`, the loss function `squared_loss` returns `0.5 * mean over outputs of the weighted mean over samples`, while the backpropagated gradient (`deltas[last] = y_pred - y`, then summed over samples and divided by `sw_sum`) is the gradient of `0.5 * sum over outputs`. The user guide documents the sum: `Loss = 1/(2n) sum_i ||y_hat_i - y_i||_2^2 + alpha/(2n) ||W||_2^2`. The L2 term is consistent with the sum version.

Consequences, all for multi-output regression only:

- `scipy.optimize.minimize(method="L-BFGS-B")` receives a loss and a gradient that do not match (the data part of the gradient is `n_outputs` times too large compared to the loss), which breaks its line search and convergence checks. In the reproducer, a linear `MLPRegressor` (no hidden layer) fitted jointly on 3 outputs stops ~270x farther from the exact ridge solution than the same model fitted on each output separately.
- `loss_` and `loss_curve_` do not report the objective being minimized (the data term is divided by `n_outputs`, the penalty is not).
- sgd/adam stop on `loss_curve_` improvements smaller than `tol`, so their stopping criterion is effectively `n_outputs` times stricter on the data term.

`test_gradient` only checks `MLPClassifier`, where `log_loss` and `binary_log_loss` sum over outputs. `poisson_loss` also sums, so only `squared_loss` is affected.

## Steps/Code to Reproduce

```python
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor

rng = np.random.RandomState(0)
X = rng.randn(200, 5)
Y = X @ rng.randn(5, 3) + 0.1 * rng.randn(200, 3)

# No hidden layer + identity output: the MLP is a ridge regression, and the 3
# outputs are 3 independent problems.
params = dict(hidden_layer_sizes=[], solver="lbfgs", alpha=50.0, tol=1e-12,
              max_iter=10_000, random_state=0)
joint = MLPRegressor(**params).fit(X, Y)
separate = [MLPRegressor(**params).fit(X, Y[:, k]) for k in range(3)]
ridge = Ridge(alpha=50.0).fit(X, Y)

print("joint fit:     max |coef - ridge coef| =",
      np.abs(joint.coefs_[0] - ridge.coef_.T).max())
print("separate fits: max |coef - ridge coef| =",
      max(np.abs(m.coefs_[0].ravel() - ridge.coef_[k]).max()
          for k, m in enumerate(separate)))
print("joint loss_:          ", joint.loss_)
print("sum of separate loss_:", sum(m.loss_ for m in separate))

# Gradient check (private API): the gradient returned to L-BFGS is not the
# gradient of the returned loss when there are several outputs (alpha=0 to
# isolate the data term).
joint.set_params(alpha=0.0)
from scipy.optimize import approx_fprime
from sklearn.neural_network._multilayer_perceptron import _pack

coef = _pack(joint.coefs_, joint.intercepts_)
args = (X, Y, None, [X, None], [None], [np.empty((5, 3))], [np.empty(3)])
loss = lambda c: joint._loss_grad_lbfgs(c, *args)[0]
grad = joint._loss_grad_lbfgs(coef, *args)[1].copy()
print("analytic / numerical gradient:",
      np.median(grad / approx_fprime(coef, loss, 1e-6)))
```

The gradient check uses private API because there is no public access to the loss and gradient passed to L-BFGS; the first part only uses public API.

## Expected Results

The joint fit should reach the ridge solution as precisely as the separate fits, `joint.loss_` should equal the sum of the separate `loss_` (the outputs are independent and the documented loss sums over them), and the analytic / numerical gradient ratio should be 1.

## Actual Results

```
joint fit:     max |coef - ridge coef| = 0.00020004481080665437
separate fits: max |coef - ridge coef| = 7.483033850341769e-07
joint loss_:           1.0281465851711769
sum of separate loss_: 1.199935710696102
analytic / numerical gradient: 3.000006837515002
```

## Root cause and possible fix

`sklearn/neural_network/_base.py:187-189`:

```python
return (
    0.5 * np.average((y_true - y_pred) ** 2, weights=sample_weight, axis=0).mean()
)
```

The `.mean()` over outputs should be `.sum()`, like in `poisson_loss`, `log_loss` and `binary_log_loss`. The gradient in `_backprop` (`sklearn/neural_network/_multilayer_perceptron.py`, `deltas[last] = activations[-1] - y`) already matches the sum and the user guide, so it should not change. Single-output results are unchanged; multi-output `loss_`/`loss_curve_` values get multiplied by roughly `n_outputs` (data term), and the iterations where the solvers stop change.

## Versions

```
sklearn: 1.10.dev0, commit 289c4d5cfa (upstream/main; ran on f570497251, no diff near the code involved)
python: 3.13.14, numpy: 2.5.3, scipy: 1.18.1, platform: linux
```
