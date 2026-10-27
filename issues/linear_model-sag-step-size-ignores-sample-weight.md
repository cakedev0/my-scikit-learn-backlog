# SAG/SAGA step size ignores `sample_weight`: Ridge(solver="sag") diverges with weights >= 3

```json
{
    "estimators": ["Ridge", "RidgeClassifier", "LogisticRegression", "ridge_regression"],
    "existing_issue": "https://github.com/scikit-learn/scikit-learn/issues/31536",
    "reproduced": true,
    "priority": 2,
    "difficulty": 2,
    "decision": 1,
    "decision_tags": ["output-change", "perf-low"]
}
```

## Classification

- **Priority (2)**: diverging fits for weights above ~3, but only with an explicit `solver="sag"`/`"saga"` (`"auto"` never picks sag in `Ridge` nor in `LogisticRegression`), and the failure comes with a `ConvergenceWarning` or an overflow error. Already tracked in #31536 with a fix in progress (#31675). (priority review, 2026-10-23: was 3)
- **Difficulty (2)**: the fix is local (`max_squared_sum` should be `max_i sw_i * ||x_i||^2`, and the intercept term scaled the same way), but it touches the shared `sag_solver`, used by Ridge and LogisticRegression, and the step size choice needs checking against the SAG/SAGA papers for weighted sums.
- **Decision (1)**: `output-change`: a smaller step changes the iterates (and the number of iterations) for every weighted fit, including those that converge today. `perf-low`: with weights spread over a wide range, the step gets smaller and convergence can be slower.

## Upstream search

Queries (issues and PRs): "sag step size sample_weight", "sag sample_weight diverge", "SAG sample weights Lipschitz", "sag sample_weight", "SAG step size", "get_auto_step_size".

- #31536 (open, "Improve sample_weight handling in sag(a)"): general issue about `sample_weight` in SAG/SAGA, which this file is a concrete instance of. It is mostly discussed through the frequency-weights equivalence test (weighted sampling vs. gradient rescaling), not through divergence.
- #31675 (open PR, "Fix sample weight handling in SAG(A)"): fixes #31536; its description lists "Calculates the step size by accounting for sample weights", i.e. the root cause below, together with weighted sampling.
- #31837 (draft PR, helper for #31675): corrects the Lipschitz constant in the test reference implementation of `get_step_size` for `sample_weight`.
- #21305 (open, "LogisticRegression with SAGA using sample_weight does not converge"): same family of symptoms, discussion focuses on uniform sampling; #34799 (open PR) adds weighted sampling for it, but does not seem to change the step size.
- #23180 (open, "Investigate SAG/SAGA solver"): general convergence concerns with Ridge.
- #32944 (closed PR, not merged): rescaled `alpha` for Ridge sag with `sample_weight`, not the step size.

So the bug is known and a fix is in progress in #31675, but none of these threads shows the simple divergence case (uniform `sample_weight=3`) or reports that Ridge returns ~1e33 coefficients with only a `ConvergenceWarning`. The reproducer below could serve as a non-regression test for #31675.

## Describe the bug and its user-facing impact

`sag_solver` computes the step size from the largest squared row norm of `X` (`sklearn/linear_model/_sag.py:302`, then `L = max_squared_sum + int(fit_intercept) + alpha_scaled` at line 68 for the squared loss, line 65 for the log loss). But the per-sample gradients are multiplied by `sample_weight`, so the Lipschitz constant of sample `i` is `sw_i * ||x_i||^2`, not `||x_i||^2`. As soon as the weights are larger than ~1 (relative to the margin the 1/L step has), the step is too large and SAG diverges.

For `Ridge`, the result is either a `ValueError: Floating-point under-/overflow occurred at epoch #...`, or worse, a model with huge coefficients and only a `ConvergenceWarning` (see below with uniform weights of 3). SAGA has a smaller step (`1 / (2L + mu n)`) and survives a bit longer but diverges for weights of ~10.

The problem is invariant to rescaling (weights `c * sw` with `alpha * c` give the same solution), so the solver behavior should not depend on the scale of the weights, but it does: dividing weights and `alpha` by 3 makes SAG converge.

`LogisticRegression(solver="sag"/"saga")` uses the same step size (log loss, `0.25 * max_squared_sum`). On my local build (whose `_logistic.py` differs from upstream/main, but `_sag.py` is identical), with `X` scaled by 3 and `sample_weight=100`, both solvers hit `max_iter` and their `coef_` are off by ~6 to ~11 compared to lbfgs.

## Steps/Code to Reproduce

```python
import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import Ridge

X, y = make_regression(n_samples=100, n_features=5, noise=1, random_state=0)
sample_weight = np.full(100, 3.0)  # same problem as repeating each row 3 times

ref = Ridge(alpha=1.0, solver="cholesky").fit(X, y, sample_weight=sample_weight)
print("cholesky:", np.round(ref.coef_, 2))
for solver in ["sag", "saga"]:
    m = Ridge(alpha=1.0, solver=solver, random_state=0)
    try:
        m.fit(X, y, sample_weight=sample_weight)
        print(f"{solver}:", m.coef_, m.n_iter_)
    except ValueError as e:
        print(f"{solver}: ValueError: {e}")
# same problem, weights and alpha divided by 3: converges fine
m = Ridge(alpha=1.0 / 3, solver="sag", random_state=0).fit(X, y, sample_weight=sample_weight / 3)
print("sag, rescaled:", np.round(m.coef_, 2), m.n_iter_)
```

## Expected Results

SAG converges to the cholesky solution, as it does after rescaling weights and `alpha` by the same factor.

## Actual Results

```
/home/arthur/scikit-learn/sklearn/linear_model/_sag.py:343: ConvergenceWarning: The max_iter was reached which means the coef_ did not converge
  warnings.warn(
cholesky: [45.46 85.55 97.7  11.7  42.16]
sag: [ 1.31721980e+33 -1.14810058e+33 -6.43662090e+32  3.47842760e+33
  4.53142221e+33] [1000]
saga: [45.46384526 85.54753009 97.69975448 11.69261894 42.15491589] [55]
sag, rescaled: [45.46 85.55 97.7  11.69 42.16] [46]
```

With `sample_weight=np.full(100, 10.0)`, both `sag` and `saga` raise `ValueError: Floating-point under-/overflow occurred at epoch #817` (resp. `#151`).

## Root cause and possible fix

`sklearn/linear_model/_sag.py:302`: `max_squared_sum = row_norms(X, squared=True).max()` ignores `sample_weight`, and callers that precompute it do the same (`sklearn/linear_model/_ridge.py:751`, `sklearn/linear_model/_logistic.py:1809` on upstream/main). The intercept term `int(fit_intercept)` in `get_auto_step_size` has the same problem.

Possible fix: use `max_i sw_i * (||x_i||^2 + fit_intercept)` as the Lipschitz constant of the data term in `get_auto_step_size` (computed in `sag_solver` when `sample_weight` is given, and in the callers that pass `max_squared_sum`). An alternative, closer to what `Ridge` does for other solvers, is to normalize the weights to mean 1 and rescale `alpha` accordingly, but that alone does not handle weights with a large spread.

See also: `linear_model-sag-sparse-intercept-decay-early-stop.md` (another SAG/SAGA convergence problem, sparse intercept decay; different root cause).

## Versions

```
sklearn: 1.10.dev0, commit 289c4d5cfa (upstream/main; reproduced on a local build whose _sag.py and _ridge.py solver code are identical)
python: 3.13, numpy: 2.5.3, scipy: 1.18.1, platform: linux
```
