# `LogisticRegression().set_callbacks(...)` warns that callbacks are ignored with the default `solver="auto"`, but they are not

```json
{
    "estimators": ["LogisticRegression"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 2,
    "difficulty": 1,
    "decision": 0,
    "decision_tags": []
}
```

## Classification

- **Priority (2)**: a wrong warning on the default configuration of the only estimator with per-iteration callback support, telling users their callbacks won't work. It's a regression on main only (#35110 was merged on 2026-10-20), so it should be fixed before 1.10.
- **Difficulty (1)**: a condition to update in `set_callbacks`, plus ideally a check on the resolved `solver_` in `fit`.
- **Decision (0)**: no trade-off.

## Upstream search

`gh pr list --search "set_callbacks solver auto"`, `--search "callbacks LogisticRegression"` (open), `gh issue list --label Callbacks`: no match. The warning comes from the callbacks API (#33322) and was correct until `solver="auto"` became the default in #35110 (merged 2026-10-20), which did not update it. The test `test_logistic_regression_callback_support_warning` (`sklearn/linear_model/tests/test_logistic.py:3522`) only checks `solver="liblinear"`.

## Describe the bug and its user-facing impact

Since `solver="auto"` is the default of `LogisticRegression`, `LogisticRegression().set_callbacks(cb)` always warns:

> Callbacks are only supported in LogisticRegression for solver='lbfgs'. This estimator has solver='auto', callbacks will be ignored during the solving of the regression.

But `"auto"` resolves to `"lbfgs"` for the default `l2` penalty, and the callbacks do run on each lbfgs iteration. Every user who tries callbacks on a default `LogisticRegression` gets told they won't work. Conversely, when `"auto"` resolves to `newton-cd`/`newton-cd-gram` (`l1_ratio > 0`), callbacks are indeed ignored, but the message blames `solver='auto'`, which the user cannot act on.

## Steps/Code to Reproduce

```python
import warnings

from sklearn.callback import ScoringMonitor
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

X, y = make_classification(n_samples=200, random_state=0)
monitor = ScoringMonitor(scoring="accuracy")

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    clf = LogisticRegression().set_callbacks(monitor)
print([str(x.message) for x in w])

clf.fit(X, y)
print("solver_:", clf.solver_)
print("rows logged per lbfgs iteration:",
      sum(r["task_name"] == "lbfgs-iter" for r in monitor.get_logs().data))
```

## Expected Results

No warning: the default solver resolves to lbfgs and the per-iteration hooks are called.

## Actual Results

```
["Callbacks are only supported in LogisticRegression for solver='lbfgs'. This estimator has solver='auto', callbacks will be ignored during the solving of the regression."]
solver_: lbfgs
rows logged per lbfgs iteration: 9
```

## Root cause and possible fix

`LogisticRegression.set_callbacks` (`sklearn/linear_model/_logistic.py:1570`) checks `self.solver != "lbfgs"`, written before `"auto"` existed. Since the solver is only known after `_check_solver` (it depends on the penalty and on the shape of `X`), the check could be `self.solver not in ("lbfgs", "auto")` in `set_callbacks`, plus a warning in `fit` when `self.solver_ != "lbfgs"` and callbacks are set. The latter also covers `set_params(solver=...)` after `set_callbacks`, which currently never warns.

## Versions

```
sklearn: 1.10.dev0, commit 289c4d5cfa (upstream/main)
python: 3.13.14, numpy: 2.5.3, scipy: 1.18.1, platform: linux
```
