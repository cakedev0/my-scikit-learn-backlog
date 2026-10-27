# `LogisticRegressionCV` inherits `set_callbacks` but never calls the callbacks, which also breaks the hooks of enclosing meta-estimators

```json
{
    "estimators": ["LogisticRegressionCV", "GridSearchCV", "Pipeline", "ProgressBar", "ScoringMonitor"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 2,
    "difficulty": 1,
    "decision": 0,
    "decision_tags": []
}
```

## Classification

- **Priority (2)**: silent: callbacks set on `LogisticRegressionCV` do nothing, without the "does not support callbacks" warning other estimators get, and inside a `GridSearchCV`/`Pipeline` with an auto-propagated callback (`ProgressBar`) the evaluation and refit tasks never get their `on_fit_task_begin`/`on_fit_task_end` hooks. No wrong model, but the callbacks API gives wrong information.
- **Difficulty (1)**: make `LogisticRegressionCV` look like an estimator without callback support (hide `set_callbacks`, or warn), until it gets real support.
- **Decision (0)**: no trade-off: `set_callbacks` never worked on this class.

## Upstream search

`gh issue list --search "LogisticRegressionCV callbacks"`, `gh pr list --search "LogisticRegressionCV callback"`, `gh issue list --label Callbacks`: no report. #33847 (merged) added callback support to `LogisticRegression` only, and `_logistic_regression_path` has a `TODO(callbacks)` about adding it to `LogisticRegressionCV` later. The open PR #34904 (common callback tests) puts `LogisticRegressionCV` in its `NO_CALLBACK_SUPPORT` skip list, so it doesn't fix this either.

## Describe the bug and its user-facing impact

`LogisticRegressionCV` subclasses `LogisticRegression`, which inherits `CallbackSupportMixin`, so it has a working `set_callbacks` method. But `LogisticRegressionCV.fit` never calls `_init_callback_context` nor any hook. Consequences:

1. `LogisticRegressionCV().set_callbacks(ScoringMonitor(...)).fit(X, y)` runs without any warning and `get_logs()` then raises "No logs to retrieve".
2. Meta-estimators decide whether a sub-estimator supports callbacks with `hasattr(sub_estimator, "set_callbacks")` (`sklearn/callback/_callback_context.py:494`). For an estimator without support (e.g. `RidgeClassifierCV`) they warn and call the hooks of the leaf task themselves. For `LogisticRegressionCV` they instead propagate the callbacks and skip them in the leaf task (`_propagated_callbacks`), expecting the sub-estimator's root task to call them, which never happens. So with `GridSearchCV(LogisticRegressionCV()).set_callbacks(ProgressBar())`, none of the "candidate-split-evaluation" and "refit-with-best-params" tasks is reported as begun or finished.

## Steps/Code to Reproduce

A minimal auto-propagated callback stands in for `ProgressBar` (which needs `rich`) and counts the finished tasks:

```python
import warnings
from collections import Counter

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegressionCV, RidgeClassifierCV
from sklearn.model_selection import GridSearchCV

warnings.filterwarnings("ignore", category=FutureWarning)


class CountEnds:
    """Minimal auto-propagated callback counting finished tasks per estimator."""

    max_propagation_depth = None

    def __init__(self):
        self.ends = Counter()

    def setup(self, estimator, context):
        pass

    def teardown(self, estimator, context):
        pass

    def on_fit_task_begin(self, estimator, context):
        pass

    def on_fit_task_end(self, estimator, context):
        self.ends[(context.estimator_name, context.task_name)] += 1


X, y = make_classification(n_samples=200, random_state=0)
for est in [RidgeClassifierCV(), LogisticRegressionCV(alphas=3, solver="lbfgs")]:
    cb = CountEnds()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", UserWarning)
        GridSearchCV(est, {"fit_intercept": [True, False]}, cv=3).set_callbacks(cb).fit(X, y)
    print(type(est).__name__, "| has set_callbacks:", hasattr(est, "set_callbacks"))
    print("  warnings:", [str(x.message) for x in w if issubclass(x.category, UserWarning)])
    print("  finished tasks:", dict(cb.ends))
```

## Expected Results

`LogisticRegressionCV` is handled like `RidgeClassifierCV` (or gets real callback support): a warning that callbacks are not supported, and the 6 evaluation tasks plus the refit task are reported as finished by `GridSearchCV`.

## Actual Results

```
RidgeClassifierCV | has set_callbacks: False
  warnings: ['The auto-propagated callbacks attached to GridSearchCV will not be propagated to RidgeClassifierCV because the latter does not support callbacks.']
  finished tasks: {('GridSearchCV', 'candidate-split-evaluation'): 6, ('GridSearchCV', 'search'): 1, ('GridSearchCV', 'refit-with-best-params'): 1, ('GridSearchCV', 'fit'): 1}
LogisticRegressionCV | has set_callbacks: True
  warnings: []
  finished tasks: {('GridSearchCV', 'search'): 1, ('GridSearchCV', 'fit'): 1}
```

And for a direct fit, `LogisticRegressionCV(solver="lbfgs").set_callbacks(ScoringMonitor(scoring="accuracy")).fit(X, y)` gives no warning, then `get_logs()` raises `ValueError: No logs to retrieve. No scores were computed during the runs or the estimator is not fitted yet`.

## Root cause and possible fix

`class LogisticRegressionCV(LogisticRegression, ...)` (`sklearn/linear_model/_logistic.py:1952`) inherits `CallbackSupportMixin.set_callbacks` through `LogisticRegression` (`sklearn/linear_model/_logistic.py:1135`), but its `fit` has no callback code. The skip logic in `CallbackContext._call_hooks` (`sklearn/callback/_callback_context.py:314-319`) and `propagate_callback_context` (`sklearn/callback/_callback_context.py:485-512`) rely on "has `set_callbacks`" meaning "calls the hooks of its root task".

Short-term fix: hide `set_callbacks` on `LogisticRegressionCV` (e.g. with `available_if(lambda self: False)`, so that `hasattr` is False and meta-estimators take the "no support" path with a warning), or make it raise. A more general guard would be for `propagate_callback_context` to check, on exit, that the sub-estimator created a root context, and call the skipped hooks otherwise. The same mechanism hides failing fits: with `GridSearchCV(LogisticRegression(), {"solver": ["lbfgs"], "l1_ratio": [0, 1]}, cv=3, error_score=np.nan)`, the 3 fits with `l1_ratio=1` raise in `fit` before `_init_callback_context`, and their evaluation tasks get neither `on_fit_task_begin` nor `on_fit_task_end` (observed with the callback above). That case is less of a concern since a failed task arguably has no "end", but the guard would cover both.

## Versions

```
sklearn: 1.10.dev0, commit 289c4d5cfa (upstream/main)
python: 3.13.14, numpy: 2.5.3, scipy: 1.18.1, platform: linux
```
