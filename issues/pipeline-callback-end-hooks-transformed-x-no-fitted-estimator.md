# `Pipeline` passes the transformed `X` to its end hooks and never a `fitted_estimator`, so `ScoringMonitor` on a `Pipeline` records nothing

```json
{
    "estimators": ["Pipeline", "ScoringMonitor", "GridSearchCV", "RandomizedSearchCV"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 2,
    "difficulty": 1,
    "decision": 1,
    "decision_tags": ["unclear-spec"]
}
```

## Classification

- **Priority (2)**: custom callbacks get a different `X` in `on_fit_task_end` than in `on_fit_task_begin` for the same task, and the built-in `ScoringMonitor` silently does nothing on a `Pipeline` (the most common way to use an estimator) or a search CV, then `get_logs()` raises "No logs to retrieve". Callbacks are experimental and the user guide only shows `ScoringMonitor` on `LogisticRegression`, which limits the impact.
- **Difficulty (1)**: pass the input `X` of each task to its end hook, and `reconstruction_attributes={}` at the end of the root task (and of the refit task in the search CVs).
- **Decision (1)**: `unclear-spec`: the developer guide says estimators "are expected to provide all the values that they are capable to produce" and to pass `reconstruction_attributes={}` when nothing is missing, which argues for the fix, but the hook docstrings don't say explicitly whether `X` at `on_fit_task_end` is the task's input.

## Upstream search

`gh issue list --search "ScoringMonitor Pipeline"`, `gh pr list --search "callback Pipeline"`, `gh issue list --label Callbacks`: callback support for `Pipeline` comes from #33787 (merged), then #33999 (task renaming) and #33566 (explicit `X`, `y` in hooks). No issue about the `X` passed at the end of the tasks or `ScoringMonitor` on meta-estimators. `test_pipeline_with_callbacks` only counts the hooks, it doesn't check their arguments.

## Describe the bug and its user-facing impact

The `FitCallback` protocol documents `X` as "The training data at this task" for both `on_fit_task_begin` and `on_fit_task_end`. `Pipeline` passes:

- for its root task: the input `X` at begin, but the input of the final step (`Xt`) at end (or the output of the final step for `fit_transform`);
- for each `fit-transform-<step>` task: the step's input at begin, its output at end.

And no task of `Pipeline` passes `reconstruction_attributes`, not even the root task once the whole pipeline is fitted. `BaseSearchCV` does the same for its root task and its refit task. So `ScoringMonitor`, which only scores when it gets a `fitted_estimator`, never logs anything when registered on a `Pipeline` or a search CV, although the user guide says regular callbacks "can be registered on an estimator at any level of a composition". If `reconstruction_attributes={}` were added at the end of the `Pipeline` root task as is, `ScoringMonitor` would score the pipeline on `Xt` and fail on the shape mismatch, so both need fixing together.

## Steps/Code to Reproduce

```python
from sklearn.callback import ScoringMonitor
from sklearn.datasets import make_classification
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline


class RecordX:
    """Record the shape of X and whether a fitted estimator is passed."""

    def __init__(self):
        self.records = []

    def setup(self, estimator, context):
        pass

    def teardown(self, estimator, context):
        pass

    def on_fit_task_begin(self, estimator, context, *, X=None):
        self.records.append(("begin", context.task_name, getattr(X, "shape", None)))

    def on_fit_task_end(self, estimator, context, *, X=None, fitted_estimator=None):
        self.records.append(
            ("end", context.task_name, getattr(X, "shape", None), fitted_estimator is not None)
        )


X, y = make_classification(n_samples=100, n_features=5, random_state=0)

rec = RecordX()
pipe = make_pipeline(PCA(n_components=2), LogisticRegression()).set_callbacks(rec)
pipe.fit(X, y)
for r in rec.records:
    print(r)

monitor = ScoringMonitor(scoring="accuracy")
for est in [make_pipeline(PCA(n_components=2), LogisticRegression()),
            GridSearchCV(LogisticRegression(), {"tol": [1e-4, 1e-3]})]:
    est.set_callbacks(monitor).fit(X, y)
    try:
        monitor.get_logs()
        print(type(est).__name__, "logged")
    except ValueError as e:
        print(type(est).__name__, "->", e)
```

## Expected Results

The begin and end hooks of a task get the same `X`: `(100, 5)` for `fit` and `fit-transform-pca`. The end of the root `fit` task gets a `fitted_estimator`, so `ScoringMonitor` logs the score of the fitted pipeline (and of the fitted search), as it does for the root task of `LogisticRegression`.

## Actual Results

```
('begin', 'fit', (100, 5))
('begin', 'fit-transform-pca', (100, 5))
('end', 'fit-transform-pca', (100, 2), False)
('begin', 'fit-final-estimator', (100, 2))
('end', 'fit-final-estimator', (100, 2), False)
('end', 'fit', (100, 2), False)
Pipeline -> No logs to retrieve. No scores were computed during the runs or the estimator is not fitted yet
GridSearchCV -> No logs to retrieve. No scores were computed during the runs or the estimator is not fitted yet
```

## Root cause and possible fix

`Pipeline.fit`, `fit_transform` and `fit_predict` reassign `X` to the transformed data and call `callback_ctx.call_on_fit_task_end(estimator=self, X=Xt, y=y)` at the end of the root task (`sklearn/pipeline.py:658`, `749`, `881`; also the final step of `fit_transform` at `sklearn/pipeline.py:744`, where `Xt` is the output of the last step). `_fit_transform_one_with_callbacks` does the same for each step (`sklearn/pipeline.py:1601`). None of these calls passes `reconstruction_attributes`, and neither do `root_callback_ctx.call_on_fit_task_end` and `refit_subctx.call_on_fit_task_end` in `BaseSearchCV.fit` (`sklearn/model_selection/_search.py:1191`, `1203`).

Fix: keep the task's input around and pass it as `X` to the end hooks, and pass `reconstruction_attributes={}` at the end of the root task of `Pipeline` (whole pipeline fitted) and of `BaseSearchCV` (after the refit). For the per-step tasks, the pipeline is not ready to predict yet, so leaving `fitted_estimator` out there is fine.

## Versions

```
sklearn: 1.10.dev0, commit 289c4d5cfa (upstream/main)
python: 3.13.14, numpy: 2.5.3, scipy: 1.18.1, platform: linux
```
