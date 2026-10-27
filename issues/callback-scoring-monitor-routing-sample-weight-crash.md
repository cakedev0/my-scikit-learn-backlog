# `ScoringMonitor` makes `fit(..., sample_weight=...)` crash when metadata routing is enabled

```json
{
    "estimators": ["ScoringMonitor", "LogisticRegression", "StandardScaler", "GridSearchCV"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 1,
    "difficulty": 2,
    "decision": 1,
    "decision_tags": ["unclear-spec"]
}
```

## Classification

- **Priority (1)**: loud error, only when combining two opt-in features (experimental callbacks and metadata routing) with `sample_weight`. The callbacks x routing design is still open (#34068, #34137) and will likely rework this path. (priority review, 2026-10-23: was 2)
- **Difficulty (2)**: the hook needs to decide how the metadata it receives is routed to the scorer, which touches the callbacks x routing design.
- **Decision (1)**: `unclear-spec`: whether the monitored score should be weighted, and how users opt in or out, is part of the open design discussion in #34068. Short of that, the crash could be avoided by not going through `process_routing` for the metadata the estimator itself forwards.

## Upstream search

`gh issue list --search "ScoringMonitor"`, `--label Callbacks`, `gh pr list --search "ScoringMonitor"`. Related, not a report of this crash:

- #34068 (open, "Callbacks x Metadata-routing"): design discussion on how metadata should reach callbacks. It does not mention that the current `ScoringMonitor` raises under routing.
- #34137 (open PR, "ENH Make callbacks metadata routers"): reworks how hooks receive metadata (callbacks get `set_on_fit_task_end_request`), which would change this code path. Under review, not merged.

## Describe the bug and its user-facing impact

`ScoringMonitor.on_fit_task_end` calls `self._scorer(fitted_estimator, X, y, **metadata)`, where `metadata` is what the estimator passes to its hooks, e.g. `{"sample_weight": sw}` for `LogisticRegression`, `StandardScaler` and the search CVs. `self._scorer` is always a `_MultimetricScorer`. With routing enabled, `_MultimetricScorer.__call__` calls `process_routing`, and since the inner scorer has no request for `sample_weight`, it raises `UnsetMetadataPassedError`. With routing disabled, the same code works and weights the scores.

So any `fit` with `sample_weight` and a `ScoringMonitor` crashes under routing, including inside `GridSearchCV(est.set_fit_request(sample_weight=True)).fit(X, y, sample_weight=sw)`. The error tells the user to call `set_score_request` on `make_scorer(accuracy_score, ...)`, an object they don't have when they passed `scoring="accuracy"`.

## Steps/Code to Reproduce

```python
import numpy as np
import sklearn
from sklearn.callback import ScoringMonitor
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

sklearn.set_config(enable_metadata_routing=True)
X, y = make_classification(n_samples=200, random_state=0)
sample_weight = np.random.RandomState(0).rand(200)

monitor = ScoringMonitor(scoring="accuracy")
LogisticRegression().set_callbacks(monitor).fit(X, y, sample_weight=sample_weight)
```

## Expected Results

`fit` succeeds, as it does with routing disabled (where the per-iteration scores are weighted by `sample_weight`).

## Actual Results

```
Traceback (most recent call last):
  ...
  File "sklearn/linear_model/_logistic.py", line 224, in __call__
  File "sklearn/callback/_callback_context.py", line 438, in call_on_fit_task_end
  File "sklearn/callback/_callback_context.py", line 354, in _call_hooks
  File "sklearn/callback/_scoring_monitor.py", line 169, in on_fit_task_end
  File "sklearn/metrics/_scorer.py", line 145, in __call__
  File "sklearn/utils/_metadata_requests.py", line 2031, in process_routing
  File "sklearn/utils/_metadata_requests.py", line 1318, in route_params
  File "sklearn/utils/_metadata_requests.py", line 844, in _route_params
  File "sklearn/utils/_metadata_requests.py", line 587, in _route_params
sklearn.exceptions.UnsetMetadataPassedError: [sample_weight] are passed but are not explicitly set as requested or not requested for make_scorer(accuracy_score, response_method='predict').score, which is used within _MultimetricScorer.score. Call `make_scorer(accuracy_score, response_method='predict').set_score_request({metadata}=True/False)` for each metadata you want to request/ignore. See the Metadata Routing User guide <https://scikit-learn.org/stable/metadata_routing.html> for more information.
```

The same error is raised from `_scoring_monitor.py` by `GridSearchCV(est, {"max_iter": [2, 3]}, cv=2).fit(X, y, sample_weight=sw)` with `est = LogisticRegression(max_iter=3).set_fit_request(sample_weight=True).set_score_request(sample_weight=False).set_callbacks(monitor)`, while the same search without the monitor works. Passing `scoring=get_scorer("accuracy").set_score_request(sample_weight=True)` to `ScoringMonitor` avoids it.

## Root cause and possible fix

`sklearn/callback/_scoring_monitor.py:167-169` forwards the hook's `metadata` as keyword arguments to a `_MultimetricScorer` built in `__init__` (`sklearn/callback/_scoring_monitor.py:110-115`), whose `__call__` routes them with `process_routing` when routing is enabled (`sklearn/metrics/_scorer.py:144-145`). Nothing on the `ScoringMonitor` side sets a request for the metadata the estimators pass to their hooks.

Options: route the hook metadata only when the scorers request it (e.g. filter `metadata` with the scorer's routing before calling it, so unrequested `sample_weight` is dropped instead of raising), or settle it as part of #34068 / #34137 where callbacks become routers. In both cases the error message should mention `ScoringMonitor`.

## Versions

```
sklearn: 1.10.dev0, commit 289c4d5cfa (upstream/main)
python: 3.13.14, numpy: 2.5.3, scipy: 1.18.1, platform: linux
```
