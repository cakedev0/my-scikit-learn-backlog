# `ScoringMonitor` on `LogisticRegression` scores each lbfgs iteration against the label-encoded `y`

```json
{
    "estimators": ["LogisticRegression", "ScoringMonitor"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 2,
    "difficulty": 1,
    "decision": 0,
    "decision_tags": []
}
```

## Classification

- **Priority (2)**: silent wrong scores for every per-iteration row as soon as the class labels are not exactly `0..n_classes-1` (e.g. `{1, 2}`, `{-1, 1}`), and a crash for string labels. Only users of the new callbacks API are affected, but `LogisticRegression` is the main estimator showcased with `ScoringMonitor`.
- **Difficulty (1)**: pass the original `y` (and the user's `sample_weight`) to `_LbfgsCallbackBridge` instead of the internal ones.
- **Decision (0)**: no trade-off, the current values are just wrong.

## Upstream search

`gh issue list --search "ScoringMonitor"`, `--search "callback LogisticRegression"`, `--label Callbacks` and `gh pr list --search "ScoringMonitor"` (titles and bodies only, not comments). Related but different: #34744 (two `ProgressBar` on one estimator), #34734 (listener cache), #35104 (pickling cost), #34068 (callbacks x metadata routing). The callback support for `LogisticRegression` comes from #33847 (merged); nothing mentions the encoded labels.

## Describe the bug and its user-facing impact

With `ScoringMonitor` set on `LogisticRegression` (lbfgs), the scores logged for the `lbfgs-iter` tasks are computed against the label-encoded, float `y` (values `0.0, 1.0, ...`), while the fitted estimator passed to the hook predicts the original labels (`classes_`). The root `fit` task gets the original `y`, so its row is right.

- Labels `{0, 1}` (what `make_classification` gives, and what the tests use): correct by luck.
- Any other numeric labels (`{1, 2}`, `{-1, 1}`, `{0, 2}`, ...): the per-iteration scores are silently wrong and look like a model that does not learn (accuracy 0.01 at the last iteration while the final model has 0.95).
- String labels: `ValueError: Mix of label input types (string and number)` raised from the hook, which aborts `fit`.

The `metadata` passed to the per-iteration hooks also has the `sample_weight` already multiplied by `class_weight`, while the root task gets the user's `sample_weight`, so the per-iteration scores are weighted differently from the final one when `class_weight` is set.

## Steps/Code to Reproduce

```python
import warnings

from sklearn.callback import ScoringMonitor
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")

X, y = make_classification(n_samples=200, random_state=0)
y = y + 1  # labels {1, 2} instead of {0, 1}

monitor = ScoringMonitor(scoring="accuracy")
clf = LogisticRegression(max_iter=5).set_callbacks(monitor).fit(X, y)

for row in monitor.get_logs().data:
    print(row["task_name"], row["task_id_path"], row.get("accuracy"))
print("clf.score(X, y):", clf.score(X, y))
```

## Expected Results

The per-iteration accuracies are those of the model at each iteration on the training data, ending close to the final `clf.score(X, y)` of 0.95 (the `ScoringMonitor` docstring: "The specified scorer is called on the training data at each iterative step of the estimator").

## Actual Results

```
fit (0,) 0.95
lbfgs-iter (0, 0) 0.015
lbfgs-iter (0, 1) 0.01
lbfgs-iter (0, 2) 0.01
lbfgs-iter (0, 3) 0.01
lbfgs-iter (0, 4) 0.01
clf.score(X, y): 0.95
```

With `y = np.array(["neg", "pos"])[y]`, `fit` raises:

```
ValueError: Mix of label input types (string and number); Got [0. 1.] and ['neg' 'pos'].
```

## Root cause and possible fix

`LogisticRegression.fit` passes `y_encoded` (`LabelEncoder` output cast to `X.dtype`) to `_logistic_regression_path` (`sklearn/linear_model/_logistic.py:1831`), which hands its local `y` and its local, class-weighted `sample_weight` to `_LbfgsCallbackBridge` (`sklearn/linear_model/_logistic.py:631-656`). The bridge forwards them as `X`/`y`/`metadata` to `call_on_fit_task_begin`/`call_on_fit_task_end` (`sklearn/linear_model/_logistic.py:211-229`), while `fitted_estimator` is a copy of the estimator, whose `predict` maps back to `classes_`.

Fix: build the bridge's `y` and `metadata` from what the user passed to `fit` (the same values as the root task at `sklearn/linear_model/_logistic.py:1786-1791`), e.g. by passing them to `_logistic_regression_path` next to `callback_ctx`, and add a test with labels other than `{0, 1}`.

## Versions

```
sklearn: 1.10.dev0, commit 289c4d5cfa (upstream/main)
python: 3.13.14, numpy: 2.5.3, scipy: 1.18.1, platform: linux
```
