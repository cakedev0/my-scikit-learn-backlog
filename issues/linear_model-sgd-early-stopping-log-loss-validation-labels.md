# SGDClassifier(loss="log_loss", early_stopping=True) monitors a wrong validation accuracy (negatives are always counted as errors)

```json
{
    "estimators": ["SGDClassifier"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 2,
    "difficulty": 1,
    "decision": 1,
    "decision_tags": ["output-change"]
}
```

## Classification

- **Priority (2)**: silent, on a common combination (`log_loss` is the loss to use for probabilities, `early_stopping=True` is a documented option). The fit still ends somewhere reasonable, since the monitored score is roughly "fraction of validation samples that are correctly predicted positives", which still moves with the model quality, but the stopping point is not based on the validation accuracy the docs describe.
- **Difficulty (1)**: the validation callback is built with `classes=[-1, 1]` while the targets are encoded in `{0, 1}` for `log_loss`. Pass `classes=[0, 1]` when `label_encode` is true.
- **Decision (1)**: `output-change`: `n_iter_` and the fitted coefficients change for `log_loss` with `early_stopping=True` and the same `random_state`.

## Upstream search

Queries: "SGDClassifier early_stopping", "SGDClassifier early_stopping log_loss", "sgd early stopping log_loss", "early_stopping validation score log_loss". Related but different: #18748 (open, custom validation set), #22208 (closed, validation mask construction). Nothing about the label encoding.

The bug was introduced in 1.6.0 by #28029 ("ENH replace Cython loss functions in SGD part 2"), which started encoding the binary targets in `{0, 1}` for `CyHalfBinomialLoss` (`label_encode`), but kept `classes = np.array([-1, 1])` for the validation score callback.

## Describe the bug and its user-facing impact

With `early_stopping=True`, `SGDClassifier` stops when the accuracy on the validation split stops improving (`n_iter_no_change`, `tol`). For `loss="log_loss"`, `fit_binary` encodes the targets of each binary problem as `y_i` in `{0, 1}`, but the validation callback predicts with `classes_ = [-1, 1]`. Every validation sample of the negative class is predicted as `-1` and compared to `0`, so it always counts as an error. The monitored score is the fraction of validation samples that are correctly predicted positives, about half the true accuracy on balanced data, and its variations are driven by the positive class only.

All the other losses use `{-1, 1}` targets and are not affected. For multiclass problems, each one-vs-rest binary problem is affected the same way.

In practice this changes when training stops: on a 1000-sample problem, `n_iter_` differs from the one obtained with a correct validation score for 4 out of 5 seeds (9 vs 8, 12 vs 7, 7 vs 9, 6 vs 10). With imbalanced classes, or when the positive class is the minority, the monitored score is dominated by the minority class.

## Steps/Code to Reproduce

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import SGDClassifier

# Record what the early-stopping callback sees. The callback calls `score` on a clone
# of the estimator, so overriding `score` is enough to observe it.
calls = []


class RecordingSGD(SGDClassifier):
    def score(self, X, y, sample_weight=None):
        score = super().score(X, y, sample_weight)
        calls.append((np.unique(y), self.classes_, score))
        return score


X, y = make_classification(n_samples=1000, random_state=0)
for loss in ["hinge", "log_loss"]:
    calls.clear()
    RecordingSGD(loss=loss, early_stopping=True, random_state=0).fit(X, y)
    y_val, classes, _ = calls[0]
    print(f"{loss}: validation y={y_val}, classes_ used by predict={classes}")
    print("  validation accuracies:", np.round([s for *_, s in calls], 2))
```

## Expected Results

The validation labels and the `classes_` used to predict match, and the validation accuracies with `log_loss` are in the same range as with `hinge` (~0.9 on this dataset).

## Actual Results

```
hinge: validation y=[-1.  1.], classes_ used by predict=[-1.  1.]
  validation accuracies: [0.88 0.93 0.9  0.92 0.93 0.93 0.92]
log_loss: validation y=[0. 1.], classes_ used by predict=[-1.  1.]
  validation accuracies: [0.45 0.46 0.45 0.41 0.45 0.43 0.44]
```

## Root cause and possible fix

`sklearn/linear_model/_stochastic_gradient.py` (commit 289c4d5cfa):

- line 450: `label_encode = isinstance(est._loss_function_, CyHalfBinomialLoss)`, and `_prepare_fit_binary` sets the negative targets to `0.0` (line 358);
- line 466: `classes = np.array([-1, 1], dtype=y_i.dtype)` is passed to `_make_validation_score_cb` regardless of `label_encode`;
- line 84: `_ValidationScoreCallback` calls `est.score(X_val, y_val)`, where `predict` returns `classes_[decision > 0]`, i.e. `-1` or `1`.

Fix:

```python
classes = np.array([0, 1] if label_encode else [-1, 1], dtype=y_i.dtype)
```

A regression test can compare, for `log_loss` with `early_stopping=True`, the recorded validation score to `accuracy_score` computed on the same validation split.

## Versions

```
sklearn: 1.10.dev0, commit 289c4d5cfa (upstream/main, pure-Python files overlaid on the local build: no compiled file differs in sklearn/linear_model)
python: 3.13.14, numpy: 2.5.3, scipy: 1.18.1, platform: linux
```
