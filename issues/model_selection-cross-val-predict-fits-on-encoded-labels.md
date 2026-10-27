# `cross_val_predict(method="predict_proba")` fits the estimator on label-encoded `y`, breaking label-dependent parameters

```json
{
    "estimators": ["cross_val_predict"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 2,
    "difficulty": 2,
    "decision": 0,
    "decision_tags": []
}
```

## Classification

- **Priority (2)**: crash on valid input in a common pattern (string labels plus a `class_weight` dict, for out-of-fold probabilities used for calibration plots or stacking). Loud error, but the message is confusing since the user never passed `[0, 1]`, and `method="predict"` works on the same estimator.
- **Difficulty (2)**: fit on the original `y` and use the label encoder only to map `estimator.classes_` to columns, for both the 1D and the multioutput (list of arrays) paths.
- **Decision (0)**: estimators would see the same labels as with `method="predict"` or a plain `fit`; the class order is the sorted one in both cases.

## Upstream search

`gh issue list --search` (GraphQL, the REST search was rate limited): "cross_val_predict class_weight predict_proba", "cross_val_predict LabelEncoder", "cross_val_predict encoded labels predict_proba", "cross_val_predict predict_proba string labels", "cross_val_predict pos_label", "cross_val_predict class_weight", "classes are not in class_weight cross_val_predict", "StackingClassifier class_weight".

- #10233 (open): SVC + `class_weight='balanced'` + LeaveOneOut. Different issue.
- #16385 (closed, no reproducer), #23899 (closed): unrelated.

No match. The encoding comes from #7889 (2016), which added it to align the columns of `predict_proba` across folds missing some classes.

## Describe the bug and its user-facing impact

For `method` in `{"predict_proba", "predict_log_proba", "decision_function"}`, `cross_val_predict` replaces `y` by `LabelEncoder().fit_transform(y)` before fitting the per-fold clones. The estimator is then trained on `0..n_classes-1` instead of the user's labels. Any parameter that refers to class labels breaks:

- `class_weight={"spam": 10, "ham": 1}` on `LogisticRegression`, `SVC`, trees, forests, HGB, `SGDClassifier`, ...: raises `ValueError: The classes, [0, 1], are not in class_weight`;
- `pos_label` of `FixedThresholdClassifier` / `TunedThresholdClassifierCV`, or a `scoring` with `pos_label` inside a `*CV` estimator: these now refer to labels that do not exist in the fold.

With `method="predict"`, `y` is not encoded and the same estimator works. A related symptom exists in `StackingClassifier`, which label-encodes `y` itself before fitting the base estimators (`StackingClassifier([("lr", LogisticRegression(class_weight={"spam": 10, "ham": 1}))]).fit(X, y)` raises the same error), but that is a separate code path in `sklearn/ensemble/_stacking.py`.

## Steps/Code to Reproduce

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict

X, y = make_classification(n_samples=100, random_state=0)
y = np.where(y == 1, "spam", "ham")
clf = LogisticRegression(class_weight={"spam": 10, "ham": 1})

print(cross_val_predict(clf, X, y, method="predict")[:3])
cross_val_predict(clf, X, y, method="predict_proba")
```

## Expected Results

Out-of-fold probabilities, with columns in the order of `np.unique(y)`, as for `clf.fit(X, y).predict_proba(X)`.

## Actual Results

```
['spam' 'spam' 'spam']
Traceback (most recent call last):
  ...
ValueError: The classes, [0, 1], are not in class_weight
```

## Root cause and possible fix

`sklearn/model_selection/_validation.py:1208-1209` (and the multioutput branch right after) overwrite `y` with its encoding, and `_fit_and_predict` fits on it. The encoding is only needed to place each fold's prediction columns in a common order (`_enforce_prediction_order` uses `estimator.classes_` as column indices).

Fix: keep fitting on the original `y`, fit the `LabelEncoder` on the full `y` only to compute `n_classes` and to translate the fold's `estimator.classes_` into column indices (`le.transform(estimator.classes_)`) in `_enforce_prediction_order`. Same per output for multioutput `y`.

## Versions

```
sklearn: 1.10.dev0, commit f570497251 (local branch; the cross_val_predict code is
identical on upstream/main 289c4d5cfa)
python: 3.13, numpy: 2.5.3, scipy: 1.18.1, platform: linux
```
