# Callable `oob_score` in forest classifiers receives encoded labels instead of the user's labels

```json
{
    "estimators": ["RandomForestClassifier", "ExtraTreesClassifier"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 2,
    "difficulty": 1,
    "decision": 1,
    "decision_tags": ["output-change"]
}
```

## Classification

- **Priority (2)**: silent wrong value for any label-dependent metric (`pos_label`, `labels`, per-class averages) when classes are not exactly `0..n_classes-1`, and a confusing error with string labels. Only hits users of the callable `oob_score` option, with a metric that depends on label values.
- **Difficulty (1)**: map both arrays back through `classes_` before calling the scorer (a few lines in `ForestClassifier._set_oob_score_and_attributes`).
- **Decision (1)**: `output-change`: users who adapted their callable to the encoded labels (e.g. `pos_label=1` meaning "second class") would get a different score after the fix.

## Upstream search

GraphQL search on scikit-learn/scikit-learn: "oob_score callable", "oob_score encoded labels", "oob_score pos_label", "oob_score_ y encoded RandomForestClassifier string labels".

- #25177 (merged): added the callable `oob_score`. Its discussion does not mention label encoding, and the test only uses `partial(f1_score, average="micro")`, which is invariant to label values.
- #8933 (closed): "BaggingClassifier.oob_score_ should not change with class label", different estimator and symptom.

No match for this bug.

## Describe the bug and its user-facing impact

With `oob_score=<callable>`, the docstring says the callable has signature `metric(y_true, y_pred)`. `ForestClassifier` calls it with `y` after `np.unique(..., return_inverse=True)` encoding (as float, shape `(n_samples, 1)`) and with `argmax` indices as predictions. So the callable never sees the user's labels:

- labels `{1, 2}` with `f1_score(..., pos_label=1)` silently computes the F1 of class `2` (encoded as `1`);
- string labels make any `pos_label="..."` or `labels=[...]` raise a confusing `ValueError` listing `[0. 1.]` as the valid labels.

## Steps/Code to Reproduce

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

X, y = make_classification(n_samples=500, weights=[0.8], random_state=0)
y = y + 1  # labels {1, 2}, class 1 is the majority class


def f1_class_1(y_true, y_pred):
    print("labels seen by the callable:", np.unique(y_true), np.unique(y_pred))
    return f1_score(y_true, y_pred, pos_label=1)


rf = RandomForestClassifier(oob_score=f1_class_1, random_state=0).fit(X, y)
y_oob = rf.classes_[rf.oob_decision_function_.argmax(axis=1)]
print("oob_score_                   :", rf.oob_score_)
print("f1_score(y, y_oob, pos_label=1):", f1_score(y, y_oob, pos_label=1))

# String labels: the callable cannot refer to the user's labels at all.
y_str = np.where(y == 1, "neg", "pos")
RandomForestClassifier(
    oob_score=lambda t, p: f1_score(t, p, pos_label="pos"), random_state=0
).fit(X, y_str)
```

## Expected Results

The callable receives `y` and OOB predictions in the original label space (as `predict` and `score` do), so `oob_score_` equals `f1_score(y, y_oob, pos_label=1)` = 0.956, and the string-label case works.

## Actual Results

```
labels seen by the callable: [0. 1.] [0 1]
oob_score_                   : 0.797752808988764
f1_score(y, y_oob, pos_label=1): 0.9562043795620438
Traceback (most recent call last):
  ...
ValueError: pos_label=pos is not a valid label. It should be one of [0. 1.]
```

## Root cause and possible fix

`sklearn/ensemble/_forest.py:894` (upstream/main) in `ForestClassifier._set_oob_score_and_attributes`:

```python
self.oob_score_ = scoring_function(
    y, np.argmax(self.oob_decision_function_, axis=1)
)
```

`y` here is the encoded target built in `_validate_y_class_weight`. For the default `accuracy_score` this does not matter, but for a user callable it does. Fix: decode both sides with `self.classes_` (per output for multi-output, as `predict` does) before calling the scorer, and ravel `y` in the single-output case. A test with a label-dependent metric (e.g. `pos_label`) on non-`0..K-1` labels would cover it.

## Versions

```
sklearn: 1.10.dev0, commit f570497251 (code identical to upstream/main 289c4d5cfa for this function)
python: 3.13, numpy: 2.5.3, scipy: 1.18.1, platform: linux
```
