# Multilabel scorers crash when a label has a single class at fit time

```json
{
    "estimators": ["get_scorer", "make_scorer", "cross_validate", "GridSearchCV", "RandomForestClassifier", "KNeighborsClassifier"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 2,
    "difficulty": 2,
    "decision": 0,
    "decision_tags": []
}
```

## Classification

- **Priority (2)**: probability-based scorers (`"average_precision"`, `"roc_auc"`, `"neg_log_loss"`, ...) crash with a cryptic numpy error on multilabel targets as soon as one label is constant in the training data, which is common for rare labels in a CV training fold. No silent wrong result, but no easy workaround besides writing a custom scorer.
- **Difficulty (2)**: `_get_response_values` and `_process_predict_proba` need to handle a list of per-output `classes_` with different lengths, and pick the column of the positive class per output.
- **Decision (0)**: the expected value is clear: `P(label=1)` is 0 (or 1) for a label that only had 0 (or 1) at fit time.

## Upstream search

Queries: "multilabel scorer label not present in training predict_proba", "_get_response_values multilabel inhomogeneous", "average_precision scorer multilabel RandomForest constant label". The multilabel branch comes from #27002 (merged, 2023, "FIX accept multilabel-indicator in _get_response_values") and #26840 (merged, 2023, new `response_method` in `make_scorer`); neither covers outputs with a single class.

## Describe the bug and its user-facing impact

For multilabel targets, `RandomForestClassifier`, `ExtraTreesClassifier`, `DecisionTreeClassifier` and `KNeighborsClassifier` return `predict_proba` as a list with one `(n_samples, n_classes_k)` array per label, and `classes_` as a list of arrays. When a label only has one value in the training data (all 0 for a rare label), its `classes_` is `[0]` and its probability array has one column.

`_get_response_values`, used by all scorers, computes `type_of_target(estimator.classes_)`. With `classes_ == [array([0, 1]), array([0, 1]), array([0])]`, the ragged input gives `"unknown"`, so the list of arrays is returned unprocessed and the metric fails in `check_array` with `setting an array element with a sequence`.

The user sees this in `cross_validate(..., scoring="average_precision")` or a grid search on multilabel data with a rare label: some folds error out (or get `error_score=nan`) with a message unrelated to the cause.

Note that even with the type check fixed, the multilabel branch takes `p[:, -1]`, which is `P(label=0)` for a label whose only class is 0: the column must be chosen from `classes_`, not by position.

## Steps/Code to Reproduce

```python
import numpy as np
from sklearn.datasets import make_multilabel_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, get_scorer

X, Y = make_multilabel_classification(n_samples=100, n_classes=3, random_state=0)
Y_train = Y[:50].copy()
Y_train[:, 2] = 0  # the third label is never seen at fit time (e.g. a rare label)

clf = RandomForestClassifier(random_state=0).fit(X[:50], Y_train)
print([c.tolist() for c in clf.classes_])

# Manual computation: P(label=1) is 0 for the label never seen during fit.
proba = clf.predict_proba(X[50:])
y_score = np.column_stack([
    p[:, list(c).index(1)] if 1 in c else np.zeros(len(p))
    for p, c in zip(proba, clf.classes_)
])
print("manual:", average_precision_score(Y[50:], y_score))
print("scorer:", get_scorer("average_precision")(clf, X[50:], Y[50:]))
```

## Expected Results

The scorer gives the same value as the manual computation (`0.8064886814941264`).

## Actual Results

```
[[0, 1], [0, 1], [0]]
manual: 0.8064886814941264
Traceback (most recent call last):
  File "repro_ml.py", line 20, in <module>
    print("scorer:", get_scorer("average_precision")(clf, X[50:], Y[50:]))
                     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^
  File "/home/arthur/scikit-learn/sklearn/utils/validation.py", line 77, in inner_f
    return f(*args, **kwargs)
  ...
  File "/home/arthur/scikit-learn/sklearn/metrics/_scorer.py", line 469, in _score
    return self._sign * self._score_func(y_true, y_pred, **scoring_kwargs)
                        ~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/arthur/scikit-learn/sklearn/utils/_param_validation.py", line 218, in wrapper
    return func(*args, **kwargs)
  File "/home/arthur/scikit-learn/sklearn/metrics/_ranking.py", line 232, in average_precision_score
    y_score = check_array(y_score, ensure_2d=False)
  File "/home/arthur/scikit-learn/sklearn/utils/validation.py", line 1032, in check_array
    array = _asarray_with_order(array, order=order, dtype=dtype, xp=xp)
  File "/home/arthur/scikit-learn/sklearn/utils/_array_api.py", line 977, in _asarray_with_order
    array = numpy.asarray(array, order=order, dtype=dtype)
ValueError: setting an array element with a sequence. The requested array has an inhomogeneous shape after 2 dimensions. The detected shape was (3, 50) + inhomogeneous part.
```

## Root cause and possible fix

`sklearn/utils/_response.py:206` (upstream/main 289c4d5cfa): `type_of_target(classes)` is "unknown" when `classes_` is a ragged list, so `_process_predict_proba` does not stack the outputs. And `sklearn/utils/_response.py:70` takes the last column of each output, which is not the positive class when an output only saw class 0.

Possible fix: when `classes_` is a list (multi-output estimator), detect the multilabel case per output (each `classes_[k]` a subset of `{0, 1}`), and build each column as `p[:, classes_[k] == 1]` if 1 was seen, else zeros.

## Versions

```
sklearn: 1.10.dev0, commit 289c4d5cfa (upstream/main; local build f570497251, no diff on sklearn/utils/_response.py)
python: 3.13, numpy: 2.5.3, scipy: 1.18.1, platform: linux
```
