# OneVsRestClassifier.partial_fit fails on multilabel indicator targets although the docstring says they are supported

```json
{
    "estimators": ["OneVsRestClassifier"],
    "existing_issue": "https://github.com/scikit-learn/scikit-learn/issues/8381",
    "reproduced": true,
    "priority": 2,
    "difficulty": 2,
    "decision": 0,
    "decision_tags": []
}
```

## Classification

- **Priority (2)**: loud error, but on a documented use case (out-of-core multilabel text classification) with no workaround short of writing the OvR loop by hand. Open since 2017 with many user reports.
- **Difficulty (2)**: `partial_fit` must detect the multilabel case on the first call and fit the `LabelBinarizer` on an indicator matrix, and check consistency on later calls. Open PR #35080 does this in ~80 lines.
- **Decision (0)**: the fix only makes a currently failing input work.

## Upstream search

- `gh issue list --search "OneVsRestClassifier partial_fit multilabel"`: #8381 (open, Bug, help wanted) is this bug, reported in 2017 with a StackOverflow link.
- `gh pr list --search "OneVsRestClassifier partial_fit multilabel"`: #35080 (open, "FIX OneVsRestClassifier.partial_fit with multilabel indicator y", fixes #8381).
- Filed here for completeness of the backlog since the issue is still open on main.

## Describe the bug and its user-facing impact

The `partial_fit` docstring documents `y` as "{array-like, sparse matrix} of shape (n_samples,) or (n_samples, n_classes). Multi-class targets. An indicator matrix turns on multilabel classification." But on the first call, the `LabelBinarizer` is fitted on the 1-D `classes` array, so it is never in multilabel mode, and transforming the indicator matrix fails. `fit` accepts the same `Y`.

## Steps/Code to Reproduce

```python
import numpy as np
from sklearn.datasets import make_multilabel_classification
from sklearn.linear_model import SGDClassifier
from sklearn.multiclass import OneVsRestClassifier

X, Y = make_multilabel_classification(n_samples=100, n_classes=3, random_state=0)
OneVsRestClassifier(SGDClassifier()).fit(X, Y)  # works
OneVsRestClassifier(SGDClassifier()).partial_fit(X, Y, classes=np.arange(3))
```

## Expected Results

`partial_fit` accepts the indicator matrix, as documented and as `fit` does.

## Actual Results

```
    return fit_method(estimator, *args, **kwargs)
  File "/home/arthur/scikit-learn/sklearn/multiclass.py", line 458, in partial_fit
    Y = self.label_binarizer_.transform(y)
  File "/home/arthur/scikit-learn/sklearn/preprocessing/_label.py", line 386, in transform
    raise ValueError("The object was not fitted with multilabel input.")
ValueError: The object was not fitted with multilabel input.
```

## Root cause and possible fix

`sklearn/multiclass.py:449`: `self.label_binarizer_.fit(self.classes_)` fits on the 1-D class list, so `y_type_` is `"multiclass"`; `sklearn/multiclass.py:458` then calls `transform` on an indicator matrix. The `np.setdiff1d(y, self.classes_)` check at line 451 is also meaningless for an indicator matrix. On the first call with a 2-D `y`, the binarizer should be fitted on an indicator matrix with `n_classes` columns (see PR #35080).

## Versions

```
sklearn: 1.10.dev0, commit 289c4d5cfa (upstream/main; reproduced on local checkout f570497251, no diff in sklearn/multiclass.py)
python: 3.13, numpy: 2.5.3, scipy: 1.18.1, platform: linux
```
