# RadiusNeighborsClassifier rejects a string outlier_label longer than the class names

```json
{
    "estimators": ["RadiusNeighborsClassifier"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 1,
    "difficulty": 1,
    "decision": 0,
    "decision_tags": []
}
```

## Classification

- **Priority (1)**: loud but misleading error for a natural use (`outlier_label="unknown"` with string classes); casting `y` to object dtype, or picking a short label, works around it.
- **Difficulty (1)**: compare dtype kinds instead of exact dtypes in the check.
- **Decision (0)**: no trade-off; the check is meant to catch a str label for numeric classes (and vice versa), which a kind check still does.

## Upstream search

- `outlier_label dtype`, `RadiusNeighborsClassifier outlier_label`: #9597 (merged, added `outlier_label` and the dtype check), #6902, #1602 (closed, older outlier label problems), #25854 (closed, request for a default label in `KNeighborsClassifier`). None about string lengths.

## Describe the bug and its user-facing impact

`RadiusNeighborsClassifier.fit` checks that `outlier_label` has "the same type as y" with `np.append(classes, label).dtype != classes.dtype`. For string classes stored as a numpy unicode array (e.g. `y` given as a list of str), the dtype includes the maximum string length (`<U3` for `["cat", "dog"]`), so any `outlier_label` longer than the longest class name changes the dtype (`<U7` for `"unknown"`) and fit raises "The dtype of outlier_label unknown is inconsistent with classes". A label of the same length (`"owl"`) is accepted, and the docstring explicitly allows a label outside the training classes (with a warning). With `y` as an object array, the same label works.

## Steps/Code to Reproduce

```python
import numpy as np
from sklearn.neighbors import RadiusNeighborsClassifier

X = np.array([[0.0], [1.0], [2.0]])
y = ["cat", "dog", "dog"]
X_test = [[0.1], [100.0]]  # the second point has no neighbor

for outlier_label in ["owl", "unknown"]:
    clf = RadiusNeighborsClassifier(radius=0.5, outlier_label=outlier_label)
    try:
        print(outlier_label, clf.fit(X, y).predict(X_test))
    except TypeError as e:
        print(outlier_label, "TypeError:", e)

clf = RadiusNeighborsClassifier(radius=0.5, outlier_label="unknown")
print(clf.fit(X, np.array(y, dtype=object)).predict(X_test))
```

## Expected Results

`['cat' 'unknown']` (with the documented "not in training classes" warning) for `"unknown"` too, whatever the length of the label and whether `y` is a list, a unicode array or an object array.

## Actual Results

```
/home/arthur/scikit-learn/sklearn/neighbors/_classification.py:864: UserWarning: Outlier label owl is not in training classes. All class probabilities of outliers will be assigned with 0.
  warnings.warn(
/home/arthur/scikit-learn/sklearn/neighbors/_classification.py:864: UserWarning: Outlier label unknown is not in training classes. All class probabilities of outliers will be assigned with 0.
  warnings.warn(
owl ['cat' 'owl']
unknown TypeError: The dtype of outlier_label unknown is inconsistent with classes ['cat' 'dog'] in y.
['cat' 'unknown']
```

## Root cause and possible fix

`sklearn/neighbors/_classification.py:707`: `np.append(classes, label).dtype != classes.dtype` is a length check for unicode dtypes. Compare kinds instead (e.g. `np.asarray(label).dtype.kind` against `classes.dtype.kind`, treating `U`/`S`/`O` strings as compatible and int/float as numeric). `predict` must then also build `y_pred` with a dtype that can hold the label (`np.result_type(classes_k, label)` instead of `classes_[0].dtype`, line 745), otherwise a long label would be truncated to the class dtype.

## Versions

```
sklearn: 1.10.dev0, commit f570497251 (no diff in sklearn/neighbors vs upstream/main 289c4d5cfa)
python: 3.13, numpy: 2.5.3, scipy: 1.18.1, platform: linux
```
