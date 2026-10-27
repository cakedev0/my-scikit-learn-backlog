# SVC and LinearSVC predict the wrong class when a class only has zero sample weights

```json
{
    "estimators": ["SVC", "LinearSVC"],
    "existing_issue": "https://github.com/scikit-learn/scikit-learn/issues/25380",
    "reproduced": true,
    "priority": 2,
    "difficulty": 2,
    "decision": 1,
    "decision_tags": ["unclear-spec", "output-change"]
}
```

## Classification

- **Priority (2)**: silent wrong predictions (labels of a class the model never saw) on widely used estimators. It needs a whole class with zero `sample_weight`, which happens when weights come from a mask or a cross-validation trick, but is not the common case.
- **Difficulty (2)**: the fix is in the Python wrappers (drop the zero-weight classes before calling libsvm/liblinear and map the labels back, or raise), but it touches two code paths and the binary/multiclass layout of `coef_`, `dual_coef_`, `intercept_`.
- **Decision (1)**: `unclear-spec`: one must decide whether such classes stay in `classes_` (with `-inf` decision values / never predicted) or whether fitting raises. `output-change` only for this currently broken case.

## Upstream search

Queries: "specified in weight is not found", "SVC sample_weight zero class", "SVC sample_weight class removed predict wrong".

- #25380 (open, "SVC and OneClassSVM fails to fit or have wrong fitted attributes with null sample weights"): same root cause. It reports the wrong `support_`, `n_support_`, `dual_coef_` and the sparse `ValueError`; a comment shows the `LinearSVC` `coef_` shape. @glemaitre suggests fixing it in `svm.cpp` (keep a mapping of removed samples and classes). This file adds the user-visible consequences not mentioned there: `predict` silently returns the masked class, and `decision_function` crashes. No linked PR found.
- #5150 (closed 2026): `SVC(probability=True)` with a zero-weight class; closed after the deprecation of `probability`, different symptom.
- #9494 / #9674 (closed): non-positive weights making the fit fail, fixed by the remove-zero-weight logic that causes this bug.

## Describe the bug and its user-facing impact

libsvm and liblinear drop the samples with a non-positive weight before training (`remove_zero_weight` in `svm.cpp` and `linear.cpp`). When all samples of one class have weight 0, the class disappears from the trained model, but the Python wrapper still sets `classes_` from the full `y`. The internal labels `0..k-1` of the reduced model are then mapped onto the wrong entries of `classes_`.

With 3 classes and class 1 masked out:

- `SVC.predict` and `LinearSVC.predict` return class 1 (the class with no weight) instead of class 2, silently.
- `SVC.decision_function` with the default `decision_function_shape="ovr"` crashes with an `IndexError`; the sparse `SVC` path raises `ValueError: indices and data should have the same size` at fit.
- `LinearSVC.coef_` has shape `(1, n_features)` while `classes_` has 3 entries.

The only hint is a line printed by the C code to stderr (`warning: class label 1 specified in weight is not found`).

The existing tests (`test_negative_weights_svc_leave_two_labels`, `test_negative_weights_svc_leave_just_one_label`) only cover the binary case, where libsvm raises a clear error.

## Steps/Code to Reproduce

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.svm import SVC, LinearSVC

X, y = make_classification(n_samples=60, n_classes=3, n_informative=4, random_state=0)
sample_weight = (y != 1).astype(float)  # class 1 only has zero weights

for Est in [SVC, LinearSVC]:
    ref = Est().fit(X[y != 1], y[y != 1])
    clf = Est().fit(X, y, sample_weight=sample_weight)
    print(f"{Est.__name__} predict, class 1 samples dropped:", ref.predict(X[:8]))
    print(f"{Est.__name__} predict, class 1 weights set to 0:", clf.predict(X[:8]))

SVC().fit(X, y, sample_weight=sample_weight).decision_function(X)
```

## Expected Results

A zero weight should be equivalent to dropping the sample, so the predictions should match the reference fit (only classes 0 and 2). At the very least, class 1 should never be predicted and `decision_function` should not crash. Raising an informative error would also be acceptable.

## Actual Results

```
warning: class label 1 specified in weight is not found
WARNING: class label 1 specified in weight is not found
warning: class label 1 specified in weight is not found
SVC predict, class 1 samples dropped: [2 2 2 0 0 2 0 2]
SVC predict, class 1 weights set to 0: [1 1 1 0 0 1 0 1]
LinearSVC predict, class 1 samples dropped: [2 2 2 0 0 2 0 2]
LinearSVC predict, class 1 weights set to 0: [1 1 1 0 0 1 0 1]
Traceback (most recent call last):
  File "svm_zero_class_final.py", line 14, in <module>
    SVC().fit(X, y, sample_weight=sample_weight).decision_function(X)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^
  File "/home/arthur/scikit-learn/sklearn/svm/_base.py", line 824, in decision_function
    return _ovr_decision_function(dec < 0, -dec, len(self.classes_))
  File "/home/arthur/scikit-learn/sklearn/utils/multiclass.py", line 566, in _ovr_decision_function
    sum_of_confidences[:, i] -= confidences[:, k]
                                ~~~~~~~~~~~^^^^^^
IndexError: index 1 is out of bounds for axis 1 with size 1
```

## Root cause and possible fix

- libsvm: `BaseSVC._validate_targets` (`sklearn/svm/_base.py:783`) encodes `y` over all classes. `svm_train` drops the zero-weight samples, so the model has fewer classes, and predictions come back as indices into the reduced label set: `set_model` in `sklearn/svm/src/libsvm/libsvm_helper.c:163` sets `model->label[i] = i`. `BaseSVC.predict` then does `self.classes_.take(y)`, which maps index 1 (class 2 of the reduced model) to `classes_[1]`.
- liblinear: `_fit_liblinear` (`sklearn/svm/_base.py:1228`) also encodes over all classes, and `train` in `linear.cpp` drops the zero-weight samples, so for 3 classes liblinear trains a single binary model. `LinearClassifierMixin.predict` maps it to `classes_[0]`/`classes_[1]`. (`LogisticRegression(solver="liblinear")` is not affected: it refuses 3+ classes.)

A fix in Python: compute the classes from the samples with positive weight, call the solver on those, and either keep the other classes in `classes_` with `-inf` decision values (and zero rows in `coef_`), or raise a clear `ValueError`. Raising is the simpler option; keeping the classes matches better what other classifiers do (e.g. `LogisticRegression` with lbfgs keeps all classes). `NuSVC` is not silently wrong here: in this example it raises `specified nu is infeasible` for every `nu` I tried (0.05, 0.1, 0.3), which is misleading but loud.

## Versions

```
sklearn: 1.10.dev0, commit 289c4d5cfa (upstream/main; ran on f570497251, no diff in the files involved)
python: 3.13.14, numpy: 2.5.3, scipy: 1.18.1, platform: linux
```
