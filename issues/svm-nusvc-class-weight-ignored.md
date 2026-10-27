# NuSVC silently ignores `class_weight`

```json
{
    "estimators": ["NuSVC"],
    "existing_issue": "https://github.com/scikit-learn/scikit-learn/issues/30332",
    "reproduced": true,
    "priority": 2,
    "difficulty": 1,
    "decision": 1,
    "decision_tags": ["output-change"]
}
```

## Classification

- **Priority (2)**: a documented parameter, including `class_weight="balanced"` for imbalanced problems, has no effect at all, silently. Maintainers know it (the common tests xfail `check_class_weight_classifiers` for `NuSVC` with "class_weight is ignored."), but users only see the docstring that promises it works.
- **Difficulty (1)**: in `BaseSVC.fit`, for `NuSVC` multiply `sample_weight` by `class_weight_[y]` before calling libsvm (nu-SVC already supports per-sample weights), or raise/warn when `class_weight` is not None.
- **Decision (1)**: `output-change`: models fitted with `class_weight` change. The alternative (rejecting `class_weight` for `NuSVC`) would need a deprecation.

## Upstream search

Queries: "NuSVC class_weight".

- #30332 (open, "NuSVC argument `class_weight` is not used"): this exact bug. @ogrisel wrote that the maintainers decided to go the `class_weight` -> `sample_weight` conversion route, with a dedicated test, even though SVM `sample_weight` fails the equivalence checks.
- #30366 (open PR, "ENH: Add class_weight support to NuSVC via sample_weight"): implements that conversion; not merged at the commit below.
- The user guide (`doc/modules/svm.rst`) says "SVC (but not NuSVC) implements the parameter class_weight", but the `NuSVC` docstring still documents it as working.

## Describe the bug and its user-facing impact

`NuSVC` accepts `class_weight` (dict or `"balanced"`), documents it as "Set the parameter C of class i to class_weight[i]*C for SVC", and computes `class_weight_`, but the fitted model is exactly the same as with `class_weight=None`. A user who sets `class_weight="balanced"` on an imbalanced problem gets the unweighted model without any warning.

libsvm only uses the class weights to scale `C` per class (`weighted_C` in `svm_train`), and `svm_train_one` does not pass `Cp`/`Cn` to `solve_nu_svc`, since nu-SVC has no `C`. scikit-learn's libsvm does use per-sample weights in `solve_nu_svc` (`C[i] = prob->W[i]`), and `sample_weight` works for `NuSVC`.

## Steps/Code to Reproduce

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.svm import NuSVC, SVC

X, y = make_classification(n_samples=200, weights=[0.9, 0.1], flip_y=0.1, random_state=0)
for clf in [SVC(), NuSVC(nu=0.1)]:
    for cw in [None, {0: 1, 1: 100}, "balanced"]:
        clf.set_params(class_weight=cw).fit(X, y)
        print(f"{type(clf).__name__:5s} class_weight={cw!s:14s} "
              f"n_predicted_class_1={(clf.predict(X) == 1).sum()}")

# Folding the class weights into sample_weight does change the NuSVC model:
sw = np.where(y == 1, 100.0, 1.0)
print("NuSVC sample_weight=100 for class 1:",
      (NuSVC(nu=0.1).fit(X, y, sample_weight=sw).predict(X) == 1).sum())
```

## Expected Results

As with `SVC`, up-weighting class 1 should increase the number of samples predicted as class 1, as the docstring says.

## Actual Results

```
SVC   class_weight=None           n_predicted_class_1=10
SVC   class_weight={0: 1, 1: 100} n_predicted_class_1=32
SVC   class_weight=balanced       n_predicted_class_1=32
NuSVC class_weight=None           n_predicted_class_1=30
NuSVC class_weight={0: 1, 1: 100} n_predicted_class_1=30
NuSVC class_weight=balanced       n_predicted_class_1=30
NuSVC sample_weight=100 for class 1: 70
```

## Root cause and possible fix

`sklearn/svm/src/libsvm/svm.cpp:1866-1868`: the `NU_SVC` case of `svm_train_one` ignores `Cp` and `Cn`, which is where `class_weight_` ends up (`weighted_C`, `svm.cpp:2460-2470`). The Python side (`BaseSVC._validate_targets`, `sklearn/svm/_base.py:784`) computes `class_weight_` and passes it to libsvm, but nothing uses it for nu-SVC.

Possible fix, in `BaseLibSVM.fit` for `_impl == "nu_svc"`: `sample_weight = sample_weight * class_weight_[y]` (with `sample_weight` defaulting to ones) and pass an empty `class_weight` to libsvm. For `C-SVC` libsvm uses `C * class_weight[y_i] * W_i` per sample, so this gives `NuSVC` the same semantics. The other option is to document that `class_weight` is ignored and deprecate it for `NuSVC`. Folding into `sample_weight` can also make a given `nu` infeasible where it was feasible before, since the feasibility check uses the weights.

## Versions

```
sklearn: 1.10.dev0, commit 289c4d5cfa (upstream/main; ran on f570497251, no diff in the files involved)
python: 3.13.14, numpy: 2.5.3, scipy: 1.18.1, platform: linux
```
