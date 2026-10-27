# `class_weight="balanced"` gives an infinite weight to a class whose samples all have zero `sample_weight`

```json
{
    "estimators": ["compute_class_weight", "LogisticRegression", "LogisticRegressionCV", "LinearSVC"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 1,
    "difficulty": 1,
    "decision": 1,
    "decision_tags": ["unclear-spec"]
}
```

## Classification

- **Priority (1)**: needs `class_weight="balanced"` together with `sample_weight` and a class whose samples all have zero weight, a rare combination; the broken fit comes with `RuntimeWarning`s and a `ConvergenceWarning` after 0 iterations, and dropping the masked samples is an easy workaround. (priority review, 2026-10-23: was 2)
- **Difficulty (1)**: a few lines in `compute_class_weight`.
- **Decision (1)**: `unclear-spec`: the weight given to an absent class does not matter for the fit (it multiplies zeros) but is returned by the public `compute_class_weight`; 0 is the natural choice, and whether `n_classes` in the formula should count it must be decided.

## Upstream search

- #34139 (closed as not planned, 2026, "Handling of classes with all-zero weights"): about classes with only zero weights in general, with `class_weight=None`. @lorentzenchr considered the current behavior consistent. The `class_weight="balanced"` case was not discussed by maintainers; three drive-by PRs adding an error or a guard in `compute_class_weight`, #32639 ("FIX Handle zero division in compute_class_weight with weighted samples", closed without review on the merits), #32941 and #34199 (auto-closed), never got a technical review and there is no issue for this case.
- New argument: with `class_weight="balanced"` the fit gives a broken constant model (`coef_ == 0`, NaN sample weights inside the solver), while the same fit with `class_weight=None` is fine. Giving a 0 (or any finite) weight to the empty class brings it back to the behavior accepted in #34139.
- Other queries: "compute_class_weight balanced sample_weight zero", "class_weight balanced sample_weight inf". The `sample_weight` support in the "balanced" mode comes from #30057 (merged, 2025, "Fix linear svc handling sample weights under class_weight='balanced'"). `issues/svm-zero-weight-class-wrong-labels.md` is about the same input in libsvm/liblinear, with a different root cause.

## Describe the bug and its user-facing impact

`compute_class_weight("balanced", ..., sample_weight=sw)` computes `sum(sw) / (n_classes * weighted_class_counts)`. When every sample of a class has zero weight, its weighted count is 0 and its class weight is `inf`.

`LogisticRegression` and `LogisticRegressionCV` then compute `sample_weight * class_weight_`, i.e. `0 * inf = nan` for those samples. lbfgs stops at iteration 0 and the estimator returns `coef_ == 0`, i.e. a constant classifier, with only a `ConvergenceWarning` and two RuntimeWarnings as hints. The same fit with `class_weight=None` works. `LinearSVC` also receives the `inf` weight through `_fit_liblinear`.

## Steps/Code to Reproduce

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_class_weight

X, y = make_classification(n_samples=300, n_classes=3, n_informative=4, random_state=0)
sample_weight = (y != 2).astype(float)  # class 2 is masked out

print(compute_class_weight("balanced", classes=np.unique(y), y=y,
                           sample_weight=sample_weight))

for class_weight in [None, "balanced"]:
    clf = LogisticRegression(class_weight=class_weight)
    clf.fit(X, y, sample_weight=sample_weight)
    print(class_weight, "accuracy on classes 0 and 1:",
          clf.score(X[y != 2], y[y != 2]), "max |coef|:", np.abs(clf.coef_).max())
```

## Expected Results

A finite class weight for class 2 (0 is the natural value, it only multiplies zero sample weights), and a "balanced" model that fits classes 0 and 1 about as well as the unweighted one.

## Actual Results

```
/home/arthur/scikit-learn/sklearn/utils/class_weight.py:96: RuntimeWarning: divide by zero encountered in divide
  recip_freq = xp.sum(weighted_class_counts) / (
/home/arthur/scikit-learn/sklearn/utils/class_weight.py:96: RuntimeWarning: divide by zero encountered in divide
  recip_freq = xp.sum(weighted_class_counts) / (
/home/arthur/scikit-learn/sklearn/linear_model/_logistic.py:1521: RuntimeWarning: invalid value encountered in multiply
  sample_weight = sample_weight * class_weight_
/home/arthur/scikit-learn/sklearn/linear_model/_logistic.py:623: ConvergenceWarning: lbfgs failed to converge after 0 iteration(s) (status=2):
ABNORMAL: 

You might also want to scale the data as shown in:
    https://scikit-learn.org/stable/modules/preprocessing.html
Please also refer to the documentation for alternative solver options:
    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
  n_iter_i = _check_optimize_result(
[0.67676768 0.65686275        inf]
None accuracy on classes 0 and 1: 0.9203980099502488 max |coef|: 1.155719857492348
balanced accuracy on classes 0 and 1: 0.4925373134328358 max |coef|: 0.0
```

## Root cause and possible fix

`sklearn/utils/class_weight.py:95-98` (upstream/main 289c4d5cfa): `weighted_class_counts` can contain zeros when `sample_weight` is given, and the division is not guarded.

Possible fix: set the weight of classes with a zero weighted count to 0, and count only the classes with a positive weighted count in the `n_classes` factor, so that masking a class gives the same weights as removing its samples. The second part changes the weights of the other classes by a constant factor (which rescales the effective regularization of `LogisticRegression`), hence the `unclear-spec` tag.

## Versions

```
sklearn: 1.10.dev0, commit 289c4d5cfa (upstream/main; local build f570497251, the "balanced" branch is identical)
python: 3.13, numpy: 2.5.3, scipy: 1.18.1, platform: linux
```

See also: `utils-balanced-class-weight-ignores-sample-weight-others.md` (other classifiers don't forward `sample_weight` to `compute_class_weight` yet; fixing that extends this bug to them).
