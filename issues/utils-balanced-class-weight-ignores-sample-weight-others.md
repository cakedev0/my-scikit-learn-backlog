# `class_weight="balanced"` ignores `sample_weight` in RidgeClassifier, trees, HistGradientBoostingClassifier and SGDClassifier

```json
{
    "estimators": ["RidgeClassifier", "RidgeClassifierCV", "DecisionTreeClassifier", "ExtraTreeClassifier", "HistGradientBoostingClassifier", "SGDClassifier", "Perceptron"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 2,
    "difficulty": 1,
    "decision": 1,
    "decision_tags": ["output-change"]
}
```

## Classification

- **Priority (2)**: silent, plausible-looking results whenever `class_weight="balanced"` is combined with non-uniform `sample_weight`: classes are balanced on raw counts instead of weighted counts, so integer weights are not equivalent to repeated samples. Same bug as the one fixed for `LogisticRegression`/`LinearSVC` in #30057, left in several other popular classifiers.
- **Difficulty (1)**: forward `sample_weight` to `compute_class_weight` at four call sites (`compute_sample_weight` needs a `sample_weight` argument, or the call sites switch to `compute_class_weight`).
- **Decision (1)**: `output-change` for these estimators when fitted with both `class_weight="balanced"` and non-uniform `sample_weight`. Nothing changes for uniform weights.

## Upstream search

Queries (`gh issue list/pr list --search`): `class_weight balanced sample_weight`, `compute_sample_weight sample_weight`.

- #30056 (closed) / #30057 (merged): the same bug for `LinearSVC`; the PR also fixed `LogisticRegression(CV)` and added `sample_weight` to `compute_class_weight`. In the review, @ogrisel suspected "this problem is not just for `LinearSVC` but for any model that accepts `class_weight="balanced"`", and @antoinebaker pointed at forests and `BaseSVC`. The author said they would look at those next; forests were fixed since (they now pass `sample_weight`), the estimators listed here were not, and I found no open issue or PR for them.
- #21504 (open, meta-issue "TST Check correct interactions of `class_weight` and `sample_weight`"): about testing that a dict `class_weight` and `sample_weight` compose correctly. Related, but it doesn't cover how `"balanced"` weights are computed.
- #32941, #32639 (closed): zero-weight class division by zero in `compute_class_weight`, a different problem (tracked in `issues/utils-compute-class-weight-balanced-zero-weight-class.md`).
- #10517 (closed, "RidgeClassifier with 'balanced' classweight", 2018): a sag convergence problem with `"balanced"`, not about sample weights.

## Describe the bug and its user-facing impact

Since #30057, `compute_class_weight("balanced", ..., sample_weight=sw)` balances the weighted class counts, and `LogisticRegression`, `LogisticRegressionCV` and `LinearSVC` pass their `sample_weight` to it. The other classifiers that accept `class_weight="balanced"` still compute it from unweighted counts:

- `RidgeClassifier` and `RidgeClassifierCV` (`_RidgeClassifierMixin._prepare_data`),
- `DecisionTreeClassifier` and `ExtraTreeClassifier` (`BaseDecisionTree._fit`),
- `HistGradientBoostingClassifier` (`_finalize_sample_weight`),
- `SGDClassifier` and `Perceptron` (`BaseSGDClassifier._partial_fit`).

So with `class_weight="balanced"`, up-weighting the samples of a class (or zero-weighting some samples) changes the class balance the model sees, while it shouldn't: the "balanced" weights are meant to cancel the class frequencies. A user who passes, e.g., survey weights or deduplication counts as `sample_weight` together with `class_weight="balanced"` gets a model that is not balanced on the population they described.

The SVC/NuSVC variant of this bug is in `issues/svm-svc-balanced-class-weight-ignores-sample-weight.md`. `RandomForestClassifier`/`ExtraTreesClassifier` already pass `sample_weight` for `"balanced"` on upstream main, so they are not listed here.

## Steps/Code to Reproduce

The reference is the documented meaning of `"balanced"` with sample weights, computed explicitly with `compute_class_weight(..., sample_weight=sw)` and passed through `sample_weight`. For each estimator, passing the same class weights as a dict with `class_weight=dict(enumerate(cw))` gives exactly the "explicit" result, so the reference is reachable through the public API.

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_class_weight

X, y = make_classification(n_samples=200, weights=[0.7], random_state=0)
sw = np.where(y == 1, 3.0, 1.0)  # up-weight class 1

# "balanced" with sample_weight, as LogisticRegression/LinearSVC do since #30057
cw = compute_class_weight("balanced", classes=np.unique(y), y=y, sample_weight=sw)

for clf in [
    LogisticRegression(),
    RidgeClassifier(),
    DecisionTreeClassifier(max_depth=3, random_state=0),
    HistGradientBoostingClassifier(max_iter=10, early_stopping=False),
    SGDClassifier(random_state=0),
]:
    balanced = clf.set_params(class_weight="balanced").fit(X, y, sample_weight=sw)
    pred_balanced = balanced.predict_proba(X)[:, 1] if hasattr(clf, "predict_proba") else balanced.decision_function(X)
    explicit = clf.set_params(class_weight=None).fit(X, y, sample_weight=sw * cw[y])
    pred_explicit = explicit.predict_proba(X)[:, 1] if hasattr(clf, "predict_proba") else explicit.decision_function(X)
    print(f"{type(clf).__name__:31} max |balanced - explicit weighted balancing| = "
          f"{np.abs(pred_balanced - pred_explicit).max():.1e}")
```

## Expected Results

All differences are 0, as for `LogisticRegression`. Equivalently, `class_weight="balanced"` with integer `sample_weight` gives the same model as with repeated samples (checked for `RidgeClassifier` and `DecisionTreeClassifier(max_depth=3)`, which are weight/repetition equivalent with `class_weight=None`).

## Actual Results

```
LogisticRegression              max |balanced - explicit weighted balancing| = 0.0e+00
RidgeClassifier                 max |balanced - explicit weighted balancing| = 4.7e-01
DecisionTreeClassifier          max |balanced - explicit weighted balancing| = 9.7e-01
HistGradientBoostingClassifier  max |balanced - explicit weighted balancing| = 2.7e-01
SGDClassifier                   max |balanced - explicit weighted balancing| = 2.9e+02
```

## Root cause and possible fix

Call sites on upstream main 289c4d5cfa that compute the class weights without `sample_weight`:

- `sklearn/linear_model/_ridge.py:1318`: `compute_sample_weight(self.class_weight, y)`;
- `sklearn/tree/_classes.py:374`: `compute_sample_weight(self.class_weight, y_original)`;
- `sklearn/ensemble/_hist_gradient_boosting/gradient_boosting.py:2094`: `compute_sample_weight(self.class_weight, y)`;
- `sklearn/linear_model/_stochastic_gradient.py:627`: `compute_class_weight(self.class_weight, classes=self.classes_, y=y)`.

`sample_weight` is available at all four places. The simplest fix is to add a `sample_weight` argument to `compute_sample_weight` (forwarded to `compute_class_weight` for `"balanced"`), pass it at these call sites, and add `class_weight="balanced"` cases to the `check_sample_weight_equivalence` parameters in `instance_generator.py` for the deterministic ones (`RidgeClassifier`, `DecisionTreeClassifier`), as was done for `LinearSVC`.

Once fixed, these estimators also inherit the zero-weight-class problem of `compute_class_weight` (`issues/utils-compute-class-weight-balanced-zero-weight-class.md`), so it is worth fixing both together.

## Versions

```
sklearn: 1.10.dev0, commit f570497251 (the four call sites are identical on upstream/main 289c4d5cfa)
python: 3.13.14, numpy: 2.5.3, scipy: 1.18.1, platform: linux
```

See also: `utils-compute-class-weight-balanced-zero-weight-class.md` (forwarding `sample_weight` to `compute_class_weight` makes these estimators hit its `inf` weight when a class has only zero weights).
