# NearestCentroid.predict disagrees with predict_proba and decision_function under uniform priors

```json
{
    "estimators": ["NearestCentroid"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 2,
    "difficulty": 2,
    "decision": 2,
    "decision_tags": ["output-change", "unclear-spec"]
}
```

## Classification

- **Priority (2)**: silent inconsistency in the default configuration (`priors="uniform"`, `metric="euclidean"`): `argmax(predict_proba)` differs from `predict` on 8-16% of the samples in the reproducer. Any downstream tool mixing the two (ROC curves vs accuracy, `CalibratedClassifierCV`, `TunedThresholdClassifierCV`, soft voting) is affected, and nothing looks wrong.
- **Difficulty (2)**: the code change is small (one branch in `predict`), but someone has to decide which of the two rules is the intended one.
- **Decision (2)**: `unclear-spec`: plain Euclidean nearest centroid (historical `predict`) and the standardized discriminant of Tibshirani et al. (the rule used by `decision_function`/`predict_proba` and by `predict` with non-uniform priors) are both reasonable. `output-change`: aligning `predict` on the discriminant changes the default predictions of a long-standing estimator; aligning the discriminant on plain distances changes `predict_proba` and makes the prior term inconsistent with the unscaled distances.

## Upstream search

- `NearestCentroid predict`: #26689 (merged, "ENH Add decision_function, predict_proba and predict_log_proba for NearestCentroid estimator") introduced the two code paths. In its review, the author states that with uniform priors "calculations can be simplified considerably by skipping the division by the within-class standard deviation", and the reviewer asked to keep `pairwise_distances_argmin` in `predict` for speed in that case. That simplification is not valid: dividing each feature by a different standard deviation changes which centroid is closest. #24645 (merged) is the earlier `pairwise_distances_argmin` optimization.
- `NearestCentroid` (all issues): #26659 (open, the feature request that #26689 implemented, no comments), #7679 (closed, shrinkage implementation), #23890, #22985, #18324 (closed, metrics and zero-variance data). None mentions the `predict`/`predict_proba` mismatch.

## Describe the bug and its user-facing impact

`NearestCentroid` has two prediction rules:

- `predict` with uniform priors (the default) returns the class of the closest centroid in plain Euclidean distance.
- `decision_function`, `predict_proba`, `predict_log_proba`, and `predict` with non-uniform priors use `-||(x - mu_k) / s||^2 + 2 log(pi_k)`, where `s` is the pooled within-class standard deviation per feature (`within_class_std_dev_`).

As soon as features have different within-class spreads, the two rules disagree. So under default settings `predict(X)` differs from `classes_[predict_proba(X).argmax(axis=1)]`, and the sign of the binary `decision_function` doesn't match `predict`. Also, `predict` jumps from one rule to the other when the priors move from exactly uniform to almost uniform.

## Steps/Code to Reproduce

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.neighbors import NearestCentroid

X, y = make_classification(n_samples=200, n_features=4, n_informative=3,
                           n_redundant=0, n_classes=3, random_state=0)
X[:, 0] *= 10  # features on different scales

clf = NearestCentroid().fit(X, y)
pred = clf.predict(X)
pred_from_proba = clf.classes_[clf.predict_proba(X).argmax(axis=1)]
pred_from_decision = clf.classes_[clf.decision_function(X).argmax(axis=1)]
print("predict != argmax(predict_proba):", np.mean(pred != pred_from_proba))
print("predict != argmax(decision_function):", np.mean(pred != pred_from_decision))

# Almost uniform priors switch predict to the standardized distance
clf2 = NearestCentroid(priors=[0.334, 0.333, 0.333]).fit(X, y)
print("predict changes with priors=[.334, .333, .333]:",
      np.mean(clf2.predict(X) != pred))
```

## Expected Results

`predict` agrees with `argmax(predict_proba)` and `argmax(decision_function)` (scikit-learn convention, also asked for in the review of #26689), and a tiny change of `priors` gives a tiny change of predictions. All three printed values should be 0.

## Actual Results

```
predict != argmax(predict_proba): 0.16
predict != argmax(decision_function): 0.16
predict changes with priors=[.334, .333, .333]: 0.16
```

## Root cause and possible fix

`sklearn/neighbors/_nearest_centroid.py:286-299` (upstream main 289c4d5cfa): when `np.isclose(self.class_prior_, 1 / len(self.classes_)).all()`, `predict` calls `pairwise_distances_argmin(X, self.centroids_)` on the raw features. `_decision_function` (`:317-325`) divides `X` and the centroids by `within_class_std_dev_` before taking squared distances. With uniform priors the log-prior term is constant, but the per-feature scaling is not, so the shortcut isn't equivalent.

Possible fixes:

- keep the fast path but run it on standardized data: `pairwise_distances_argmin(X / s, centroids_ / s)` (with the same `mask` for zero std). `predict` then matches `decision_function` everywhere, at the cost of changing default predictions;
- or make `_decision_function` use unscaled distances and keep `predict` as is. That keeps historical predictions but changes `predict_proba` and departs from the shrunken centroid paper cited in the docstring.

Both change outputs, which is why this needs a decision. The metric="manhattan" path has no `predict_proba`/`decision_function`, so it's not affected.

## Versions

```
sklearn: 1.10.dev0, commit f570497251 (code identical on upstream/main 289c4d5cfa)
python: 3.13.14, numpy: 2.5.3, scipy: 1.18.1, platform: linux
```
