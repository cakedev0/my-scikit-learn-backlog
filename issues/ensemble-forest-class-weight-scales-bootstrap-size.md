# Forest bootstrap size with float `max_samples` scales with `class_weight` values

```json
{
    "estimators": ["RandomForestClassifier", "ExtraTreesClassifier"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 2,
    "difficulty": 1,
    "decision": 2,
    "decision_tags": ["unclear-spec", "output-change"]
}
```

## Classification

- **Priority (2)**: `class_weight={0: 1, 1: 100}` with `max_samples=0.5` is a common setup for imbalanced data, and it now draws ~50x more bootstrap indices than documented (time and memory grow with the class weights, e.g. tens of GB for 1e6 rows and a 1:10000 ratio). Small class weights go the other way and silently give trees fitted on a handful of rows, with a warning that talks about "the total sum of sample weights" although the user passed none. Limited to dict/list `class_weight` combined with a float `max_samples` (since 1.9).
- **Difficulty (1)**: pass the user `sample_weight` instead of the combined weights to `_get_n_samples_bootstrap`.
- **Decision (2)**: the current behavior was a deliberate choice in #31529 (bootstrap size relative to the sum of the combined `sample_weight * class_weight` used to draw indices), with an acknowledged ambiguity. Changing it needs agreement on what `max_samples` means with `class_weight`; it changes results for these fits.

## Upstream search

GraphQL search on scikit-learn/scikit-learn: "max_samples class_weight random forest", "max_samples sample_weight sum forest bootstrap", "is:open n_samples_bootstrap".

- #31529 (merged, released in 1.9): "FIX Draw indices using sample_weight in Random Forests". In the review, @antoinebaker noted "an ambiguity when we have `class_weight` and a relative `max_samples`" and proposed making the bootstrap size relative to the combined weights, which is what got merged; @ogrisel mentioned a future audit of `sample_weight` scale invariance. The consequence for `class_weight` dicts (size proportional to their scale, contradicting the docstring) does not seem to have been discussed, and the tests were switched to integer `max_samples`.
- #34884 / #35058 / #34673 (open): zero `sample_weight` with `bootstrap=False` in Bagging, different code path.

No open issue for this.

## Describe the bug and its user-facing impact

The `max_samples` docstring of the forests says: "If float, then draw `max_samples * X.shape[0]` unweighted samples or `max_samples * sample_weight.sum()` weighted samples." Without `sample_weight`, users expect `max_samples * n_samples` rows per tree. But `fit` passes the product of `sample_weight` and the expanded `class_weight` to `_get_n_samples_bootstrap`, so the bootstrap size is `max_samples * sum(class_weight[y_i])`. Multiplying all class weights by a constant, which should not matter, changes the number of drawn rows by that constant.

## Steps/Code to Reproduce

```python
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

X, y = make_classification(n_samples=1000, random_state=0)
for cw in [None, {0: 1, 1: 1}, {0: 1, 1: 100}, {0: 0.01, 1: 0.01}]:
    rf = RandomForestClassifier(n_estimators=1, max_samples=0.5, class_weight=cw, random_state=0)
    rf.fit(X, y)
    print(f"class_weight={cw!s:20} bootstrap size={len(rf.estimators_samples_[0])}")
```

## Expected Results

500 drawn rows per tree in all four cases (`max_samples * X.shape[0]`, as documented when no `sample_weight` is given); `class_weight` only changes which rows are likely to be drawn.

## Actual Results

```
UserWarning: Using the fractional value max_samples=0.5 when the total sum of sample weights is 10.0 results in a low number (5) of bootstrap samples. We recommend passing `max_samples` as an integer instead.
class_weight=None                 bootstrap size=500
class_weight={0: 1, 1: 1}         bootstrap size=500
class_weight={0: 1, 1: 100}       bootstrap size=25151
class_weight={0: 0.01, 1: 0.01}   bootstrap size=5
```

## Root cause and possible fix

`sklearn/ensemble/_forest.py:470` on upstream/main:

```python
n_samples_bootstrap = _get_n_samples_bootstrap(
    X.shape[0], self.max_samples, _sample_weight
)
```

where `_sample_weight = sample_weight * expanded_class_weight`. Possible fix: pass the user `sample_weight` (or `None`) here and keep the combined weights only as drawing probabilities. This keeps the frequency semantics of `sample_weight` that #31529 introduced, and makes `max_samples` invariant to the scale of `class_weight`. `class_weight="balanced"` is unaffected either way since its expanded weights sum to `n_samples` (or to `sample_weight.sum()`).

## Versions

```
sklearn: 1.10.dev0, commit f570497251 (logic identical to upstream/main 289c4d5cfa; line number above is from upstream/main)
python: 3.13, numpy: 2.5.3, scipy: 1.18.1, platform: linux
```
