# AdaBoost gives weight 1.0 to a perfect estimator, so earlier estimators can outvote it

```json
{
    "estimators": ["AdaBoostClassifier", "AdaBoostRegressor"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 1,
    "difficulty": 1,
    "decision": 1,
    "decision_tags": ["output-change", "unclear-spec"]
}
```

## Classification

- **Priority (1)**: the ensemble can end up worse on the training set than its last member, silently. It needs a weak learner that reaches zero weighted training error after the first iteration (fairly deep trees on clean data), which is not the typical AdaBoost setting.
- **Difficulty (1)**: the change is local to the two `_boost` methods.
- **Decision (1)**: `unclear-spec`: SAMME's weight is infinite for zero error, so someone must pick a finite convention (e.g. keep only the perfect estimator, or use the maximum weight seen so far). `output-change`: predictions change for fits that hit this branch.

## Upstream search

GraphQL search on scikit-learn/scikit-learn: "AdaBoost perfect fit estimator weight", "AdaBoostClassifier estimator_error zero weight 1", "AdaBoost early termination perfect estimator weight", "AdaBoost zero error estimator_weight", "AdaBoost perfect classifier".

- #20443 (open): AdaBoost training error can increase with more trees; different cause (sample weight dynamics), no mention of this branch.
- #3815 (closed): first classifier worse than random, different branch.

No match. `git log -S` shows the `return sample_weight, 1.0, 0.0` line dates from the original 2013 implementation. Until 1.6, the default algorithm was SAMME.R, where every estimator weight is 1, so the constant was harmless there. SAMME.R was removed in 1.6 (#29997), and the 1.0 constant is now only used next to SAMME weights `log((1 - err) / err) + log(K - 1)`, which are typically well above 1.

## Describe the bug and its user-facing impact

When a boosting iteration (other than the first) produces an estimator with zero weighted training error, `_boost` returns an estimator weight of `1.0` and `fit` stops. In SAMME, the weight `learning_rate * (log((1 - err) / err) + log(K - 1))` grows to infinity as `err -> 0`, so the perfect estimator should dominate. With a fixed weight of 1.0, it is often the lowest-weighted member of the ensemble and is outvoted by the earlier, imperfect ones. Boosting then stops, so nothing corrects it.

In the reproducer, the second tree classifies the whole training set correctly, but the first tree (weight 4.08) outvotes it (weight 1.0) and the ensemble has 98.3% training accuracy. `AdaBoostRegressor` has the same branch for the weighted median.

## Steps/Code to Reproduce

```python
from sklearn.datasets import make_classification
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

X, y = make_classification(
    n_samples=60, n_features=4, n_informative=3, n_redundant=0, flip_y=0, random_state=5
)
ada = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=4, random_state=0), random_state=0
).fit(X, y)
n = len(ada.estimators_)
print("estimator_errors_ :", ada.estimator_errors_[:n])
print("estimator_weights_:", ada.estimator_weights_[:n])
print("training accuracy of the last estimator:", ada.estimators_[-1].score(X, y))
print("training accuracy of the ensemble      :", ada.score(X, y))
```

## Expected Results

The perfect estimator gets a weight at least as large as any other estimator (its SAMME weight is unbounded), so the ensemble is at least as accurate on the training set as that estimator: 1.0 here.

## Actual Results

```
estimator_errors_ : [0.01666667 0.        ]
estimator_weights_: [4.07753744 1.        ]
training accuracy of the last estimator: 1.0
training accuracy of the ensemble      : 0.9833333333333333
```

## Root cause and possible fix

`sklearn/ensemble/_weight_boosting.py:538` (`AdaBoostClassifier._boost`) and `sklearn/ensemble/_weight_boosting.py:1039` (`AdaBoostRegressor._boost`):

```python
if estimator_error <= 0:
    return sample_weight, 1.0, 0.0
```

The constant matched SAMME.R, where all weights are 1. Possible fixes: when the error is zero and earlier estimators exist, drop them and keep only the perfect one (it alone gives zero training error), or give it a weight that dominates (e.g. larger than the sum of the previous weights). The first option is simpler to reason about. The regressor needs the same treatment for its weighted median.

## Versions

```
sklearn: 1.10.dev0, commit f570497251 (sklearn/ensemble/_weight_boosting.py identical to upstream/main 289c4d5cfa)
python: 3.13, numpy: 2.5.3, scipy: 1.18.1, platform: linux
```
