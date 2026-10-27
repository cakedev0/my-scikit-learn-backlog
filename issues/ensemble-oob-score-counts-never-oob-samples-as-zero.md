# Forest and Bagging `oob_score_` count never-out-of-bag samples as predicting 0 / the first class

```json
{
    "estimators": ["RandomForestClassifier", "RandomForestRegressor", "ExtraTreesClassifier", "ExtraTreesRegressor", "BaggingClassifier", "BaggingRegressor"],
    "existing_issue": "https://github.com/scikit-learn/scikit-learn/issues/21490",
    "reproduced": true,
    "priority": 1,
    "difficulty": 2,
    "decision": 1,
    "decision_tags": ["output-change", "api-change"]
}
```

## Classification

- **Priority (1)**: the score can be very wrong (R² 0.87 reported as 0.03 below), but only with few estimators (a sample is never out-of-bag with probability ~0.632^n_estimators, so < 1e-4 with 20 trees and negligible with the default 100), and a `UserWarning` is raised.
- **Difficulty (2)**: mask the never-out-of-bag rows in four `_set_oob_score*` methods (forest classifier/regressor, Bagging classifier/regressor), set them to NaN in `oob_decision_function_` / `oob_prediction_`, and handle the multi-output case.
- **Decision (1)**: `output-change`: `oob_score_` changes whenever the warning fires. `api-change`: `oob_prediction_` / `oob_decision_function_` would contain NaN for those rows (as the docstring of the classifiers already says) instead of 0.

## Upstream search

- `gh search issues --include-prs`: "oob_score do not have OOB scores", "oob score samples without oob predictions zero", "oob_prediction_ zero", "oob_decision_function_ NaN".
- #21490 (open): reports that the docstrings of `oob_decision_function_` (NaN) and `oob_prediction_` no longer match the code (0 since #19162). @glemaitre confirms in the thread that the score itself is wrong because the zero-initialized rows are included, and proposes NaN predictions plus a mask in the score. This file is that bug; it adds the size of the effect and the Bagging estimators.
- #33879 (open PR): "FIX OOB predictions produce NaN for never-left-out samples in forest/bagging", implements that proposal; not merged.
- #16792 (open, feature request): asks for a way to score over all inputs when the warning fires, assuming the score then only uses rows with OOB predictions, which is not what the code does.
- `issues/ensemble-oob-score-ignores-sample-weight.md` is a different bug (weights not used in the score).

## Describe the bug and its user-facing impact

When `oob_score=True` and some training rows are in-bag for every estimator, the forests and Bagging estimators warn "Some inputs do not have OOB scores" but still include those rows in `oob_score_`, with a dummy prediction:

- regressors: `oob_prediction_` is 0 for these rows and `r2_score` is computed on all rows. With targets far from 0, a handful of rows is enough to drive R² to ~0;
- classifiers: `oob_decision_function_` is all zeros (forests) or NaN (Bagging) for these rows, `argmax` picks class index 0, and these rows count in the accuracy as predictions of the first class.

So `oob_score_` is not the out-of-bag estimate of the score but a mix with a constant predictor. Users comparing `oob_score_` across `n_estimators` values (a common way to choose it) see an artificially low score for small forests. The forest classifiers' docstring also says `oob_decision_function_` "might contain NaN", which is false.

## Steps/Code to Reproduce

```python
import numpy as np
from sklearn.datasets import make_regression, make_classification
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import r2_score, accuracy_score

X, y = make_regression(n_samples=500, n_features=5, noise=1, random_state=0)
y += 1000  # targets far from 0
reg = RandomForestRegressor(n_estimators=10, oob_score=True, random_state=0).fit(X, y)

# Rows that are out-of-bag for at least one tree
in_bag_everywhere = np.ones(len(y), dtype=bool)
for samples in reg.estimators_samples_:
    in_bag_everywhere &= np.isin(np.arange(len(y)), samples)
has_oob = ~in_bag_everywhere
print("rows without any OOB prediction:", (~has_oob).sum())
print("their oob_prediction_:", reg.oob_prediction_[~has_oob])
print("oob_score_:", reg.oob_score_)
print("R2 on rows with an OOB prediction:", r2_score(y[has_oob], reg.oob_prediction_[has_oob]))

X, y = make_classification(n_samples=500, weights=[0.1], random_state=0)
clf = RandomForestClassifier(n_estimators=5, oob_score=True, random_state=0).fit(X, y)
in_bag_everywhere = np.ones(len(y), dtype=bool)
for samples in clf.estimators_samples_:
    in_bag_everywhere &= np.isin(np.arange(len(y)), samples)
has_oob = ~in_bag_everywhere
print("rows without any OOB prediction:", (~has_oob).sum(), "true labels:", np.bincount(y[~has_oob]))
print("their oob_decision_function_:", clf.oob_decision_function_[~has_oob][:3])
print("oob_score_:", clf.oob_score_)
print("accuracy on rows with an OOB prediction:",
      accuracy_score(y[has_oob], clf.oob_decision_function_[has_oob].argmax(axis=1)))
```

## Expected Results

`oob_score_` computed on the rows that have at least one out-of-bag prediction (0.871 and 0.843 here), and NaN in `oob_prediction_` / `oob_decision_function_` for the other rows, as the classifiers' docstring says.

## Actual Results

```
/home/arthur/scikit-learn/sklearn/ensemble/_forest.py:648: UserWarning: Some inputs do not have OOB scores. This probably means too few trees were used to compute any reliable OOB estimates.
  warn(
/home/arthur/scikit-learn/sklearn/ensemble/_forest.py:648: UserWarning: Some inputs do not have OOB scores. This probably means too few trees were used to compute any reliable OOB estimates.
  warn(
rows without any OOB prediction: 5
their oob_prediction_: [0. 0. 0. 0. 0.]
oob_score_: 0.0331631077905975
R2 on rows with an OOB prediction: 0.8711927616018056
rows without any OOB prediction: 53 true labels: [ 4 49]
their oob_decision_function_: [[0. 0.]
 [0. 0.]
 [0. 0.]]
oob_score_: 0.762
accuracy on rows with an OOB prediction: 0.843400447427293
```

Five rows out of 500 turn an R² of 0.87 into 0.03. `BaggingRegressor(n_estimators=10, oob_score=True, random_state=0)` on the same regression data gives `oob_score_` 0.027 vs 0.865 on the rows with an OOB prediction.

## Root cause and possible fix

- `sklearn/ensemble/_forest.py:647-657` (`_compute_oob_predictions`): rows with `n_oob_pred == 0` get their count set to 1, so their prediction stays at its initial 0. `_set_oob_score_and_attributes` (lines 874-898 for classifiers, 1200-1220 for regressors) then scores all rows.
- `sklearn/ensemble/_bagging.py:910-921` (classifier, 0/0 = NaN in `oob_decision_function_` but `argmax(predictions)` = class 0 in the score) and `1462-1473` (regressor, prediction 0).

Fix, as proposed by @glemaitre in #21490 and implemented in #33879: set never-out-of-bag rows to NaN and compute the default score (and the callable `oob_score`) only on rows with a prediction. The trade-off is that `oob_prediction_` can now contain NaN, which downstream code may not expect.

## Versions

```
sklearn: 1.10.dev0, commit 289c4d5cfa
python: 3.13.14, numpy: 2.5.3, scipy: 1.18.1, platform: linux
```
