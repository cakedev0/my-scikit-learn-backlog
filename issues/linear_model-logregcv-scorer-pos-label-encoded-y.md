# LogisticRegressionCV misreads or rejects `pos_label` and `labels` of user scorers because it scores on label-encoded targets

```json
{
    "estimators": ["LogisticRegressionCV"],
    "existing_issue": "https://github.com/scikit-learn/scikit-learn/issues/34531",
    "reproduced": true,
    "priority": 2,
    "difficulty": 2,
    "decision": 0,
    "decision_tags": []
}
```

## Classification

- **Priority (2)**: a scorer with an explicit `pos_label` is the standard way to score a binary problem whose labels aren't `{0, 1}` or `{-1, 1}`. Depending on the metric, `LogisticRegressionCV` either crashes or silently selects `alpha` on the score of the wrong class. The silent case needs integer labels that overlap with the encoded ones (e.g. `{1, 2}`), which is less common.
- **Difficulty (2)**: the scorer kwargs referring to labels (`pos_label`, `labels`) must be translated with the fitted `LabelEncoder` before being passed to `_log_reg_scoring_path`, and the existing override of `pos_label`/`labels` must not collide with them. Not hard, but it touches the "rebuild the scorer" hack added in #32828.
- **Decision (0)**: user scorers would then behave as in `GridSearchCV(LogisticRegression(), ...)`.

## Upstream search

Queries: "LogisticRegressionCV pos_label scorer", "LogisticRegressionCV make_scorer".

- #34531 (open): `LogisticRegressionCV(scoring="f1_macro")` raises the same `TypeError: make_scorer() got multiple values for keyword argument 'pos_label'` on binary problems (the `f1_macro` scorer stores `pos_label=None`), reported as a 1.9 regression. Same root cause as case 1 below.
- #34540 (open PR) merges the kwargs with the internal `pos_label=n_classes - 1` taking precedence: the `TypeError` goes away but a user `pos_label` is then silently ignored (always the last class).
- #34624 (open PR) only sets the internal `pos_label` when the scorer doesn't have one: the user `pos_label` is then compared to the encoded `y`, which gives the `ValueError` or the silently wrong class of case 2 below.
- #34560, #34703 (closed PRs) for #34531.

New argument for #34531: neither open PR handles that `pos_label`/`labels` are expressed in the original labels while `LogisticRegressionCV` scores on encoded ones. Translating them with the fitted `LabelEncoder` fixes the crash, the string-label `ValueError` and the silently wrong class at once.

Related: #32828 (merged, in 1.9.0) added the `labels=`/`pos_label=` override in `_log_reg_scoring_path` to handle folds with missing classes. That override is what turns the old "pos_label is not a valid label" error into a `TypeError` for metrics that accept `labels`.

## Describe the bug and its user-facing impact

`LogisticRegressionCV.fit` label-encodes `y` (`y_encoded = le.transform(y)`, `_logistic.py:2664`) and calls the user scorer on `y_test` in `{0, ..., n_classes - 1}` (`_logistic.py:1108`). Any scorer argument expressed in terms of the original labels is then wrong:

1. Metrics with both `pos_label` and `labels` (`f1_score`, `precision_score`, `recall_score`, `fbeta_score`, `jaccard_score`, `brier_score_loss`): the scorer is rebuilt with `pos_label=n_classes - 1` and `labels=...` **plus** the user kwargs (`_logistic.py:1093-1105`), so `make_scorer(f1_score, pos_label=...)` raises `TypeError: make_scorer() got multiple values for keyword argument 'pos_label'`. Same for `make_scorer(log_loss, labels=[...])` with `labels`.
2. Metrics with `pos_label` but no `labels` (`average_precision_score`, or a custom metric): the user `pos_label` is compared to the encoded `y_test`. With string labels it raises `ValueError: pos_label=yes is not a valid label: It should be one of [0. 1.]`. With integer labels overlapping the encoded ones, e.g. classes `{1, 2}` and `pos_label=1`, it silently scores the other class (encoded 1 is the original class 2), and `alpha_` is selected on that score.

The same scorers work as expected with `GridSearchCV(LogisticRegression(), ...)` and `cross_val_score`.

## Steps/Code to Reproduce

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import average_precision_score, f1_score, make_scorer
from sklearn.model_selection import StratifiedKFold, cross_val_score

X, y = make_classification(n_samples=200, weights=[0.8], random_state=0)
y = y + 1  # classes are {1, 2}
cv = StratifiedKFold(3)

# 1. Silently wrong: pos_label refers to the original labels, but LogisticRegressionCV
# scores on label-encoded targets {0, 1}, so pos_label=1 becomes class 2.
ap_class1 = make_scorer(
    average_precision_score, pos_label=1, response_method="predict_proba"
)
lrcv = LogisticRegressionCV(
    alphas=[1.0], l1_ratios=(0,), cv=cv, scoring=ap_class1, use_legacy_attributes=False
).fit(X, y)
print("LogisticRegressionCV scores_:", lrcv.scores_.ravel().round(3))
print("cross_val_score:             ",
      cross_val_score(LogisticRegression(alpha=1.0), X, y, cv=cv, scoring=ap_class1).round(3))

# 2. Crash: scorers with pos_label and a `labels` parameter (f1, precision, ...)
LogisticRegressionCV(scoring=make_scorer(f1_score, pos_label=1), l1_ratios=(0,),
                     use_legacy_attributes=False).fit(X, y)
```

## Expected Results

`scores_` equal to the `cross_val_score` values for the same `alpha` and folds (average precision of class 1), and the `f1_score` scorer accepted, as in `GridSearchCV`.

## Actual Results

```
LogisticRegressionCV scores_: [0.95  0.902 0.904]
cross_val_score:              [0.996 0.962 0.993]
Traceback (most recent call last):
  ...
  File "sklearn/linear_model/_logistic.py", line 1098, in _log_reg_scoring_path
    scoring = make_scorer(
        scoring._score_func,
    ...<4 lines>...
        **getattr(scoring, "_kwargs", {}),
    )
TypeError: sklearn.metrics._scorer.make_scorer() got multiple values for keyword argument 'pos_label'
```

With string labels `{"no", "yes"}`: `make_scorer(f1_score, pos_label="yes")` gives the same `TypeError`, `make_scorer(average_precision_score, pos_label="yes", response_method="predict_proba")` gives `ValueError: pos_label=yes is not a valid label: It should be one of [0. 1.]`, and `make_scorer(log_loss, labels=["no", "yes"], ...)` gives `TypeError: ... multiple values for keyword argument 'labels'`.

## Root cause and possible fix

`sklearn/linear_model/_logistic.py` (commit 289c4d5cfa): `fit` passes `y_encoded` to `_log_reg_scoring_path` (line 2664), and the scorer only knows the original labels. Lines 1093-1105 rebuild the scorer with `labels=log_reg.classes_` (encoded) and `pos_label=n_classes - 1`, then splat `scoring._kwargs` on top, which collides when the user set either key. For metrics without `labels`, the user `pos_label` is passed through unchanged (line 1108).

Possible fix: in `fit`, translate label-valued scorer kwargs with the fitted encoder before the CV loop (`pos_label -> le.transform([pos_label])[0]`, `labels -> le.transform(labels)`), and in `_log_reg_scoring_path` only set `pos_label`/`labels` when the user didn't. Alternatively, score on the original labels by setting `log_reg.classes_ = self.classes_` and passing `y` (not `y_encoded`) for the test folds, which would make every scorer behave as in `GridSearchCV`, but it means indexing both `y` and `y_encoded` per fold.

## Versions

```
sklearn: 1.10.dev0, commit 289c4d5cfa (upstream/main, pure-Python files overlaid on the local build: no compiled file differs in sklearn/linear_model)
python: 3.13.14, numpy: 2.5.3, scipy: 1.18.1, platform: linux
```
