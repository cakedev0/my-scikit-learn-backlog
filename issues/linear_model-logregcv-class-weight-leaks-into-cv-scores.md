# LogisticRegressionCV weights its CV scores by class_weight (regression from #34499)

```json
{
    "estimators": ["LogisticRegressionCV"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 3,
    "difficulty": 1,
    "decision": 0,
    "decision_tags": []
}
```

## Classification

- **Priority (3)**: silent change of the model selection for every `LogisticRegressionCV(class_weight=...)` user, with any `scoring` (including the default). `class_weight="balanced"` is common with `LogisticRegressionCV`. It's a regression on main (not released yet), so it's cheap to fix before 1.10.
- **Difficulty (1)**: keep the user `sample_weight` aside before multiplying it by the class weights, and pass that one to the scorer and to `_log_reg_scoring_path` for `sw_test`.
- **Decision (0)**: restores the 1.9 behavior, which is also what `GridSearchCV(LogisticRegression(class_weight=...))` does.

## Upstream search

Queries: "LogisticRegressionCV class_weight", "LogisticRegressionCV class_weight scoring", "LogisticRegressionCV class_weight score". Only old, unrelated issues come up (#5450, closed: "balanced" weights computed per fold vs on the full data).

Introduced by #34499 ("MNT clean up class_weight in LogisticRegression", merged in July 2026, not in any release). Before it, `class_weight` was passed down to `_log_reg_scoring_path` and only applied to the training folds; the scorer got the user `sample_weight` only.

## Describe the bug and its user-facing impact

Since #34499, `LogisticRegressionCV.fit` multiplies `sample_weight` by the class weights before the CV loop and then uses that product everywhere, including for scoring:

- default `scoring`: `_log_reg_scoring_path` calls `log_reg.score(X_test, y_test, sample_weight=sw_test)` with `sw_test` taken from the class-weighted `sample_weight`;
- explicit `scoring`: `score_kwargs["sample_weight"] = sample_weight` (or `process_routing(..., sample_weight=sample_weight)` with metadata routing) also gets the class-weighted array.

So with `class_weight="balanced"`, `scoring="accuracy"` silently becomes a class-weighted accuracy (close to balanced accuracy), and `scoring="neg_log_loss"` a class-weighted log loss. `class_weight` is a training choice: the selection metric should not depend on it, and it doesn't in `GridSearchCV(LogisticRegression(class_weight=...))`. Here, the selected `alpha_` changes by two orders of magnitude.

## Steps/Code to Reproduce

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import GridSearchCV, StratifiedKFold

X, y = make_classification(n_samples=300, weights=[0.9], random_state=0)
cv = StratifiedKFold(3)
alphas = np.logspace(-2, 4, 7)

lrcv = LogisticRegressionCV(
    alphas=alphas, l1_ratios=(0,), class_weight="balanced", cv=cv,
    scoring="accuracy", use_legacy_attributes=False,
).fit(X, y)
gs = GridSearchCV(
    LogisticRegression(class_weight="balanced"), {"alpha": alphas}, cv=cv,
    scoring="accuracy",
).fit(X, y)

print("mean CV accuracy per alpha")
print("  LogisticRegressionCV:", lrcv.scores_.mean(axis=0).ravel().round(3))
print("  GridSearchCV:        ", gs.cv_results_["mean_test_score"].round(3))
print("selected alpha: LogisticRegressionCV", lrcv.alpha_, "GridSearchCV", gs.best_params_["alpha"])
```

## Expected Results

Same per-alpha CV accuracies and same selected `alpha` as `GridSearchCV` (the folds and models are the same), as in scikit-learn 1.9.

## Actual Results

```
mean CV accuracy per alpha
  LogisticRegressionCV: [0.782 0.797 0.797 0.828 0.832 0.808 0.56 ]
  GridSearchCV:         [0.863 0.867 0.87  0.87  0.88  0.9   0.903]
selected alpha: LogisticRegressionCV 100.0 GridSearchCV 10000.0
```

## Root cause and possible fix

`sklearn/linear_model/_logistic.py` (commit 289c4d5cfa):

- line 2681: `sample_weight = sample_weight * class_weight_`, then this array is used for training (fine) and for scoring:
- lines 2707-2716: `process_routing(self, "fit", sample_weight=sample_weight, ...)` and `score_kwargs["sample_weight"] = sample_weight`;
- line 2755 passes it to `_log_reg_scoring_path`, where `sw_test = sample_weight[test_xp]` (line 1021) is used by the default scorer (line 1072).

Fix: keep `sample_weight_user = sample_weight` before applying the class weights, use it for the routing / `score_kwargs`, and pass it separately to `_log_reg_scoring_path` for `sw_test` (the training folds keep using the class-weighted array). A regression test can compare `scores_` with `cross_val_score(LogisticRegression(class_weight=...), ...)` on the same folds.

## Versions

```
sklearn: 1.10.dev0, commit 289c4d5cfa (upstream/main, pure-Python files overlaid on the local build: no compiled file differs in sklearn/linear_model)
python: 3.13.14, numpy: 2.5.3, scipy: 1.18.1, platform: linux
```
