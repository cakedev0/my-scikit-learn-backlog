# `StackingClassifier.get_feature_names_out` returns too few names for multilabel `predict_proba`

```json
{
    "estimators": ["StackingClassifier"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 1,
    "difficulty": 1,
    "decision": 0,
    "decision_tags": []
}
```

## Classification

- **Priority (1)**: wrong feature names and a crash with `set_output(transform="pandas")`, but only for multilabel targets with base estimators whose `predict_proba` returns a list of arrays (forests, KNN, `MultiOutputClassifier`, ...), and only when output names are used.
- **Difficulty (1)**: record the number of output columns per estimator rather than per element of `X_meta`.
- **Decision (0)**: no trade-off.

## Upstream search

GraphQL search on scikit-learn/scikit-learn: "StackingClassifier get_feature_names_out multilabel", "stacking multilabel feature names", "StackingClassifier _n_feature_outs". Only unrelated results (#24916, error message for unfitted `get_feature_names_out`). No match.

## Describe the bug and its user-facing impact

For a multilabel target, a base estimator whose `predict_proba` returns a list of `n_outputs` arrays contributes `n_outputs` columns to the stacked features. `_concatenate_predictions` appends one array per output to `X_meta`, then stores `self._n_feature_outs = [pred.shape[1] for pred in X_meta]`, i.e. one entry per output, not per estimator. `get_feature_names_out` zips this list with the estimator names, so it truncates: with 2 estimators and 3 labels, `transform` returns 6 columns but `get_feature_names_out` returns 2 names. With `passthrough=True`, the result is 22 names for 26 columns. `set_output(transform="pandas")` then fails.

## Steps/Code to Reproduce

```python
from sklearn.datasets import make_multilabel_classification
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier

X, y = make_multilabel_classification(n_samples=100, n_classes=3, random_state=0)
stack = StackingClassifier(
    estimators=[
        ("rf", RandomForestClassifier(n_estimators=5, random_state=0)),
        ("knn", KNeighborsClassifier()),
    ],
    final_estimator=MultiOutputClassifier(LogisticRegression()),
).fit(X, y)
print("transform output columns:", stack.transform(X).shape[1])
print("feature names:", stack.get_feature_names_out())
stack.set_output(transform="pandas").transform(X)
```

## Expected Results

6 names, e.g. `stackingclassifier_rf0`, ..., `stackingclassifier_rf2`, `stackingclassifier_knn0`, ..., and a pandas output that works.

## Actual Results

```
transform output columns: 6
feature names: ['stackingclassifier_rf' 'stackingclassifier_knn']
Traceback (most recent call last):
  ...
ValueError: Length mismatch: Expected axis has 6 elements, new values have 2 elements
```

## Root cause and possible fix

`sklearn/ensemble/_stacking.py:117` appends one array per output for list predictions, and `sklearn/ensemble/_stacking.py:133` derives `_n_feature_outs` from `X_meta` element by element. `get_feature_names_out` (`sklearn/ensemble/_stacking.py:335`) assumes one entry per non-dropped estimator. Fix: compute the column count per estimator in the loop over `predictions` (sum over outputs for list predictions), and maybe name the columns per output and class. A multilabel case in the existing feature-names test would cover it.

## Versions

```
sklearn: 1.10.dev0, commit f570497251 (sklearn/ensemble/_stacking.py identical to upstream/main 289c4d5cfa)
python: 3.13, numpy: 2.5.3, scipy: 1.18.1, platform: linux
```
