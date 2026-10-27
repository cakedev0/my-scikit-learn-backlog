# HistGradientBoosting early stopping passes `sample_weight` to a callable `scoring` documented as `scorer(estimator, X, y)`

```json
{
    "estimators": ["HistGradientBoostingClassifier", "HistGradientBoostingRegressor"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 1,
    "difficulty": 1,
    "decision": 1,
    "decision_tags": ["unclear-spec"]
}
```

## Classification

- **Priority (1)**: loud `TypeError` with an easy workaround (add a `sample_weight=None` parameter to the scorer). It is surprising with `class_weight`, where the user never passed any weight.
- **Difficulty (1)**: only pass `sample_weight` when the scorer accepts it, or document it.
- **Decision (1)**: someone has to choose between documenting that callables must accept `sample_weight` and silently dropping the weights for callables that don't (`unclear-spec`).

## Upstream search

Core API dump of all open issues/PRs and of issues labeled `module:ensemble`, grepped for scoring, scorer, early stopping. No match. The full-text `gh search issues` query ("HistGradientBoosting scoring callable sample_weight") failed on the shared search API rate limit, so this relies on titles only.

## Describe the bug and its user-facing impact

The `scoring` docstring says a callable must have the signature `scorer(estimator, X, y)`. But when early stopping is on and the training data has weights, HGB calls `scorer(estimator, X, y, sample_weight=...)`, so a scorer written as documented fails with `TypeError: my_scorer() got an unexpected keyword argument 'sample_weight'`.

Weights are there as soon as `sample_weight` is passed to `fit`, and also with `class_weight` alone, since class weights are turned into sample weights. Scorers built with `make_scorer` accept `sample_weight`, so they are not affected.

## Steps/Code to Reproduce

```python
from sklearn.datasets import make_classification
from sklearn.ensemble import HistGradientBoostingClassifier

X, y = make_classification(n_samples=300, weights=[0.8], random_state=0)

def my_scorer(estimator, X, y):
    return (estimator.predict(X) == y).mean()

clf = HistGradientBoostingClassifier(early_stopping=True, scoring=my_scorer)
clf.fit(X, y)  # works
clf.set_params(class_weight="balanced")
clf.fit(X, y)
```

## Expected Results

The fit works with a scorer that has the documented signature, with or without weights.

## Actual Results

```
  File "/home/arthur/scikit-learn/sklearn/ensemble/_hist_gradient_boosting/gradient_boosting.py", line 1053, in _score_with_raw_predictions
    return self._scorer(self, X, y, sample_weight=sample_weight)
           ~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
TypeError: my_scorer() got an unexpected keyword argument 'sample_weight'
```

## Root cause and possible fix

`_score_with_raw_predictions` (`sklearn/ensemble/_hist_gradient_boosting/gradient_boosting.py:1053` on the commit below) always passes `sample_weight=` to `self._scorer` when weights exist, and `check_scoring` returns user callables unchanged. Weights were passed to the scorer since sample weight support in HGB (#14696).

Possible fixes:

- pass `sample_weight` only if the callable accepts it (`"sample_weight" in inspect.signature(scorer).parameters`, or `isinstance(scorer, _BaseScorer)`), and maybe warn that the validation score is then unweighted;
- or document in `scoring` that callables receive `sample_weight` when the training data is weighted (including through `class_weight`).

## Versions

```
sklearn: 1.10.dev0, commit 289c4d5cfa (upstream/main; run on a local editable build whose gradient_boosting.py differs from upstream/main only in docstrings)
python: 3.13, numpy: 2.5.3, scipy: 1.18.1, platform: linux
```
