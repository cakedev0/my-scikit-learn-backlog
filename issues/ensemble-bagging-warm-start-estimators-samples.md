# `estimators_samples_` of Bagging and IsolationForest only covers the last `fit` call with `warm_start=True`

```json
{
    "estimators": ["BaggingClassifier", "BaggingRegressor", "IsolationForest"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 1,
    "difficulty": 1,
    "decision": 0,
    "decision_tags": []
}
```

## Classification

- **Priority (1)**: silently wrong fitted attribute (`estimators_samples_` shorter than `estimators_` and shifted), but it needs two rarely combined features: `warm_start` refits and reading `estimators_samples_`, an attribute few users touch. Predictions are unaffected. (priority review, 2026-10-23: was 2)
- **Difficulty (1)**: keep the seeds of previous fits (`self._seeds = np.concatenate([old_seeds, seeds])` on warm start).
- **Decision (0)**: no trade-off.

## Upstream search

GraphQL search on scikit-learn/scikit-learn: "estimators_samples_ warm_start", "_seeds warm_start bagging", "estimators_samples_ bagging", "bagging warm_start estimators_samples_ wrong", "warm_start estimators_features_ samples bagging".

- #16436 / #16437 (closed / merged): `_get_estimators_indices` did not regenerate indices properly. Different cause (RNG use), fixed.
- #9524 / #9723 (closed / merged): non-deterministic `estimators_samples_`. Different cause, fixed.
- #7118 (merged): introduced `_seeds` (to regenerate indices instead of storing them), without handling warm start. #4500 (merged) added `warm_start` to bagging before that.

No match for this bug.

## Describe the bug and its user-facing impact

`BaseBagging._fit` stores the seeds of the estimators fitted in the current call only (`self._seeds = seeds`). `_get_estimators_indices`, used by `estimators_samples_`, iterates over `self._seeds`. After a warm-start refit that adds estimators, `estimators_samples_` therefore has `n_estimators - n_previous` entries instead of `n_estimators`, and entry `i` describes estimator `n_previous + i`. Code like `zip(bag.estimators_, bag.estimators_samples_)` silently pairs the wrong samples with each estimator and drops the last ones.

`estimators_features_` is extended correctly, so the two attributes also disagree with each other. `IsolationForest` inherits the bug.

## Steps/Code to Reproduce

```python
import numpy as np
from sklearn.datasets import make_regression
from sklearn.ensemble import BaggingRegressor, IsolationForest

X, y = make_regression(n_samples=100, random_state=0)

bag = BaggingRegressor(n_estimators=3, warm_start=True, random_state=0).fit(X, y)
samples_before = bag.estimators_samples_
bag.set_params(n_estimators=6).fit(X, y)
print("len(estimators_)        :", len(bag.estimators_))
print("len(estimators_samples_):", len(bag.estimators_samples_))
# The first entry now describes the 4th estimator, not the 1st one:
print("same as before warm start:", np.array_equal(bag.estimators_samples_[0], samples_before[0]))

iso = IsolationForest(n_estimators=3, warm_start=True, random_state=0).fit(X)
iso.set_params(n_estimators=6).fit(X)
print("IsolationForest:", len(iso.estimators_), len(iso.estimators_samples_))
```

## Expected Results

`len(estimators_samples_) == len(estimators_) == 6`, and the first three entries unchanged by the second `fit`. The docstring says it is "the subset of drawn samples for each base estimator".

## Actual Results

```
len(estimators_)        : 6
len(estimators_samples_): 3
same as before warm start: False
IsolationForest: 6 3
```

## Root cause and possible fix

`sklearn/ensemble/_bagging.py:536` overwrites the stored seeds on every call:

```python
seeds = random_state.randint(MAX_INT, size=n_more_estimators)
self._seeds = seeds
```

and `_get_estimators_indices` (`sklearn/ensemble/_bagging.py:580`) loops over `self._seeds`. Fix: on warm start, append the new seeds to the previous ones. Note that `_max_samples`, `_max_features` and `_sample_weight` are also overwritten by each `fit`, so indices of older estimators are only regenerated correctly if those did not change between calls; storing them per estimator (or rejecting a change under warm start) would close that gap too, but is a separate, more exotic case.

## Versions

```
sklearn: 1.10.dev0, commit f570497251 (sklearn/ensemble/_bagging.py identical to upstream/main 289c4d5cfa)
python: 3.13, numpy: 2.5.3, scipy: 1.18.1, platform: linux
```
