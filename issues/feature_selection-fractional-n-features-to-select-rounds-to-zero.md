# `RFE` and `SequentialFeatureSelector` select 0 features when a fractional `n_features_to_select` rounds down to 0

```json
{
    "estimators": ["RFE", "SequentialFeatureSelector"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 1,
    "difficulty": 1,
    "decision": 0,
    "decision_tags": []
}
```

## Classification

- **Priority (1)**: needs `n_features_to_select < 1 / n_features`, e.g. a fraction tuned on wide data reused on narrow data. `RFE` fails loudly but only after running the whole elimination, with an error that does not point to the parameter; `SequentialFeatureSelector` only warns at `transform` and returns an empty array.
- **Difficulty (1)**: `max(1, int(n_features * fraction))` in both places, the same clipping `RFE` already applies to a fractional `step`.
- **Decision (0)**: only changes the case that currently selects nothing.

## Upstream search

Queries (`gh search issues --include-prs`): "RFE n_features_to_select float zero features", "RFE n_features_to_select fraction"; `gh issue list` / `gh pr list --search` "RFE n_features_to_select 0 features", "RFE n_features_to_select float".

- #17090 (merged): added the fraction option to `RFE`, with `int(n_features * fraction)`.
- #10368 (open): fractional `step` in RFECV is computed from the initial number of features, a different parameter.
- #28725 (closed): `n_features_to_select` larger than `n_features`, the opposite edge.

No match.

## Describe the bug and its user-facing impact

Both estimators accept `n_features_to_select` as a float in `(0, 1]` ("the fraction of features to select") and convert it with `int(n_features * fraction)`, which truncates to 0 when the fraction is smaller than `1 / n_features`. The parameter validation accepts these values, so:

- `RFE` eliminates features until none are left (one fit per feature), then crashes when fitting the final estimator on an `(n_samples, 0)` array;
- `SequentialFeatureSelector` sets `n_features_to_select_ = 0`, fits no candidate, and `transform` returns an `(n_samples, 0)` array with a "No features were selected" warning.

Selecting zero features is never what a user asking for a positive fraction wants.

## Steps/Code to Reproduce

```python
from sklearn.datasets import make_classification
from sklearn.feature_selection import RFE, SequentialFeatureSelector
from sklearn.linear_model import LogisticRegression

X, y = make_classification(n_features=5, random_state=0)
est = LogisticRegression()

sfs = SequentialFeatureSelector(est, n_features_to_select=0.1).fit(X, y)
print("SFS selected features:", sfs.n_features_to_select_, sfs.transform(X).shape)

RFE(est, n_features_to_select=0.1).fit(X, y)
```

## Expected Results

At least one feature selected (or an error at the start of `fit` saying that `n_features_to_select` selects no feature).

## Actual Results

```
/home/arthur/scikit-learn/sklearn/feature_selection/_base.py:130: UserWarning: No features were selected: either the data is too noisy or the selection test too strict.
  warnings.warn(
SFS selected features: 0 (100, 0)
Traceback (most recent call last):
  File "rfe_frac.py", line 11, in <module>
    RFE(est, n_features_to_select=0.1).fit(X, y)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^
  File "/home/arthur/scikit-learn/sklearn/base.py", line 1403, in wrapper
    return fit_method(estimator, *args, **kwargs)
  File "/home/arthur/scikit-learn/sklearn/feature_selection/_rfe.py", line 271, in fit
    return self._fit(X, y, **routed_params.estimator.fit)
           ~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/arthur/scikit-learn/sklearn/feature_selection/_rfe.py", line 361, in _fit
    self.estimator_.fit(_safe_indexing(X, features, axis=1), y, **fit_params)
    ~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  ...
ValueError: Found array with 0 feature(s) (shape=(100, 0)) while a minimum of 1 is required by LogisticRegression.
```

## Root cause and possible fix

- `sklearn/feature_selection/_rfe.py:305`: `n_features_to_select = int(n_features * self.n_features_to_select)`.
- `sklearn/feature_selection/_sequential.py:255`: `self.n_features_to_select_ = int(n_features * self.n_features_to_select)`.

Use `max(1, int(...))` in both, as `_rfe.py:308` already does for a fractional `step`.

## Versions

```
sklearn: 1.10.dev0, commit 289c4d5cfa
python: 3.13.14, numpy: 2.5.3, scipy: 1.18.1, platform: linux
```
