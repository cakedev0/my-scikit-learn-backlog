# `IterativeImputer` routes the full `sample_weight` to estimators fitted on the observed rows only

```json
{
    "estimators": ["IterativeImputer"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 1,
    "difficulty": 1,
    "decision": 0,
    "decision_tags": []
}
```

## Classification

- **Priority (1)**: a loud shape-mismatch error, only with metadata routing enabled (opt-in) and per-sample metadata routed through the experimental `IterativeImputer`; the fact that nobody reported it in two years says the combination is barely used. (priority review, 2026-10-23: was 2)
- **Difficulty (1)**: slice the routed params with the same row mask as `X_train`/`y_train` (`_check_method_params(X, params, indices=...)`).
- **Decision (0)**: no trade-off, nothing works today.

## Upstream search

`gh issue list --search "IterativeImputer sample_weight"`, `--search "IterativeImputer routing"`, `gh pr list --search "IterativeImputer metadata routing"`: only #28187 (merged, added the routing) and #28246 (closed, `IterativeImputer` in `ColumnTransformer`, different topic). No report of this bug. The common routing test (`sklearn/tests/test_metaestimators_metadata_routing.py:327`) uses an `X` without missing values, so every row is used and the bug is not triggered.

## Describe the bug and its user-facing impact

With metadata routing enabled, `IterativeImputer.fit(X, sample_weight=sw)` forwards `sw` to the `fit` of each per-feature estimator. But each of these estimators is fitted on the rows where the target feature is observed, `X_train = X_filled[~missing_row_mask]`, while `sw` is passed unchanged with all `n_samples` entries. As soon as one feature to impute has a missing value, the sub-estimator raises a shape mismatch. For metadata that a custom estimator does not validate, it would be silently misaligned with the rows instead.

## Steps/Code to Reproduce

```python
import numpy as np
import sklearn
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge

sklearn.set_config(enable_metadata_routing=True)
rng = np.random.RandomState(0)
X = rng.randn(100, 3)
X[rng.rand(100) < 0.1, 0] = np.nan  # some missing values in the first column
sample_weight = rng.rand(100)

imputer = IterativeImputer(estimator=BayesianRidge().set_fit_request(sample_weight=True))
imputer.fit(X, sample_weight=sample_weight)
```

## Expected Results

The imputer fits, with each per-feature estimator receiving the weights of the rows it is fitted on.

## Actual Results

```
  File "sklearn/impute/_iterative.py", line 820, in fit_transform
    Xt, estimator = self._impute_one_feature(
  ...
  File "sklearn/impute/_iterative.py", line 416, in _impute_one_feature
    estimator.fit(X_train, y_train, **params)
  File "sklearn/base.py", line 1403, in wrapper
    return fit_method(estimator, *args, **kwargs)
  File "sklearn/linear_model/_bayes.py", line 243, in fit
    sample_weight = _check_sample_weight(sample_weight, X, dtype=dtype)
  File "sklearn/utils/validation.py", line 2186, in _check_sample_weight
    raise ValueError(
ValueError: sample_weight.shape == (100,), expected (88,)!
```

## Root cause and possible fix

`IterativeImputer._impute_one_feature` (`sklearn/impute/_iterative.py:404-416`) subsets `X_filled` with `~missing_row_mask` to build `X_train` and `y_train`, then calls `estimator.fit(X_train, y_train, **params)` with the params routed for the whole `X` (`sklearn/impute/_iterative.py:827`). The fix is to pass `_check_method_params(X_filled, params, indices=np.flatnonzero(~missing_row_mask))` instead, as `_fit_and_score` does for CV folds, and to add missing values to the `X` used for `IterativeImputer` in the common routing test.

## Versions

```
sklearn: 1.10.dev0, commit 289c4d5cfa (upstream/main)
python: 3.13.14, numpy: 2.5.3, scipy: 1.18.1, platform: linux
```
