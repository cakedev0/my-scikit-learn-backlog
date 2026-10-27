# Routed `X_val`/`y_val` are split into CV folds when they happen to have as many rows as `X`

```json
{
    "estimators": ["cross_validate", "GridSearchCV", "RandomizedSearchCV", "HistGradientBoostingRegressor", "HistGradientBoostingClassifier"],
    "existing_issue": "https://github.com/scikit-learn/scikit-learn/issues/34270",
    "reproduced": true,
    "priority": 2,
    "difficulty": 3,
    "decision": 2,
    "decision_tags": ["unclear-spec", "api-change"]
}
```

## Classification

- **Priority (2)**: silent: early stopping of each CV fit uses only the validation rows whose index is in the training fold, and the validation scores look plausible. It only happens when `len(X_val) == len(X)` (e.g. a 50/50 train/validation split), which the user has no reason to think matters. Routing `X_val` through CV tools is new (#34957, auto-requests for the validation set), so few users hit it yet.
- **Difficulty (3)**: the CV tools decide what to split with a length heuristic in `_check_method_params`; fixing it needs per-metadata knowledge of what is sample-aligned, i.e. the design discussed in #34270.
- **Decision (2)**: `unclear-spec` and `api-change`: #34270 lists options (declare properties at request time, globally, or per consumer) with no decision yet.

## Upstream search

#34270 (open RFC, "Declare metadata properties to be used by the routers") describes this exact case as its motivation: "if you pass a `X_val` that has the same size a `X` to a grid search, it will be split". There is no dedicated bug report, and no PR. Searched `gh issue list --search "X_val cross validation sliced"`, `--search "X_val GridSearchCV"`, and the bodies/comments of #35147, #35144, #34957 (validation set routing PRs): no mention.

## Describe the bug and its user-facing impact

With metadata routing enabled, `cross_validate`, `GridSearchCV`, etc. pass the routed fit params of each fold through `_check_method_params(X, params, indices=train)`, which indexes every array-like param whose length equals `len(X)`. `X_val`, `y_val` and `sample_weight_val` are not aligned with `X`, but when they have exactly `len(X)` rows they are subset with the training fold indices. HistGradientBoosting then early-stops on a random half of the validation set. With one row less, the full validation set is used. The refit of a search uses the full `X_val`, so the CV results and the refitted model use different validation data.

## Steps/Code to Reproduce

```python
import numpy as np
import sklearn
from sklearn.datasets import make_regression
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, cross_validate

sklearn.set_config(enable_metadata_routing=True)
X, y = make_regression(n_samples=400, n_features=5, noise=10, random_state=0)
X_train, y_train, X_val, y_val = X[:200], y[:200], X[200:], y[200:]

hgb = HistGradientBoostingRegressor(
    early_stopping=True, max_iter=20, n_iter_no_change=100, random_state=0
).set_fit_request(X_val=True, y_val=True)

for n_val in [200, 199]:  # same number of rows as X_train, or one less
    res = cross_validate(
        hgb, X_train, y_train, cv=KFold(2), return_estimator=True,
        params={"X_val": X_val[:n_val], "y_val": y_val[:n_val]},
    )
    est = res["estimator"][0]
    full = -0.5 * mean_squared_error(y_val[:n_val], est.predict(X_val[:n_val]))
    print(f"n_val={n_val}: validation_score_[-1]={est.validation_score_[-1]:.2f}, "
          f"score on the X_val that was passed={full:.2f}")
```

## Expected Results

In both cases, the last validation score (minus half the MSE for the default loss) is computed on the whole `X_val` that was passed, so it matches the score recomputed on it.

## Actual Results

```
n_val=200: validation_score_[-1]=-974.23, score on the X_val that was passed=-923.62
n_val=199: validation_score_[-1]=-928.26, score on the X_val that was passed=-928.26
```

## Root cause and possible fix

`_fit_and_score` calls `_check_method_params(X, params=fit_params, indices=train)` (`sklearn/model_selection/_validation.py:798`), and `_check_method_params` (`sklearn/utils/validation.py:2311-2325`) indexes any array-like whose `_num_samples` equals that of `X`. The same helper is used by the other CV meta-estimators (`RFECV`, `CalibratedClassifierCV`, `cross_val_predict`, ...).

A targeted fix would hard-code that the validation-set metadata (`X_val`, `y_val`, `sample_weight_val`) is never split, as suggested in the #34270 discussion; the general fix is the metadata properties proposed there.

## Versions

```
sklearn: 1.10.dev0, commit 289c4d5cfa (upstream/main)
python: 3.13.14, numpy: 2.5.3, scipy: 1.18.1, platform: linux
```
