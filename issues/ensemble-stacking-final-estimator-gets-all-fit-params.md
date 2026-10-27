# With metadata routing, `Stacking*.fit` passes all metadata to `final_estimator` regardless of its requests

```json
{
    "estimators": ["StackingClassifier", "StackingRegressor"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 2,
    "difficulty": 2,
    "decision": 1,
    "decision_tags": ["new-error", "output-change"]
}
```

## Classification

- **Priority (2)**: with routing enabled, the final estimator ignores its own `set_fit_request` (silent: `sample_weight=False` is not honored), and any metadata meant for a base estimator only (`groups`, custom metadata) makes `fit` crash. Routing is opt-in, which limits the number of users hit.
- **Difficulty (2)**: add `final_estimator_` with a `fit -> fit` mapping in `get_metadata_routing` and pass the routed params in `fit`; the routing tests need a case for the final estimator.
- **Decision (1)**: `new-error`: after the fix, a final estimator that consumes `sample_weight` but has no request set raises `UnsetMetadataPassedError`, like every other routed consumer (unless auto-requests are enabled). `output-change`: a final estimator with `set_fit_request(sample_weight=False)` stops being weighted.

## Upstream search

`gh issue list --search "Stacking final_estimator metadata routing"`, `--search "StackingClassifier sample_weight final estimator"`, `--search "stacking routing"`, `gh pr list --search "StackingClassifier metadata routing"`, `--search "Stacking routing final_estimator"`. Routing for stacking was added in #28701 (merged), and #35095 (open, auto-requests on routers) does not touch `_stacking.py`. #16537 (closed) is about `sample_weight` in `cross_val_predict`, a different code path. No report of this bug.

## Describe the bug and its user-facing impact

`_BaseStacking.get_metadata_routing` only maps `final_estimator_` for `predict`, not for `fit`. But `_BaseStacking.fit` ends with `_fit_single_estimator(self.final_estimator_, X_meta, y, fit_params=fit_params)` with the raw `**fit_params` the user passed, not routed ones. With `enable_metadata_routing=True`:

1. the final estimator receives `sample_weight` even if it was explicitly told not to (`set_fit_request(sample_weight=False)`), so its fit is silently weighted;
2. metadata requested by a base estimator only (e.g. `groups` for a `GridSearchCV(cv=GroupKFold())` base estimator) is passed to the final estimator's `fit`, which raises a `TypeError`;
3. `get_metadata_routing()` does not describe what `fit` does, so routers wrapping a stacking estimator cannot know that `final_estimator_` consumes metadata.

## Steps/Code to Reproduce

```python
import numpy as np
import sklearn
from sklearn.datasets import make_regression
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import GridSearchCV, GroupKFold, cross_val_predict

sklearn.set_config(enable_metadata_routing=True)
X, y = make_regression(n_samples=100, noise=10, random_state=0)
rng = np.random.RandomState(0)
sample_weight = rng.rand(100)
groups = rng.randint(0, 5, size=100)

# 1. The final estimator explicitly does NOT request sample_weight, but gets it
ridge = Ridge().set_fit_request(sample_weight=True)
final = LinearRegression().set_fit_request(sample_weight=False)
stack = StackingRegressor([("ridge", ridge)], final_estimator=final)
stack.fit(X, y, sample_weight=sample_weight)

X_meta = cross_val_predict(ridge, X, y, params={"sample_weight": sample_weight})[:, None]
print("final_estimator_.coef_:", stack.final_estimator_.coef_)
print("unweighted fit        :", LinearRegression().fit(X_meta, y).coef_)
print("weighted fit          :", LinearRegression().fit(X_meta, y, sample_weight=sample_weight).coef_)

# 2. Metadata only requested by a base estimator is passed to the final estimator
search = GridSearchCV(Ridge(), {"alpha": [0.1, 1.0]}, cv=GroupKFold(3))
StackingRegressor([("search", search)], final_estimator=LinearRegression()).fit(
    X, y, groups=groups
)
```

## Expected Results

1. `final_estimator_.coef_` equals the unweighted fit, since the final estimator does not request `sample_weight`.
2. `groups` is routed to the `GridSearchCV` base estimator only and `fit` succeeds.

## Actual Results

```
final_estimator_.coef_: [1.00758877]
unweighted fit        : [0.98145043]
weighted fit          : [1.00758877]
Traceback (most recent call last):
  ...
  File "sklearn/ensemble/_stacking.py", line 277, in fit
    _fit_single_estimator(self.final_estimator_, X_meta, y, fit_params=fit_params)
  File "sklearn/ensemble/_base.py", line 45, in _fit_single_estimator
    estimator.fit(X, y, **fit_params)
  File "sklearn/base.py", line 1403, in wrapper
    return fit_method(estimator, *args, **kwargs)
TypeError: LinearRegression.fit() got an unexpected keyword argument 'groups'
```

## Root cause and possible fix

`sklearn/ensemble/_stacking.py:277` passes the raw `fit_params` to the final estimator, while `get_metadata_routing` (`sklearn/ensemble/_stacking.py:386-419`) only adds `final_estimator_` with `MethodMapping().add(caller="predict", callee="predict")`. With routing enabled, `_fit_single_estimator` (`sklearn/ensemble/_base.py:43-45`) forwards everything as is.

Fix: add `caller="fit", callee="fit"` to the `final_estimator_` mapping and call `_fit_single_estimator(self.final_estimator_, X_meta, y, fit_params=routed_params.final_estimator_.fit)` when routing is enabled (keeping `fit_params` for the legacy `sample_weight` path). Users who rely on the final estimator getting `sample_weight` will need `final_estimator.set_fit_request(sample_weight=True)`, as for the base estimators.

## Versions

```
sklearn: 1.10.dev0, commit 289c4d5cfa (upstream/main)
python: 3.13.14, numpy: 2.5.3, scipy: 1.18.1, platform: linux
```
