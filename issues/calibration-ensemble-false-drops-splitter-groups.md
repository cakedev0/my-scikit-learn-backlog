# `CalibratedClassifierCV(ensemble=False)` does not route `groups` to the CV splitter

```json
{
    "estimators": ["CalibratedClassifierCV"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 1,
    "difficulty": 1,
    "decision": 0,
    "decision_tags": []
}
```

## Classification

- **Priority (1)**: loud error, only with metadata routing enabled, `ensemble=False` and a group-aware splitter. Workarounds: `ensemble=True`, or pass the precomputed splits `cv=list(GroupKFold(3).split(X, y, groups))`.
- **Difficulty (1)**: forward the splitter's routed params to `cross_val_predict` (or pass the precomputed splits), plus a `cv_routing_methods` entry in the routing common test.
- **Decision (0)**: makes a documented routing path work; nothing changes for code that works today.

## Upstream search

- `gh search issues --include-prs`: "CalibratedClassifierCV groups ensemble", "CalibratedClassifierCV metadata routing groups", "cross_val_predict groups CalibratedClassifierCV".
- #12052 (closed 2023): "CalibratedClassifierCV doesn't support groups parameter in fit method", closed once metadata routing added splitter routing. That works for `ensemble=True` only.
- `issues/` has no file on this.

No match for this bug.

## Describe the bug and its user-facing impact

With `enable_metadata_routing=True`, `CalibratedClassifierCV.get_metadata_routing()` declares the CV splitter as a consumer of `fit` metadata, so `fit(X, y, groups=groups)` with `cv=GroupKFold(...)` is accepted. `ensemble=True` routes `groups` to `cv.split`. `ensemble=False` builds the out-of-fold predictions with `cross_val_predict(..., cv=cv, params=routed_params.estimator.fit)`, which drops the splitter's params, so `GroupKFold.split` gets `groups=None` and raises.

Users who want group-aware calibration of a single model fitted on all the data (`ensemble=False`, cheaper to fit and predict) hit an error that says `groups` is missing although they passed it.

## Steps/Code to Reproduce

```python
import numpy as np
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

sklearn.set_config(enable_metadata_routing=True)
X, y = make_classification(n_samples=200, random_state=0)
groups = np.arange(200) // 10

CalibratedClassifierCV(LogisticRegression(), cv=GroupKFold(3), ensemble=True).fit(
    X, y, groups=groups
)
print("ensemble=True: ok")
CalibratedClassifierCV(LogisticRegression(), cv=GroupKFold(3), ensemble=False).fit(
    X, y, groups=groups
)
```

## Expected Results

Both fits succeed, and the `ensemble=False` out-of-fold predictions use the group folds.

## Actual Results

```
ensemble=True: ok
Traceback (most recent call last):
  File ".../cal_groups_repro.py", line 16, in <module>
    CalibratedClassifierCV(LogisticRegression(), cv=GroupKFold(3), ensemble=False).fit(
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^
        X, y, groups=groups
        ^^^^^^^^^^^^^^^^^^^
    )
  ...
  File "/home/arthur/scikit-learn/sklearn/calibration.py", line 568, in fit
    predictions = cross_val_predict(
        estimator=this_estimator,
    ...<5 lines>...
        params=routed_params.estimator.fit,
    )
  File "/home/arthur/scikit-learn/sklearn/utils/_param_validation.py", line 218, in wrapper
    return func(*args, **kwargs)
  File "/home/arthur/scikit-learn/sklearn/model_selection/_validation.py", line 1191, in cross_val_predict
    splits = list(cv.split(X, y, **routed_params.splitter.split))
  File "/home/arthur/scikit-learn/sklearn/model_selection/_split.py", line 411, in split
    for train, test in super().split(X, y, groups):
                       ~~~~~~~~~~~~~^^^^^^^^^^^^^^
  File "/home/arthur/scikit-learn/sklearn/model_selection/_split.py", line 142, in split
    for test_index in self._iter_test_masks(X, y, groups):
                      ~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^
  File "/home/arthur/scikit-learn/sklearn/model_selection/_split.py", line 154, in _iter_test_masks
    for test_index in self._iter_test_indices(X, y, groups):
                      ~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^
  File "/home/arthur/scikit-learn/sklearn/model_selection/_split.py", line 607, in _iter_test_indices
    raise ValueError("The 'groups' parameter should not be None.")
ValueError: The 'groups' parameter should not be None.
```

## Root cause and possible fix

`sklearn/calibration.py:568-576` (commit below): the `ensemble=False` branch calls `cross_val_predict(estimator, X, y, cv=cv, ..., params=routed_params.estimator.fit)` and never uses `routed_params.splitter.split`, whereas the `ensemble=True` branch (line 560) calls `cv.split(X, y, **routed_params.splitter.split)`.

Simplest fix: compute the splits once, `splits = list(cv.split(X, y, **routed_params.splitter.split))`, and pass `cv=splits` to `cross_val_predict` (the "at least one sample per class per fold" check above is unaffected). `sklearn/tests/test_metaestimators_metadata_routing.py:112` has no `cv_routing_methods` for `CalibratedClassifierCV`, so `test_metadata_is_routed_correctly_to_splitter` skips it; adding it with `ensemble=False` would cover the fix.

## Versions

```
sklearn: 1.10.dev0, commit 289c4d5cfa
python: 3.13.14, numpy: 2.5.3, scipy: 1.18.1, platform: linux
```
