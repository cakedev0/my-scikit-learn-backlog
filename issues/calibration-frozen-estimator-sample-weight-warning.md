# CalibratedClassifierCV(FrozenEstimator(clf)) warns that the calibration "is likely to be incorrect" when sample_weight is passed

```json
{
    "estimators": ["CalibratedClassifierCV", "FrozenEstimator"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 2,
    "difficulty": 1,
    "decision": 0,
    "decision_tags": []
}
```

## Classification

- **Priority (2)**: a false and alarming warning on the workflow the user guide recommends for calibrating a prefitted model (it replaced `cv="prefit"`, which did not warn). The results are right, but users are told they are probably wrong.
- **Difficulty (1)**: skip the check when the estimator is a `FrozenEstimator` (its `fit` is a no-op), as `ensemble="auto"` already special-cases it.
- **Decision (0)**: no trade-off, the warning is wrong for this case.

## Upstream search

- `gh issue list --search "FrozenEstimator sample_weight warning"`: no match.
- `gh issue list --search "does not appear to accept sample_weight"`: no match.
- `gh pr list --search "CalibratedClassifierCV FrozenEstimator sample_weight"`: #30171 (merged, deprecates `cv="prefit"` in favor of `FrozenEstimator`), #30873 (merged, sample_weight dtype upcast), neither touches this warning.
- In 1.5.2, the `cv="prefit"` branch ran before the `supports_sw` check, so the old equivalent workflow did not warn (checked with `git show 1.5.2:sklearn/calibration.py`). This is a regression from the migration to `FrozenEstimator`.

## Describe the bug and its user-facing impact

When metadata routing is disabled (the default), `CalibratedClassifierCV.fit` inspects `signature(estimator.fit)` and, if it has no `sample_weight` parameter, warns:

> Since FrozenEstimator does not appear to accept sample_weight, sample weights will only be used for the calibration itself. [...] Be warned that the result of the calibration is likely to be incorrect.

`FrozenEstimator.fit(self, X, y, *args, **kwargs)` is a no-op, so using the weights only for the calibrator is exactly what the user wants. The warning fires on every fit with `sample_weight` of the documented "calibrate an already fitted classifier" workflow (`doc/modules/calibration.rst`, "Alternatively an already fitted classifier can be calibrated by using a FrozenEstimator").

## Steps/Code to Reproduce

```python
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.datasets import make_classification
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression

X, y = make_classification(random_state=0)
clf = LogisticRegression().fit(X[:50], y[:50])
sample_weight = np.random.RandomState(0).rand(50)
CalibratedClassifierCV(FrozenEstimator(clf)).fit(X[50:], y[50:], sample_weight=sample_weight)
```

## Expected Results

No warning: the frozen classifier is not refitted, and the weights are used for the calibrator, which is correct.

## Actual Results

```
/home/arthur/scikit-learn/sklearn/calibration.py:502: UserWarning: Since FrozenEstimator does not appear to accept sample_weight, sample weights will only be used for the calibration itself. This can be caused by a limitation of the current scikit-learn API. See the following issue for more details: https://github.com/scikit-learn/scikit-learn/issues/21134. Be warned that the result of the calibration is likely to be incorrect.
  warnings.warn(
```

## Root cause and possible fix

`sklearn/calibration.py:498-512` (upstream/main):

```python
fit_parameters = signature(estimator.fit).parameters
supports_sw = "sample_weight" in fit_parameters
if sample_weight is not None and not supports_sw:
    warnings.warn(...)
```

Skip the warning (and the `supports_sw` logic) when `isinstance(estimator, FrozenEstimator)`, as done a few lines above for `ensemble="auto"`. No existing test covers `FrozenEstimator` with `sample_weight` in `sklearn/tests/test_calibration.py`.

## Versions

```
sklearn: 1.10.dev0, commit 289c4d5cfa (upstream/main; reproduced on local checkout f570497251, whose diff in sklearn/calibration.py only touches calibration_curve)
python: 3.13, numpy: 2.5.3, scipy: 1.18.1, platform: linux
```
