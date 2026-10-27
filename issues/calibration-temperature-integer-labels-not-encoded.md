# `CalibratedClassifierCV(method="temperature")` gives wrong probabilities for integer labels other than `0..n_classes-1`

```json
{
    "estimators": ["CalibratedClassifierCV"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 3,
    "difficulty": 1,
    "decision": 0,
    "decision_tags": []
}
```

## Classification

- **Priority (3)**: silent wrong probabilities for very common label encodings (`-1/1`, `1/2`, `1/2/3`), with both `ensemble=True` and `ensemble=False`. With `1/2` the fitted inverse temperature hits the lower bound `exp(-10)` and every calibrated probability is ~0.5. The labels are also used as column indices in a Cython kernel without bounds checks, so out-of-range labels read outside the logits array.
- **Difficulty (1)**: label-encode `y` for `method="temperature"` whatever its dtype, i.e. drop the `isinstance(y[0], str)` condition.
- **Decision (0)**: results only change for the cases that are currently wrong.

## Upstream search

- `gh search issues --include-prs`: "temperature scaling labels CalibratedClassifierCV", "temperature calibration encoded", `method="temperature"`.
- #35160 "Temperature calibration gives wrong probabilities when labels are not 0..k-1" describes this bug. It was closed on 2026-10-18 with no comment and no linked PR, in the same second as #35155-#35161 from the same account, so it reads as a moderation bulk-close, not a "won't fix". The code on main (289c4d5cfa) is unchanged. `existing_issue` is left `null` because that issue is closed.
- #31068 (merged, 1.8) added temperature scaling and passed the raw `y` to the calibrator. #32246 (merged, 1.8) added label encoding, but only for string `y`.
- #34313 (merged): changes how logits are computed for the calibrators; it does not touch label encoding.

## Describe the bug and its user-facing impact

With `method="temperature"`, `CalibratedClassifierCV.fit` label-encodes `y` only when `y[0]` is a `str`. Other labels go unchanged to `_TemperatureScaling.fit`, which uses them as class indices in `HalfMultinomialLoss` (`raw_prediction[i, int(y_true[i])]`). So:

- labels `1/2` (binary) or `1/2/3` (multiclass): the true class of every sample is shifted by one, the best fit is "no information" and `beta_` goes to the search bound `exp(-10)`: `predict_proba` is uniform;
- labels `-1/1`: label `-1` indexes column -1 with `wraparound=False`, which reads the previous row's last logit; `beta_` is meaningless.
- index `n_classes` (label `2` with 2 classes) reads the next row, or past the end of the buffer for the last sample (undefined behavior).

There is no error or warning, and `predict` (argmax of the calibrated probabilities) may still look fine for multiclass, so users may only see it in the probabilities. String labels and float `0./1.` labels are fine.

## Steps/Code to Reproduce

```python
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

X, y = make_classification(n_samples=600, random_state=0)
for name, labels in [("0/1", [0, 1]), ("'a'/'b'", ["a", "b"]),
                     ("-1/1", [-1, 1]), ("1/2", [1, 2])]:
    y_named = np.asarray(labels)[y]
    clf = CalibratedClassifierCV(LogisticRegression(), method="temperature", cv=3)
    proba = clf.fit(X, y_named).predict_proba(X)
    betas = [float(c.calibrators[0].beta_) for c in clf.calibrated_classifiers_]
    print(f"labels {name:8s} beta_={np.round(betas, 4)}  max proba={proba.max():.3f}")
```

## Expected Results

Renaming the classes should not change the calibration: same `beta_` and same probabilities for all four encodings, as for `method="sigmoid"` / `"isotonic"`.

## Actual Results

```
labels 0/1      beta_=[0.9086 1.0044 1.3509]  max proba=1.000
labels 'a'/'b'  beta_=[0.9086 1.0044 1.3509]  max proba=1.000
labels -1/1     beta_=[0.1815 0.1822 0.1352]  max proba=0.965
labels 1/2      beta_=[0. 0. 0.]  max proba=0.500
```

Same with a prefitted `FrozenEstimator` (`ensemble=False`): `beta_` 0.192 for `-1/1` and 0.0 for `1/2`, vs 1.28 for `0/1`. With 3 classes coded `1/2/3`, `beta_` is ~4.5e-05 in every fold and the log-loss goes from 0.587 to 1.099 (= log 3).

## Root cause and possible fix

`sklearn/calibration.py:483` (commit below):

```python
if self.method == "temperature" and isinstance(y[0], str):
    y = label_encoder_.transform(y=y)
```

`_fit_calibrator` (line 849) then calls `_TemperatureScaling.fit(predictions, y, ...)`, whose `labels = column_or_1d(y, dtype=dtype_)` (line 1205) is used as class indices by the multinomial loss (`sklearn/_loss/_loss.pyx.tp:1271`, compiled with `boundscheck=False, wraparound=False`). The `_fit_calibrator` docstring says that for temperature "`y` needs to be label encoded", but only strings are encoded.

Fix: encode `y` whenever `method == "temperature"` (keeping the array API move). `test_temperature_scaling` only uses `0..n_classes-1` labels and the array API test only strings; a test with labels `1/2/3` and `-1/1` comparing with `0..k-1` would cover it.

## Versions

```
sklearn: 1.10.dev0, commit 289c4d5cfa
python: 3.13.14, numpy: 2.5.3, scipy: 1.18.1, platform: linux
```
