# NuSVC's `probability` deprecation warning tells users to switch to `SVC`

```json
{
    "estimators": ["NuSVC"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 1,
    "difficulty": 1,
    "decision": 0,
    "decision_tags": []
}
```

## Classification

- **Priority (1)**: a misleading message, no wrong result. But it is shown to every `NuSVC` user who sets `probability` during the deprecation period, and following it literally (`CalibratedClassifierCV(SVC(), ...)`) silently swaps nu-SVC for C-SVC.
- **Difficulty (1)**: a one-character typo, `"nu_scv"` -> `"nu_svc"`.
- **Decision (0)**: no trade-off.

## Upstream search

Queries: "NuSVC deprecation", "NuSVC probability deprecation message" (PRs), "nu_scv".

No match. The typo comes from #32050 (merged, "Deprecate use of `probability=True` in SVC and NuSVC"); its test only matches `"probability.+parameter.+deprecated"`.

## Describe the bug and its user-facing impact

Since the deprecation of `probability` in 1.9, `NuSVC(probability=True).fit(X, y)` warns:

```
Use `CalibratedClassifierCV(SVC(), ensemble=False)` instead of `SVC(probability=True)`
```

instead of naming `NuSVC`. A user following the advice replaces their nu-SVC by a C-SVC with default `C`, which is a different model.

## Steps/Code to Reproduce

```python
import warnings
from sklearn.datasets import make_classification
from sklearn.svm import NuSVC

X, y = make_classification(random_state=0)
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    NuSVC(probability=True).fit(X, y)
print(w[0].message)
```

## Expected Results

```
... Use `CalibratedClassifierCV(NuSVC(), ensemble=False)` instead of `NuSVC(probability=True)`
```

## Actual Results

```
The `probability` parameter was deprecated in 1.9 and will be removed in version 1.11. Use `CalibratedClassifierCV(SVC(), ensemble=False)` instead of `SVC(probability=True)`
```

## Root cause and possible fix

`sklearn/svm/_base.py:231`:

```python
if self._impl == "nu_scv":
```

`_impl` is `"nu_svc"` for `NuSVC`, so the branch is never taken. Fix the string and extend the test at `sklearn/svm/tests/test_svm.py:1523` to check the estimator name in the message for both classes.

## Versions

```
sklearn: 1.10.dev0, commit 289c4d5cfa (upstream/main; ran on f570497251, no diff in the files involved)
python: 3.13.14, numpy: 2.5.3, scipy: 1.18.1, platform: linux
```
