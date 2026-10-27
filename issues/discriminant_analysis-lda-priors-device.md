# `LinearDiscriminantAnalysis(priors=...)` fails with array API inputs on a non-CPU device

```json
{
    "estimators": ["LinearDiscriminantAnalysis"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 1,
    "difficulty": 1,
    "decision": 0,
    "decision_tags": []
}
```

## Classification

- **Priority (1)**: user-provided `priors` (a constructor array, which the user guide says follows `X`) make `fit` crash on any GPU. Loud error; passing `priors` as an array already on `X`'s device works around it.
- **Difficulty (1)**: one missing `device=` argument.
- **Decision (0)**: plain fix.

## Upstream search

Grepped all 319 issues/PRs labeled "Array API" for `priors` and `LinearDiscriminant`: #34475 (merged, mixed namespace/device support for `X`/`y`), #22554 and #25956 (initial support), nothing about `priors`. The array API tests of LDA don't set `priors`. `gh issue list/pr list --search "LinearDiscriminantAnalysis priors device"` (GraphQL): only #33626 (merged, solver inconsistencies, unrelated).

## Describe the bug and its user-facing impact

With `array_api_dispatch=True` and `X` on a non-CPU device (torch on CUDA/XPU/MPS, CuPy, or `array_api_strict`'s `device1`), `LinearDiscriminantAnalysis(priors=np.array([...])).fit(X, y)` raises a device mismatch error. The default `priors=None` works.

## Steps/Code to Reproduce

```python
import array_api_strict as xp
import numpy as np
import sklearn
from sklearn.datasets import make_classification
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

X, y = make_classification(n_samples=100, random_state=0)
priors = np.array([0.3, 0.7])
print("NumPy:", LinearDiscriminantAnalysis(priors=priors).fit(X, y).score(X, y))

# array_api_strict simulates a non-CPU device; torch on a GPU fails the same way
device = xp.Device("device1")
with sklearn.config_context(array_api_dispatch=True):
    X_xp, y_xp = xp.asarray(X, device=device), xp.asarray(y, device=device)
    print("priors=None:", LinearDiscriminantAnalysis().fit(X_xp, y_xp).score(X_xp, y_xp))
    LinearDiscriminantAnalysis(priors=priors).fit(X_xp, y_xp)
```

## Expected Results

`fit` succeeds, with `priors_` on `X`'s device, as documented in the user guide ("all other array inputs, including constructor arguments, ... will be converted to match the array library and device of `X`").

## Actual Results

```
NumPy: 0.92
priors=None: 0.9
Traceback (most recent call last):
  ...
    self.xbar_ = self.priors_ @ self.means_
  ...
ValueError: Arrays from two different devices (array_api_strict.Device('CPU_DEVICE') and array_api_strict.Device('device1')) can not be combined.
```

On torch XPU (Intel Arc iGPU):

```
RuntimeError: Expected all tensors to be on the same device, but got mat2 is on xpu:0, different from other tensors on cpu (when checking argument in method wrapper_XPU_mm)
```

## Root cause and possible fix

`sklearn/discriminant_analysis.py:693`: `self.priors_ = xp.asarray(self.priors, dtype=X.dtype)` has no `device=`, so a NumPy or list `priors` lands on the default device. Use `move_to(...)` or `xp.asarray(self.priors, dtype=X.dtype, device=array_device(X))`.

Related, two lines below (line 700): renormalizing priors that don't sum to 1 uses the method `self.priors_.sum()`, which is not part of the array API standard (`array_api_strict` raises `AttributeError: 'Array' object has no attribute 'sum'`). `xp.sum(self.priors_)` fixes it.

## Versions

```
sklearn: 1.10.dev0, commit f570497251 (discriminant_analysis.py identical to upstream/main 289c4d5cfa)
python: 3.13.14, numpy: 2.5.3, scipy: 1.18.1, torch: 2.14.1+xpu, array_api_strict: 2.6.1, platform: linux
```
