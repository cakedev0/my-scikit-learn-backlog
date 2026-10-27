# `cohen_kappa_score(weights=...)` fails for array API inputs on a non-CPU device

```json
{
    "estimators": ["cohen_kappa_score"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 1,
    "difficulty": 1,
    "decision": 0,
    "decision_tags": []
}
```

## Classification

- **Priority (1)**: `weights="linear"` / `"quadratic"` (the usual setting for ordinal ratings) crashes on any GPU, for a function documented as array API compatible. Loud error, and moving inputs to CPU works around it.
- **Difficulty (1)**: one missing `device=` argument.
- **Decision (0)**: plain fix.

## Upstream search

Grepped all 319 issues/PRs labeled "Array API" for `kappa`: #32619 (merged, adds array API support), #34423 and #33373 (unrelated). #34442 (merged) added mixed namespace/device support for `cohen_kappa_score` but the array API tests call it with default `weights=None`, and CPU-only namespaces don't hit the bug. `gh issue list/pr list --search "cohen_kappa_score weights"` (GraphQL): nothing related (#31172 is about `zero_division`).

## Describe the bug and its user-facing impact

With `array_api_dispatch=True` and inputs on a non-CPU device (torch on CUDA/XPU/MPS, CuPy, or `array_api_strict`'s `device1`), `cohen_kappa_score` with `weights="linear"` or `weights="quadratic"` raises a device mismatch error. `weights=None` works.

## Steps/Code to Reproduce

```python
import array_api_strict as xp
import numpy as np
import sklearn
from sklearn.metrics import cohen_kappa_score

rng = np.random.RandomState(0)
y1, y2 = rng.randint(0, 3, 30), rng.randint(0, 3, 30)
print("NumPy:", cohen_kappa_score(y1, y2, weights="linear"))

# array_api_strict simulates a non-CPU device; torch on a GPU fails the same way
device = xp.Device("device1")
with sklearn.config_context(array_api_dispatch=True):
    print("weights=None:", cohen_kappa_score(xp.asarray(y1, device=device), xp.asarray(y2, device=device)))
    cohen_kappa_score(xp.asarray(y1, device=device), xp.asarray(y2, device=device), weights="linear")
```

## Expected Results

`0.2146596858638744`, as with NumPy.

## Actual Results

```
NumPy: 0.2146596858638744
weights=None: 0.23857868020304562
Traceback (most recent call last):
  ...
  File "sklearn/metrics/_classification.py", line 1014, in cohen_kappa_score
    w_mat += xp.arange(n_classes)
  ...
ValueError: Arrays from two different devices (array_api_strict.Device('device1') and array_api_strict.Device('CPU_DEVICE')) can not be combined.
```

On torch XPU (Intel Arc iGPU) with `weights="quadratic"`:

```
RuntimeError: Expected all tensors to be on the same device, but found at least two devices, xpu:0 and cpu!
```

## Root cause and possible fix

`sklearn/metrics/_classification.py:1014`: `w_mat += xp.arange(n_classes)` creates the range on the default device. Use `xp.arange(n_classes, dtype=max_float_dtype, device=device)`. Adding `weights` to the array API metric checks (with `array_api_strict`'s `device1`) would cover it.

## Versions

```
sklearn: 1.10.dev0, commit f570497251 (metrics/_classification.py identical to upstream/main 289c4d5cfa)
python: 3.13.14, numpy: 2.5.3, scipy: 1.18.1, torch: 2.14.1+xpu, array_api_strict: 2.6.1, platform: linux
```
