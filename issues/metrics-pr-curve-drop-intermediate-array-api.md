# `precision_recall_curve` and `det_curve` crash with array API inputs when `drop_intermediate=True`

```json
{
    "estimators": ["precision_recall_curve", "det_curve"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 1,
    "difficulty": 1,
    "decision": 0,
    "decision_tags": []
}
```

## Classification

- **Priority (1)**: loud error, only with array API dispatch on non-NumPy inputs and the non-default `drop_intermediate=True`; the default works. (priority review, 2026-10-23: was 2)
- **Difficulty (1)**: `roc_curve` already has the array API version of the same code (`xp.nonzero` and bool casts), it can be copied.
- **Decision (0)**: plain fix.

## Upstream search

Grepped all 319 issues/PRs labeled "Array API" for `drop_intermediate`, `precision_recall_curve`, `det_curve`: only the PRs adding support (#32249, #32586) and #34442 (which fixed `roc_curve` among others). None mentions `drop_intermediate`. The array API checks in `sklearn/metrics/tests/test_common.py` call the metrics with default parameters only, so `drop_intermediate=True` is not tested. `gh issue list --search "drop_intermediate array"` (GraphQL): #31635 (open) is about `roc_curve`'s dropping logic, unrelated.

## Describe the bug and its user-facing impact

With `array_api_dispatch=True` and torch (or any non-NumPy) inputs, `precision_recall_curve(..., drop_intermediate=True)` and `det_curve(..., drop_intermediate=True)` raise a `TypeError`. NumPy inputs work.

## Steps/Code to Reproduce

```python
import numpy as np
import torch
import sklearn
from sklearn.metrics import det_curve, precision_recall_curve

rng = np.random.RandomState(0)
y_true = rng.randint(0, 2, 20)
y_score = rng.rand(20)

print("NumPy:", precision_recall_curve(y_true, y_score, drop_intermediate=True)[2].shape)
with sklearn.config_context(array_api_dispatch=True):
    for f in [precision_recall_curve, det_curve]:
        try:
            f(torch.asarray(y_true), torch.asarray(y_score), drop_intermediate=True)
        except TypeError as e:
            print(f"torch, {f.__name__}:", type(e).__name__, e)
```

## Expected Results

The same curves as with NumPy inputs, as torch tensors.

## Actual Results

```
NumPy: (16,)
torch, precision_recall_curve: TypeError where() missing 2 required positional arguments: 'x1' and 'x2'
torch, det_curve: TypeError where() missing 2 required positional arguments: 'x1' and 'x2'
```

With `array_api_strict` inputs the errors are `Only boolean dtypes are allowed in logical_or(...)` (`precision_recall_curve`) and `object of type 'Array' has no len()` (`det_curve`).

## Root cause and possible fix

`sklearn/metrics/_ranking.py:1176` (`precision_recall_curve`) and `sklearn/metrics/_ranking.py:421` (`det_curve`) use the one-argument `xp.where(cond)`, which is not part of the array API standard, and call `xp.logical_or` on float arrays (`xp.diff(tps)`). `det_curve` also uses `len(fps)` at line 414.

`roc_curve` (line 1334) has a compliant version of the same pattern: `xp.nonzero(...)[0]` with `xp.astype(..., xp.bool)` before `logical_or`, and `fps.shape[0]` instead of `len`. Applying the same to the two other functions fixes it, plus a `drop_intermediate=True` case in the array API tests.

## Versions

```
sklearn: 1.10.dev0, commit f570497251 (metrics/_ranking.py identical to upstream/main 289c4d5cfa)
python: 3.13.14, numpy: 2.5.3, scipy: 1.18.1, torch: 2.14.1+xpu, array_api_strict: 2.6.1, platform: linux
```
