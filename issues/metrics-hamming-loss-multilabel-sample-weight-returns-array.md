# `hamming_loss` returns a 0-d array instead of a Python float for multilabel inputs with `sample_weight`

```json
{
    "estimators": ["hamming_loss"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 1,
    "difficulty": 1,
    "decision": 0,
    "decision_tags": []
}
```

## Classification

- **Priority (1)**: the value is correct, only the return type breaks the documented convention (a scalar metric returns a Python float). Code doing `json.dump`, f-string formatting or comparisons with NumPy can trip on a 0-d GPU tensor, but it is a narrow case (multilabel and weighted).
- **Difficulty (1)**: wrap the final division in `float(...)`, as the binary/multiclass branch does.
- **Decision (0)**: plain fix.

## Upstream search

`gh pr list --search "hamming_loss array API float"`: #30838 (merged, adds array API support) and #32755 (merged, mixed inputs test). Grepped the 319 "Array API" issues/PRs for `hamming`: also #33086 (merged, unrelated input checks). Nothing about the return type.

## Describe the bug and its user-facing impact

With `array_api_dispatch=True`, `hamming_loss` on multilabel indicator inputs with `sample_weight` returns a 0-d array of the input namespace (a torch tensor, on the GPU if the inputs are) instead of a Python float. Without `sample_weight`, or for binary/multiclass inputs, it returns a float. The user guide says: "When a scoring function returns a scalar value, it will return a Python scalar (typically a `float` instance) instead of an array scalar value."

## Steps/Code to Reproduce

```python
import numpy as np
import torch
import sklearn
from sklearn.metrics import hamming_loss

rng = np.random.RandomState(0)
y_true, y_pred = rng.randint(0, 2, (20, 3)), rng.randint(0, 2, (20, 3))
sw = rng.rand(20)

print("NumPy:", repr(hamming_loss(y_true, y_pred, sample_weight=sw)))
with sklearn.config_context(array_api_dispatch=True):
    y_true_t, y_pred_t, sw_t = torch.asarray(y_true), torch.asarray(y_pred), torch.asarray(sw)
    print("torch:", repr(hamming_loss(y_true_t, y_pred_t)))
    print("torch, sample_weight:", repr(hamming_loss(y_true_t, y_pred_t, sample_weight=sw_t)))
```

## Expected Results

`0.5130312062904324` as a Python float in the last line.

## Actual Results

```
NumPy: np.float64(0.5130312062904324)
torch: 0.55
torch, sample_weight: tensor(0.5130, dtype=torch.float64)
```

## Root cause and possible fix

`sklearn/metrics/_classification.py:3295-3297` (upstream/main): `float(n_differences) / (y_true.shape[0] * y_true.shape[1] * weight_average)`, where `weight_average = _average(sample_weight, xp=xp)` is a 0-d array, so the result is a 0-d array. Wrap the whole expression in `float(...)` (or convert `weight_average` first). Even with NumPy it returns `np.float64` rather than a plain float in this branch, which is harmless since `np.float64` subclasses `float`.

## Versions

```
sklearn: 1.10.dev0, commit f570497251 (metrics/_classification.py identical to upstream/main 289c4d5cfa)
python: 3.13.14, numpy: 2.5.3, scipy: 1.18.1, torch: 2.14.1+xpu, platform: linux
```
