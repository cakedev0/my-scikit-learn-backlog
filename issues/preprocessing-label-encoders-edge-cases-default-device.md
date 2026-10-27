# `LabelBinarizer` (single class) and `LabelEncoder` (empty input) return arrays on the default device

```json
{
    "estimators": ["LabelBinarizer", "label_binarize", "LabelEncoder"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 1,
    "difficulty": 1,
    "decision": 0,
    "decision_tags": []
}
```

## Classification

- **Priority (1)**: edge cases (a single class, an empty array), and the values are right. But on a GPU the output silently lands on the CPU, so the next operation with on-device arrays fails with a device mismatch far from the cause.
- **Difficulty (1)**: three `xp.asarray`/`xp.zeros` calls missing `device=`.
- **Decision (0)**: plain fix.

## Upstream search

`gh issue list/pr list --search "LabelBinarizer device"` (GraphQL) and grep of the 319 "Array API" issues/PRs for `LabelBinarizer`, `label_binarize`, `LabelEncoder`: only the PRs adding support (#27381, #28626, #32582) and #34065 (merged, RidgeClassifier `classes_` namespace). Nothing about these edge cases.

## Describe the bug and its user-facing impact

With `array_api_dispatch=True` and labels on a non-CPU device:

- `LabelBinarizer().fit_transform(y)` (and `label_binarize`) returns an array on the default (CPU) device when `y` has a single class, but on `y`'s device otherwise;
- `LabelEncoder.transform` and `LabelEncoder.inverse_transform` return a CPU array for an empty input.

This breaks the documented convention that outputs are on the same device as the inputs.

## Steps/Code to Reproduce

```python
import array_api_strict as xp
import sklearn
from sklearn.preprocessing import LabelBinarizer, LabelEncoder

device = xp.Device("device1")  # simulates a non-CPU device
with sklearn.config_context(array_api_dispatch=True):
    y = xp.asarray([1, 2, 2], device=device)
    print("LabelBinarizer, 2 classes:", LabelBinarizer().fit_transform(y).device)
    y_one = xp.asarray([2, 2, 2], device=device)
    print("LabelBinarizer, 1 class:  ", LabelBinarizer().fit_transform(y_one).device)
    empty = xp.asarray([], dtype=xp.int64, device=device)
    le = LabelEncoder().fit(y)
    for name, out in [("transform", le.transform(empty)), ("inverse_transform", le.inverse_transform(empty))]:
        print(f"LabelEncoder.{name}, empty:", out.device, out.dtype)
```

## Expected Results

All outputs on `device1`.

## Actual Results

```
LabelBinarizer, 2 classes: array_api_strict.Device('device1')
LabelBinarizer, 1 class:   array_api_strict.Device('CPU_DEVICE')
LabelEncoder.transform, empty: array_api_strict.Device('CPU_DEVICE') array_api_strict.float64
LabelEncoder.inverse_transform, empty: array_api_strict.Device('CPU_DEVICE') array_api_strict.float64
```

Torch XPU (Intel Arc iGPU) gives the same: `cpu` tensors for these cases, `xpu:0` otherwise.

## Root cause and possible fix

On upstream/main, `sklearn/preprocessing/_label.py`:

- line 607 (`label_binarize`, single class): `Y = xp.zeros((n_samples, 1), dtype=int_dtype_)` has no `device=device`;
- lines 139 and 161 (`LabelEncoder.transform` / `inverse_transform`): `return xp.asarray([])` has no device (nor dtype).

Add `device=device` (from `get_namespace_and_device(y)`). The float64 dtype of the empty output matches the NumPy behavior, so I'd keep it unless someone wants to change both (an integer dtype for `transform` and `classes_.dtype` for `inverse_transform` would be more consistent).

## Versions

```
sklearn: 1.10.dev0, commit f570497251 (preprocessing/_label.py identical to upstream/main 289c4d5cfa)
python: 3.13.14, numpy: 2.5.3, scipy: 1.18.1, torch: 2.14.1+xpu, array_api_strict: 2.6.1, platform: linux
```
