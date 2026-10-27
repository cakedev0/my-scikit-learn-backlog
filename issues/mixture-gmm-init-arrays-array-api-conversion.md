# `GaussianMixture` does not convert `weights_init`, `means_init`, `precisions_init` to the namespace, device and dtype of `X`

```json
{
    "estimators": ["GaussianMixture"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 1,
    "difficulty": 1,
    "decision": 0,
    "decision_tags": []
}
```

## Classification

- **Priority (1)**: loud errors, only with array API dispatch combined with user-provided init arrays (both minority uses), and converting the init arrays to `X`'s namespace/device/dtype by hand is a simple workaround. (priority review, 2026-10-23: was 2)
- **Difficulty (1)**: convert the three arrays with `move_to` and cast to `X.dtype` in `_check_parameters`, instead of calling `check_array` with namespace dtypes on a foreign array.
- **Decision (0)**: plain fix.

## Upstream search

Grepped all 319 issues/PRs labeled "Array API" for `GaussianMixture`, `weights_init`, `means_init`: #30777 (merged, adds support), #31639, #31607, #31784 (merged, unrelated fixes/tags), #33945 (docs). `test_gaussian_mixture_array_api_compliance` passes the init arrays only after converting them to `X`'s namespace, device and dtype (`xp.asarray(param_value, device=device)` with `astype(dtype_name)`), so the mismatched cases are not tested. `gh issue list/pr list --search` (GraphQL) "GaussianMixture means_init array API", "weights_init torch": #30777 and #31452 (docs on mixed inputs), nothing about init arrays.

## Describe the bug and its user-facing impact

With `array_api_dispatch=True`, `GaussianMixture(init_params="random", means_init=...)` (same for `weights_init` and `precisions_init`) fails in `fit` when the init array is not already in the namespace, device and dtype of `X`:

- NumPy init array, torch `X`: `TypeError: Cannot interpret 'torch.float64' as a data type`;
- float64 tensor, float32 `X`: `RuntimeError: expected m1 and m2 to have the same dtype`;
- CPU tensor, `X` on GPU: `ValueError: Input arrays use different devices: xpu:0, cpu`.

The user guide says that "all other array inputs, including constructor arguments, ... will be converted to match the array library and device of `X`". With NumPy inputs, float32 `X` and float64 init arrays work fine.

## Steps/Code to Reproduce

```python
import numpy as np
import torch
import sklearn
from sklearn.mixture import GaussianMixture

rng = np.random.RandomState(0)
X = np.r_[rng.randn(50, 2), rng.randn(50, 2) + 5]
means_init = np.array([[0.0, 0.0], [5.0, 5.0]])

gm = GaussianMixture(2, init_params="random", means_init=means_init, random_state=0)
print("NumPy:", gm.fit(X).means_.round(2).tolist())

with sklearn.config_context(array_api_dispatch=True):
    for name, X_t, m in [
        ("X float64, means_init NumPy", torch.asarray(X), means_init),
        ("X float32, means_init torch float64", torch.asarray(X, dtype=torch.float32), torch.asarray(means_init)),
        ("X float64, means_init torch float64", torch.asarray(X), torch.asarray(means_init)),
    ]:
        gm = GaussianMixture(2, init_params="random", means_init=m, random_state=0)
        try:
            print(f"{name}:", gm.fit(X_t).means_.round(decimals=2).tolist())
        except Exception as e:
            print(f"{name}: {type(e).__name__}: {e}")
```

## Expected Results

All three fits succeed and give the NumPy result, `[[0.02, 0.1], [4.98, 5.18]]`.

## Actual Results

```
NumPy: [[0.02, 0.1], [4.98, 5.18]]
X float64, means_init NumPy: TypeError: Cannot interpret 'torch.float64' as a data type
X float32, means_init torch float64: RuntimeError: expected m1 and m2 to have the same dtype, but got: double != float
X float64, means_init torch float64: [[0.02, 0.1], [4.98, 5.18]]
```

With `X` on torch XPU (Intel Arc iGPU) and `means_init` a float64 CPU tensor:

```
ValueError Input arrays use different devices: xpu:0, cpu
```

## Root cause and possible fix

`GaussianMixture._check_parameters` (`sklearn/mixture/_gaussian_mixture.py:801-822`) passes `X`'s namespace to `_check_weights`, `_check_means` and `_check_precisions`, which call `check_array(arr, dtype=[xp.float64, xp.float32])` (lines 42, 82, 139). For a NumPy `arr`, `check_array` uses NumPy and fails on `np.dtype(torch.float64)`. For a tensor, it keeps the tensor's own dtype and device, so later operations with `X` mix float64/float32 or CPU/GPU.

Fix: in `_check_parameters`, `move_to(arr, xp=xp, device=array_device(X))` and cast to `X.dtype` before (or instead of) the `check_array` call.

## Versions

```
sklearn: 1.10.dev0, commit f570497251 (mixture/ identical to upstream/main 289c4d5cfa)
python: 3.13.14, numpy: 2.5.3, scipy: 1.18.1, torch: 2.14.1+xpu, platform: linux
```
