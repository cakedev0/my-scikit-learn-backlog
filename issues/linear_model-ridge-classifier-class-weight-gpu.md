# `RidgeClassifier` and `RidgeClassifierCV` with `class_weight` fail for array API `y` on a non-CPU device

```json
{
    "estimators": ["RidgeClassifier", "RidgeClassifierCV"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 2,
    "difficulty": 1,
    "decision": 0,
    "decision_tags": []
}
```

## Classification

- **Priority (2)**: `class_weight="balanced"` is a standard setting for imbalanced data, and both estimators are documented as array API compatible. Any GPU `y` makes `fit` crash. Loud, and keeping `y` as a NumPy array works around it.
- **Difficulty (1)**: move `y` to NumPy before calling `compute_sample_weight`, which is a NumPy-only helper.
- **Decision (0)**: plain fix (one small device-to-host copy of `y`, negligible next to the fit).

## Upstream search

Grepped all 319 issues/PRs labeled "Array API" for `class_weight` and `compute_sample_weight`: no match. `gh issue list/pr list --search` (GraphQL) "RidgeClassifier class_weight array API", "compute_sample_weight array API": #34883 (closed, string labels in `compute_class_weight`, unrelated) and #32763 (open, class_weight for naive Bayes, unrelated). On torch CPU the call works because `np.atleast_1d` silently converts the tensor, so CPU-only array API tests can't catch it.

## Describe the bug and its user-facing impact

With `array_api_dispatch=True` and `y` on a non-CPU device (torch on CUDA/XPU/MPS, CuPy, or `array_api_strict`'s `device1`), `RidgeClassifier(class_weight=...)` and `RidgeClassifierCV(class_weight=...)` raise in `fit`. Without `class_weight` the same fit works.

## Steps/Code to Reproduce

```python
import array_api_strict as xp
import sklearn
from sklearn.datasets import make_classification
from sklearn.linear_model import RidgeClassifier, RidgeClassifierCV

X, y = make_classification(n_samples=100, weights=[0.8], random_state=0)

# array_api_strict simulates a non-CPU device; torch on a GPU fails the same way
device = xp.Device("device1")
with sklearn.config_context(array_api_dispatch=True):
    X_xp, y_xp = xp.asarray(X, device=device), xp.asarray(y, device=device)
    print("class_weight=None:", RidgeClassifier(solver="svd").fit(X_xp, y_xp).score(X_xp, y_xp))
    for clf in [
        RidgeClassifier(solver="svd", class_weight="balanced"),
        RidgeClassifierCV(class_weight={0: 1, 1: 4}),
    ]:
        try:
            clf.fit(X_xp, y_xp)
        except Exception as e:
            print(f"{type(clf).__name__}: {type(e).__name__}: {e}")
```

## Expected Results

Both fits succeed, as on CPU.

## Actual Results

```
class_weight=None: 0.94
RidgeClassifier: RuntimeError: Can't convert array on the 'array_api_strict.Device('device1')' device to a NumPy array.
RidgeClassifierCV: RuntimeError: Can't convert array on the 'array_api_strict.Device('device1')' device to a NumPy array.
```

Same script on torch XPU (Intel Arc iGPU):

```
class_weight=None: 0.9399999976158142
RidgeClassifier: TypeError: can't convert xpu:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.
RidgeClassifierCV: TypeError: can't convert xpu:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.
```

## Root cause and possible fix

`_RidgeClassifierMixin._prepare_data` (`sklearn/linear_model/_ridge.py:1317-1319` on upstream/main) calls `compute_sample_weight(self.class_weight, y)` with `y` in its original namespace. `compute_sample_weight` (`sklearn/utils/class_weight.py`) is NumPy-only and starts with `np.atleast_1d(y)`, which fails for device arrays. The result is already moved back with `move_to(reweighting, xp=xp, device=device)`, so only the input side is missing: pass `move_to(y, xp=np, device="cpu")`.

## Versions

```
sklearn: 1.10.dev0, commit f570497251 (the code path is identical on upstream/main 289c4d5cfa)
python: 3.13.14, numpy: 2.5.3, scipy: 1.18.1, torch: 2.14.1+xpu, array_api_strict: 2.6.1, platform: linux
```
