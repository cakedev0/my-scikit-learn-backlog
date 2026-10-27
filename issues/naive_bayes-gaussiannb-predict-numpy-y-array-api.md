# `GaussianNB.predict` and `CalibratedClassifierCV.predict` fail on GPU when fitted with array API `X` and a NumPy integer `y`

```json
{
    "estimators": ["GaussianNB", "CalibratedClassifierCV"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 2,
    "difficulty": 1,
    "decision": 0,
    "decision_tags": []
}
```

## Classification

- **Priority (2)**: the user guide explicitly promotes this mixed case ("everything follows X", `y` left on CPU in a pipeline that moves `X` to GPU). `predict` crashes on any GPU, and on CPU it silently returns NumPy instead of the input namespace. Workaround: move `y` to the same namespace/device before `fit`.
- **Difficulty (1)**: the same few lines in two `predict` methods; `LinearClassifierMixin.predict` already shows the right pattern.
- **Decision (0)**: plain fix (on CPU, `predict` would return a tensor instead of a NumPy array, which is what the documented convention says).

## Upstream search

Grepped all 319 issues/PRs labeled "Array API" for `GaussianNB`, `naive_bayes`, `Calibrat`: #32497 (merged, adds GaussianNB support), #32246 (merged, temperature scaling) and #31869 (open, tracking issue for CalibratedClassifierCV), none about mixed `y`. `test_gnb_array_api_compliance` covers string `y` in NumPy and numeric `y` in the array namespace, not numeric `y` in NumPy. The common check that would catch it, `check_array_api_mixed_inputs`, is xfailed for `GaussianNB` in `sklearn/utils/_test_common/instance_generator.py` with another reason ("PyTorch bug when asarray used on array-api-strict boolean array"), which hides this failure.

`gh issue list --search` (GraphQL): "GaussianNB predict device", "everything follows X classes_ numpy predict". Related: #34047 (closed by #34065) settled that `classes_` follows `y` and predictions follow `X` (as in `LogisticRegression`); #34604 / #35053 (open RFC and PR) propose storing all fitted attributes as NumPy, which would change how this is fixed but is not merged.

## Describe the bug and its user-facing impact

With `array_api_dispatch=True`, fitting `GaussianNB` or `CalibratedClassifierCV(method="temperature")` with `X` on a GPU and `y` a NumPy array of integer labels works, but `predict` then fails:

- on a non-CPU device it raises (torch: `can't convert xpu:0 device type tensor to numpy`);
- on torch CPU it returns a NumPy array instead of a tensor.

`LogisticRegression` (also used as the base estimator below) handles the same inputs correctly.

## Steps/Code to Reproduce

```python
import array_api_strict as xp
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

X, y = make_classification(n_samples=200, n_classes=3, n_informative=4, random_state=0)

# array_api_strict simulates a non-CPU device; torch on a GPU fails the same way
X_xp = xp.asarray(X, device=xp.Device("device1"))
with sklearn.config_context(array_api_dispatch=True):
    for clf in [
        LogisticRegression(),
        GaussianNB(),
        CalibratedClassifierCV(LogisticRegression(), method="temperature"),
    ]:
        clf.fit(X_xp, y)  # y stays a NumPy array of integer labels
        try:
            pred = clf.predict(X_xp)
            print(f"{type(clf).__name__}: {type(pred).__module__}, {pred.device}")
        except Exception as e:
            print(f"{type(clf).__name__}: {type(e).__name__}: {e}")
```

## Expected Results

All three return an array of `X`'s namespace on `X`'s device, like `LogisticRegression`.

## Actual Results

```
LogisticRegression: array_api_strict._array_object, array_api_strict.Device('device1')
GaussianNB: RuntimeError: Can't convert array on the 'array_api_strict.Device('device1')' device to a NumPy array.
CalibratedClassifierCV: RuntimeError: Can't convert array on the 'array_api_strict.Device('device1')' device to a NumPy array.
```

Same script with torch, on CPU and on XPU (Intel Arc iGPU):

```
cpu LogisticRegression -> Tensor cpu
cpu GaussianNB -> ndarray cpu
cpu CalibratedClassifierCV -> ndarray cpu
xpu LogisticRegression -> Tensor xpu:0
xpu GaussianNB -> TypeError can't convert xpu:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.
xpu CalibratedClassifierCV -> TypeError can't convert xpu:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.
```

## Root cause and possible fix

`_BaseNB.predict` (`sklearn/naive_bayes.py:113-116`) and `CalibratedClassifierCV.predict` (`sklearn/calibration.py:670-674` on upstream/main) do:

```python
pred_indices = xp.argmax(jll, axis=1)
if isinstance(self.classes_[0], str):
    pred_indices = move_to(pred_indices, xp=np, device="cpu")
return self.classes_[pred_indices]
```

`classes_` stays in `y`'s namespace (NumPy) as documented, but the indices are only moved to NumPy for string labels. For numeric NumPy labels, NumPy indexing with a device tensor fails (GPU) or converts implicitly and returns NumPy (CPU).

`LinearClassifierMixin.predict` (`sklearn/linear_model/_base.py:411-424`) does it right: move the indices to `classes_`'s namespace/device, `take`, then move the result back to `X`'s namespace/device unless the labels are strings. Reusing that logic (ideally as a shared helper) fixes both, and a "numeric NumPy `y`" case in the GaussianNB and calibration array API tests would cover it.

## Versions

```
sklearn: 1.10.dev0, commit f570497251 (naive_bayes.py identical to upstream/main 289c4d5cfa; calibration.py differs locally but CalibratedClassifierCV.predict is identical)
python: 3.13.14, numpy: 2.5.3, scipy: 1.18.1, torch: 2.14.1+xpu, array_api_strict: 2.6.1, platform: linux
```
