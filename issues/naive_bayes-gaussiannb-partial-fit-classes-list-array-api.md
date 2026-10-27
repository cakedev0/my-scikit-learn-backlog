# `GaussianNB.partial_fit` crashes with array API `y` and `classes` passed as a list

```json
{
    "estimators": ["GaussianNB"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 1,
    "difficulty": 2,
    "decision": 0,
    "decision_tags": []
}
```

## Classification

- **Priority (1)**: `partial_fit` requires `classes` on the first call, and every example passes it as a list or `np.unique(y)`. With array API `y` that crashes with a cryptic torch `TypeError`. Loud, and passing `classes` as a tensor works around it.
- **Difficulty (2)**: `classes_` (in the namespace of `classes`) and `y` (in another namespace) are mixed in a few places of `_partial_fit` (`isin`, `searchsorted`, `y == y_i`), and `predict` has the related issue in `naive_bayes-gaussiannb-predict-numpy-y-array-api.md`. Simple in each place, but needs a decision on which namespace `classes_` lives in when `classes` and `y` differ.
- **Decision (0)**: plain fix.

## Upstream search

Grepped all 319 issues/PRs labeled "Array API" for `GaussianNB`, `naive_bayes`, `partial_fit`: only #32497 (merged, adds GaussianNB array API support). `test_gnb_array_api_compliance` only tests `fit`. `gh issue list/pr list --search "partial_fit classes array API"` (GraphQL): nothing related (#30161 refactors `_check_partial_fit_first_call` validation, closed).

## Describe the bug and its user-facing impact

With `array_api_dispatch=True`, `GaussianNB().partial_fit(X, y, classes=[0, 1, 2])` raises a `TypeError` when `y` is a torch tensor. It works with NumPy `y`, and with `classes` given as a tensor.

## Steps/Code to Reproduce

```python
import numpy as np
import torch
import sklearn
from sklearn.datasets import make_classification
from sklearn.naive_bayes import GaussianNB

X, y = make_classification(n_samples=200, n_classes=3, n_informative=4, random_state=0)
print("NumPy:", GaussianNB().partial_fit(X, y, classes=[0, 1, 2]).score(X, y))

with sklearn.config_context(array_api_dispatch=True):
    X_t, y_t = torch.asarray(X), torch.asarray(y)
    clf = GaussianNB().partial_fit(X_t, y_t, classes=torch.asarray([0, 1, 2]))
    print("torch, classes=tensor:", clf.score(X_t, y_t))
    GaussianNB().partial_fit(X_t, y_t, classes=[0, 1, 2])
```

## Expected Results

The same model as with `classes=torch.asarray([0, 1, 2])`, no error.

## Actual Results

```
NumPy: 0.73
torch, classes=tensor: 0.7300000190734863
Traceback (most recent call last):
  ...
  File "sklearn/naive_bayes.py", line 486, in _partial_fit
    unique_y_in_classes = xpx.isin(unique_y, classes, xp=xp_y)
  File "sklearn/externals/array_api_extra/_set.py", line 68, in isin
    return xp.isin(a, b, assume_unique=assume_unique, invert=invert)
TypeError: isin() received an invalid combination of arguments - got (Tensor, numpy.ndarray, invert=bool, assume_unique=bool), but expected one of:
 ...
```

## Root cause and possible fix

`_check_partial_fit_first_call` sets `self.classes_ = unique_labels(classes)`, a NumPy array for a list. `_partial_fit` (`sklearn/naive_bayes.py:483-497`) then mixes it with `y` in `y`'s namespace (`xp_y`): `xpx.isin(unique_y, classes, xp=xp_y)`, then `xp_y.searchsorted(classes, y_i)`.

Possible fix: when `y` is not NumPy, move `classes_` to `y`'s namespace and device once in `_partial_fit` (as `fit` does implicitly through `unique_labels(y)`), or do the class lookup in `classes_`'s namespace. Unless `classes_` is moved, `predict` hits the issue described in `issues/naive_bayes-gaussiannb-predict-numpy-y-array-api.md` on GPU afterwards.

## Versions

```
sklearn: 1.10.dev0, commit f570497251 (naive_bayes.py identical to upstream/main 289c4d5cfa)
python: 3.13.14, numpy: 2.5.3, scipy: 1.18.1, torch: 2.14.1+xpu, platform: linux
```
