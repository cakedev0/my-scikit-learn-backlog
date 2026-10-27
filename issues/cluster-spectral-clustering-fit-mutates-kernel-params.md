# SpectralClustering.fit mutates the user's `kernel_params` dict

```json
{
    "estimators": ["SpectralClustering"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 1,
    "difficulty": 1,
    "decision": 0,
    "decision_tags": []
}
```

## Classification

- **Priority (1)**: breaks the "fit does not modify constructor parameters" contract and can make a later fit crash, but only when `kernel_params` is passed together with a string kernel, which the docstring says is ignored anyway.
- **Difficulty (1)**: copy the dict (`params = dict(self.kernel_params or {})`), one line.
- **Decision (0)**: no trade-off.

## Upstream search

Queries: "SpectralClustering kernel_params", "kernel_params modified fit". No match (#1791, #8611 are feature requests, #23851 is parameter validation).

## Describe the bug and its user-facing impact

When `affinity` is a string kernel (e.g. the default `"rbf"`) and `kernel_params` is a dict, `fit` writes `gamma`, `degree` and `coef0` into that dict in place. So `get_params()["kernel_params"]` changes after `fit`, the user's own dict object is modified, and a later `fit` with a callable `affinity` (e.g. after `set_params(affinity=my_kernel)`) passes these unexpected keyword arguments to the callable and crashes.

## Steps/Code to Reproduce

```python
import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.datasets import make_blobs

X, _ = make_blobs(n_samples=30, random_state=0)

def my_kernel(x, y, scale=1.0):
    return np.exp(-scale * np.sum((x - y) ** 2))

sc = SpectralClustering(n_clusters=3, kernel_params={"scale": 0.1}, random_state=0)
print("kernel_params before fit:", sc.get_params()["kernel_params"])
sc.fit(X)  # default affinity="rbf"
print("kernel_params after fit: ", sc.get_params()["kernel_params"])

sc.set_params(affinity=my_kernel).fit(X)
```

## Expected Results

`kernel_params` is left untouched by `fit` (`{'scale': 0.1}`), and the second `fit` calls `my_kernel(x, y, scale=0.1)`.

## Actual Results

```
kernel_params before fit: {'scale': 0.1}
kernel_params after fit:  {'scale': 0.1, 'gamma': 1.0, 'degree': 3, 'coef0': 1}
Traceback (most recent call last):
  ...
TypeError: my_kernel() got an unexpected keyword argument 'gamma'
```

## Root cause and possible fix

`sklearn/cluster/_spectral.py:722-728`:

```python
params = self.kernel_params
if params is None:
    params = {}
if not callable(self.affinity):
    params["gamma"] = self.gamma
    params["degree"] = self.degree
    params["coef0"] = self.coef0
```

`params` aliases `self.kernel_params`. Fix: `params = dict(self.kernel_params or {})`. `KernelPCA` and `KernelRidge` build a fresh dict for string kernels and do not have this problem. `Nystroem._get_kernel_params` (`sklearn/kernel_approximation.py:1095`) uses the same aliasing pattern and mutates `kernel_params` when `gamma`/`degree`/`coef0` are set with a string kernel.

## Versions

```
sklearn: 1.10.dev0, commit 289c4d5cfa (upstream/main; local checkout f570497251 has no change in sklearn/cluster/_spectral.py)
python: 3.13, numpy: 2.5, scipy: 1.18, platform: linux
```
