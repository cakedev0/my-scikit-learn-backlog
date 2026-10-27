# MiniBatchKMeans(compute_labels=False) crashes when fit runs a single minibatch step

```json
{
    "estimators": ["MiniBatchKMeans"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 1,
    "difficulty": 1,
    "decision": 0,
    "decision_tags": []
}
```

## Classification

- **Priority (1)**: crash with an obscure `TypeError`, but only for `max_iter=1` with `n_samples < 2 * batch_size` and `compute_labels=False`. Easy workaround (`compute_labels=True` or `max_iter >= 2`).
- **Difficulty (1)**: fall back to the first batch inertia (or compute the inertia) when the EWA was never initialized. A couple of lines.
- **Decision (0)**: it currently raises.

## Upstream search

Queries: "MiniBatchKMeans compute_labels", "MiniBatchKMeans compute_labels inertia_". No match (hits are old segfault/memory reports and #17622, the MiniBatchKMeans refactoring that introduced the EWA logic).

## Describe the bug and its user-facing impact

With `compute_labels=False`, `fit` sets `inertia_ = self._ewa_inertia * sum_of_weights`. The EWA of the batch inertia is only initialized from the second step on, since `_mini_batch_convergence` returns early on step 1 ("Ignore first iteration because it's inertia from initialization"). When `fit` runs a single step, `_ewa_inertia` is still `None` and `fit` fails with `TypeError`.

The number of steps is `(max_iter * n_samples) // min(batch_size, n_samples)`, so this happens for `max_iter=1` whenever `n_samples < 2 * batch_size` (with the default `batch_size=1024`: any dataset below 2048 samples).

## Steps/Code to Reproduce

```python
import numpy as np
from sklearn.cluster import MiniBatchKMeans

X = np.random.RandomState(0).rand(100, 2)
MiniBatchKMeans(n_clusters=3, max_iter=1, compute_labels=False, random_state=0).fit(X)
```

## Expected Results

The model is fitted, with `inertia_` set to an estimate of the inertia (as for more steps).

## Actual Results

```
Traceback (most recent call last):
  ...
  File "/home/arthur/scikit-learn/sklearn/cluster/_kmeans.py", line 2202, in fit
    self.inertia_ = self._ewa_inertia * sum_of_weights
                    ~~~~~~~~~~~~~~~~~~^~~~~~~~~~~~~~~~
TypeError: unsupported operand type(s) for *: 'NoneType' and 'float'
```

(line 2199 on `upstream/main`; the local checkout has 3 unrelated extra lines in this file.)

## Root cause and possible fix

`sklearn/cluster/_kmeans.py:2199` (upstream/main) uses `self._ewa_inertia`, which `_mini_batch_convergence` (line 1965, `if step == 1: ... return False`) leaves at `None` after a single step. Fix: in `fit`, when `self._ewa_inertia is None`, use the last `batch_inertia / self._batch_size` instead, or compute the inertia on the data as when `compute_labels=True`.

## Versions

```
sklearn: 1.10.dev0, commit 289c4d5cfa (upstream/main; local checkout f570497251 only changes the Lloyd code in sklearn/cluster/_kmeans.py)
python: 3.13, numpy: 2.5, scipy: 1.18, platform: linux
```
