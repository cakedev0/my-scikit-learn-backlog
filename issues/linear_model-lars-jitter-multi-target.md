# Lars/LassoLars with `jitter` fail on multi-target `y`

```json
{
    "estimators": ["Lars", "LassoLars"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 1,
    "difficulty": 1,
    "decision": 0,
    "decision_tags": []
}
```

## Classification

- **Priority (1)**: crash on valid input with a raw numpy broadcasting error, but `jitter` is a rarely used option.
- **Difficulty (1)**: one line, draw the noise with `size=y.shape`.
- **Decision (0)**: no trade-off (single-target results and random draws are unchanged).

## Upstream search

Queries (issues and PRs): "Lars jitter", "Lars jitter multi-target".

- #2746 (closed) and #15179 (merged PR): introduced `jitter`; the tests only cover a 1D `y`. No report of the multi-target failure.

## Describe the bug and its user-facing impact

`Lars` and `LassoLars` support a 2D `y` (one path per target), and `jitter` adds uniform noise to `y` to break ties. But the noise is drawn with `size=len(y)`, i.e. shape `(n_samples,)`, and added to `y` of shape `(n_samples, n_targets)`, which numpy broadcasts along the last axis. The fit fails with a broadcasting error, unless `n_targets == n_samples`, in which case the noise is silently drawn per target instead of per sample.

## Steps/Code to Reproduce

```python
import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import Lars

X, Y = make_regression(n_samples=30, n_features=4, n_targets=2, random_state=0)
print(Lars().fit(X, Y).coef_.shape)  # multi-target is supported
Lars(jitter=1e-3, random_state=0).fit(X, Y)
```

## Expected Results

The fit succeeds, with independent noise added to each entry of `Y`.

## Actual Results

```
(2, 4)
Traceback (most recent call last):
  File ".../repro_jitter.py", line 7, in <module>
    Lars(jitter=1e-3, random_state=0).fit(X, Y)
  File "/home/arthur/scikit-learn/sklearn/base.py", line 1403, in wrapper
    return fit_method(estimator, *args, **kwargs)
  File "/home/arthur/scikit-learn/sklearn/linear_model/_least_angle.py", line 1193, in fit
    y = y + noise
ValueError: operands could not be broadcast together with shapes (30,2) (30,) 
```

## Root cause and possible fix

`sklearn/linear_model/_least_angle.py:1192` (upstream/main): `noise = rng.uniform(high=self.jitter, size=len(y))`. Use `size=y.shape` instead; for a 1D `y` this draws exactly the same numbers as today.

## Versions

```
sklearn: 1.10.dev0, commit 289c4d5cfa (upstream/main; reproduced on a local build whose _least_angle.py is identical)
python: 3.13, numpy: 2.5.3, scipy: 1.18.1, platform: linux
```
