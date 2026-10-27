# `make_classification` raises `AttributeError` when `weights` is a tuple

```json
{
    "estimators": ["make_classification"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 1,
    "difficulty": 1,
    "decision": 0,
    "decision_tags": []
}
```

## Classification

- **Priority (1)**: a crash with an unhelpful error on valid input (`weights` is documented as array-like and validated as `"array-like"`), easy to work around with a list. It is a regression from 1.7.
- **Difficulty (1)**: one line.
- **Decision (0)**: no trade-off.

## Upstream search

Queries: "make_classification weights tuple". No match. The `.copy()` call comes from #30196 (merged, 2025-01, "ENH Add parameter return_X_y to `make_classification`"), which stopped modifying `weights` in place to store it in the returned Bunch; before it, a tuple was used as is.

## Describe the bug and its user-facing impact

`make_classification(weights=(0.3, 0.7))` fails with `AttributeError: 'tuple' object has no attribute 'copy'`. Lists and arrays work, and a tuple of length `n_classes - 1` also works (it goes through `np.resize`).

## Steps/Code to Reproduce

```python
import numpy as np
from sklearn.datasets import make_classification

X, y = make_classification(weights=[0.3, 0.7], flip_y=0, random_state=0)
print(np.bincount(y))
X, y = make_classification(weights=(0.3, 0.7), flip_y=0, random_state=0)
print(np.bincount(y))
```

## Expected Results

```
[30 70]
[30 70]
```

## Actual Results

```
[30 70]
Traceback (most recent call last):
  File "repro_mc1.py", line 6, in <module>
    X, y = make_classification(weights=(0.3, 0.7), flip_y=0, random_state=0)
           ~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/arthur/scikit-learn/sklearn/utils/_param_validation.py", line 218, in wrapper
    return func(*args, **kwargs)
  File "/home/arthur/scikit-learn/sklearn/datasets/_samples_generator.py", line 262, in make_classification
    weights_ = weights.copy()
               ^^^^^^^^^^^^
AttributeError: 'tuple' object has no attribute 'copy'
```

## Root cause and possible fix

`sklearn/datasets/_samples_generator.py:262` (upstream/main 289c4d5cfa): `weights_ = weights.copy()` assumes a list or an ndarray. Use `weights_ = list(weights)` (or `np.asarray(weights, dtype=float).copy()`), which also covers the other branches.

## Versions

```
sklearn: 1.10.dev0, commit 289c4d5cfa (upstream/main; local build f570497251, no diff on sklearn/datasets/_samples_generator.py)
python: 3.13, numpy: 2.5.3, scipy: 1.18.1, platform: linux
```
