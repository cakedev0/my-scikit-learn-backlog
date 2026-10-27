# OneVsOneClassifier.partial_fit fails with string class labels

```json
{
    "estimators": ["OneVsOneClassifier"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 1,
    "difficulty": 1,
    "decision": 0,
    "decision_tags": []
}
```

## Classification

- **Priority (1)**: loud error, only in `OneVsOneClassifier.partial_fit` (a rarely used combination) with a NumPy string `y`; object dtype or a pandas Series works. The misleading message is the main cost. (priority review, 2026-10-23: was 2)
- **Difficulty (1)**: one line, build the binary target with an integer dtype as `_fit_ovo_binary` does.
- **Decision (0)**: no trade-off.

## Upstream search

- `gh issue list --search "OneVsOneClassifier partial_fit"`: no match (#4167 is the original partial_fit feature request).
- `gh issue list --search "OneVsOneClassifier string"`: #23779 (open) is about custom X input types, not labels.
- `gh search issues "OneVsOneClassifier partial_fit string labels"`: nothing found before the rate limit.

## Describe the bug and its user-facing impact

`OneVsOneClassifier(...).partial_fit(X, y, classes=...)` raises when `y` is a numpy array (or list) of strings, while `fit` works on the same data. The error comes from the base estimator and says `classes should include all valid labels that can be in y`, which sends the user looking for a problem in their `classes` argument.

The binary target for each pair is built with `np.zeros_like(y)`. For a `<U1` array this gives `['', '', ...]`, and `y_binary[y == j] = 1` writes the string `'1'`. The base estimator then receives labels `''`/`'1'` with `classes=np.array((0, 1))`.

## Steps/Code to Reproduce

```python
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.multiclass import OneVsOneClassifier

rng = np.random.RandomState(0)
X = rng.randn(60, 3)
y = np.array(["a", "b", "c"])[rng.randint(0, 3, 60)]

OneVsOneClassifier(SGDClassifier()).fit(X, y)  # works
OneVsOneClassifier(SGDClassifier()).partial_fit(X, y, classes=["a", "b", "c"])
```

## Expected Results

`partial_fit` works like `fit`, and like `OneVsRestClassifier.partial_fit` on the same data.

## Actual Results

```
Traceback (most recent call last):
  File ".../r_ovo.py", line 10, in <module>
    OneVsOneClassifier(SGDClassifier()).partial_fit(X, y, classes=["a", "b", "c"])
  File "/home/arthur/scikit-learn/sklearn/base.py", line 1403, in wrapper
    return fit_method(estimator, *args, **kwargs)
  File "/home/arthur/scikit-learn/sklearn/multiclass.py", line 902, in partial_fit
    self.estimators_ = Parallel(n_jobs=self.n_jobs)(
  ...
  File "/home/arthur/scikit-learn/sklearn/multiclass.py", line 663, in _partial_fit_ovo_binary
    return _partial_fit_binary(
  ...
  File "/home/arthur/scikit-learn/sklearn/utils/class_weight.py", line 80, in compute_class_weight
    raise ValueError("classes should include all valid labels that can be in y")
ValueError: classes should include all valid labels that can be in y
```

Same error with `y` as a Python list of strings. With `y.astype(object)` or a pandas Series it works, because `zeros_like` then gives integer zeros.

## Root cause and possible fix

`sklearn/multiclass.py:658`, in `_partial_fit_ovo_binary`:

```python
y_binary = np.zeros_like(y)
y_binary[y == j] = 1
```

Use `y_binary = np.zeros(y.shape, dtype=int)`, as `_fit_ovo_binary` does at `sklearn/multiclass.py:634`.

## Versions

```
sklearn: 1.10.dev0, commit 289c4d5cfa (upstream/main; reproduced on local checkout f570497251, no diff in sklearn/multiclass.py)
python: 3.13, numpy: 2.5.3, scipy: 1.18.1, platform: linux
```
