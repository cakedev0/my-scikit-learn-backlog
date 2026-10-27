# `cross_val_predict` crashes with array API `X` and a NumPy string `y`

```json
{
    "estimators": ["cross_val_predict"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 1,
    "difficulty": 2,
    "decision": 0,
    "decision_tags": []
}
```

## Classification

- **Priority (1)**: loud error on a niche combination: array API dispatch with a non-NumPy `X`, a NumPy string `y`, and `cross_val_predict` with the default `method="predict"` (`fit`, the other CV tools and `method="predict_proba"` work). Encoding `y` by hand is a workaround. (priority review, 2026-10-23: was 2)
- **Difficulty (2)**: `y` must stay in its own namespace when it can't be moved, and the per-fold predictions (string NumPy arrays) must then be concatenated in their own namespace, not `X`'s.
- **Decision (0)**: plain fix.

## Upstream search

`gh issue list/pr list --search` (GraphQL): "cross_val_predict array API string", "cross_val_predict torch". Related:

- #33633 (merged): stopped forcing `y` into `X`'s namespace in `GridSearchCV`, `cross_validate` and `cross_val_score` so that a NumPy string `y` works. `cross_val_predict` was not part of it and still calls `move_to(y, ...)` unconditionally.
- #34142 (merged): common estimator test for mixed string `y` and array API `X` (estimators only, not `cross_val_predict`).
- #32270 (merged): array API support for `cross_val_predict`.
- `issues/model_selection-cross-val-predict-fits-on-encoded-labels.md`: different bug in the label encoding of `method="predict_proba"`.

## Describe the bug and its user-facing impact

With `array_api_dispatch=True`, `X` a torch tensor and `y` a NumPy array of strings, `cross_val_predict(clf, X, y)` raises a `TypeError` while trying to move `y` to torch. The same inputs work with `clf.fit(X, y).predict(X)`, `cross_val_score`, and with `method="predict_proba"` (where `y` is label-encoded first).

## Steps/Code to Reproduce

```python
import numpy as np
import torch
import sklearn
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, cross_val_score

X, y = make_classification(n_samples=40, random_state=0)
y = np.array(["neg", "pos"])[y]

with sklearn.config_context(array_api_dispatch=True):
    X_t = torch.asarray(X)
    print("fit/predict:", LogisticRegression().fit(X_t, y).predict(X_t)[:3])
    print("cross_val_score:", cross_val_score(LogisticRegression(), X_t, y, cv=2))
    print("predict_proba:", cross_val_predict(LogisticRegression(), X_t, y, cv=2, method="predict_proba").shape)
    cross_val_predict(LogisticRegression(), X_t, y, cv=2)
```

## Expected Results

A NumPy array of string predictions, as returned by `LogisticRegression.predict` for this input.

## Actual Results

```
fit/predict: ['pos' 'pos' 'neg']
cross_val_score: [0.95 0.9 ]
predict_proba: torch.Size([40, 2])
Traceback (most recent call last):
  ...
BufferError: DLPack only supports signed/unsigned integers, float and complex dtypes (or dtypes registered by third-party packages).

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  ...
  File "sklearn/model_selection/_validation.py", line 1216, in cross_val_predict
    y = move_to(y, xp=xp, device=device)
  File "sklearn/utils/_array_api.py", line 640, in move_to
    array_converted = xp.asarray(array, device=device)
  ...
TypeError: can't convert np.ndarray of type numpy.str_. The only supported types are: float64, float32, float16, complex64, complex128, int64, int32, int16, int8, uint64, uint32, uint16, uint8, and bool.
```

## Root cause and possible fix

`sklearn/model_selection/_validation.py:1216` (upstream/main) unconditionally does `y = move_to(y, xp=xp, device=device)`. Following #33633, `y` can be left as is and the wrapped estimator handles it (array API estimators are expected to accept a mixed string `y`, see the common test from #34142). Removing that line is not enough though: the per-fold predictions are then NumPy string arrays, and line 1251 concatenates them with `xp.concat` and indexes with `inv_test_indices` created on `X`'s device. The concatenation and reordering should use the namespace and device of the predictions (`get_namespace_and_device(predictions[0])`).

## Versions

```
sklearn: 1.10.dev0, commit f570497251 (cross_val_predict identical to upstream/main 289c4d5cfa)
python: 3.13.14, numpy: 2.5.3, scipy: 1.18.1, torch: 2.14.1+xpu, platform: linux
```
