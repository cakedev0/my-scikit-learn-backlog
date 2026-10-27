# `MinMaxScaler`/`MaxAbsScaler.partial_fit`: a batch where a feature is all-NaN makes its scale NaN forever

```json
{
    "estimators": ["MinMaxScaler", "MaxAbsScaler"],
    "existing_issue": "https://github.com/scikit-learn/scikit-learn/issues/35143",
    "reproduced": true,
    "priority": 1,
    "difficulty": 1,
    "decision": 0,
    "decision_tags": []
}
```

## Classification

- **Priority (1)**: NaN support in `partial_fit` is documented ("NaNs are treated as missing values: disregarded in fit"), but one batch where a feature is entirely missing makes `transform` return `nan` for that feature for the rest of the stream, with only a numpy `RuntimeWarning`. The trigger is uncommon and maintainers labeled the open upstream issue "Low Priority".
- **Difficulty (1)**: use `fmin`/`fmax` (NaN-ignoring) instead of `minimum`/`maximum` to merge the running statistics.
- **Decision (0)**: a batch without observations of a feature should not change its statistics.

## Upstream search

Queries: "MinMaxScaler partial_fit nan", "partial_fit all nan", "StandardScaler partial_fit nan".

- #35143 (open, "Low Priority", "Needs Investigation"): exactly this bug (all-NaN batch after valid batches). @lesteve sees it as a low-priority variation of #35070. This report adds that the order of the batches does not matter (an all-NaN first batch also poisons the statistics).

## Describe the bug and its user-facing impact

`MinMaxScaler.partial_fit` and `MaxAbsScaler.partial_fit` compute the batch min/max with `nanmin`/`nanmax`, which return `nan` for a feature that is all-NaN in the batch. They then merge it with the running statistics with `minimum`/`maximum`, which propagate NaN. From then on `data_min_`/`data_max_` (resp. `max_abs_`) and `scale_` are `nan` for that feature, whatever the order of the batches, and `transform` outputs `nan`. `fit` on the concatenated data works.

## Steps/Code to Reproduce

```python
import numpy as np
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler

X1 = np.array([[1.0, np.nan], [2.0, np.nan]])  # feature 1 entirely missing in this batch
X2 = np.array([[3.0, 4.0], [4.0, 6.0]])

for Scaler in [MinMaxScaler, MaxAbsScaler]:
    full = Scaler().fit(np.vstack([X1, X2]))
    inc = Scaler().partial_fit(X1).partial_fit(X2)
    print(Scaler.__name__)
    print("  fit        :", full.scale_, full.transform(X2)[:, 1])
    print("  partial_fit:", inc.scale_, inc.transform(X2)[:, 1])
```

## Expected Results

The same `scale_` and output for `fit` and `partial_fit`.

## Actual Results

```
RuntimeWarning: All-NaN slice encountered
  return xp.nanmin(a, axis=axis)
RuntimeWarning: All-NaN slice encountered
  return xp.nanmax(a, axis=axis)
RuntimeWarning: All-NaN slice encountered
  return xp.nanmax(a, axis=axis)
MinMaxScaler
  fit        : [0.33333333 0.5       ] [0. 1.]
  partial_fit: [0.33333333        nan] [nan nan]
MaxAbsScaler
  fit        : [4. 6.] [0.66666667 1.        ]
  partial_fit: [ 4. nan] [nan nan]
```

Swapping the batches gives the same `nan`s. `MaxAbsScaler` on sparse input has the same issue when the feature's stored values in a batch are all NaN.

## Root cause and possible fix

- `MinMaxScaler.partial_fit`, `sklearn/preprocessing/_data.py:519-520`: `xp.minimum(self.data_min_, data_min)` / `xp.maximum(self.data_max_, data_max)`.
- `MaxAbsScaler.partial_fit`, `sklearn/preprocessing/_data.py:1323`: `xp.maximum(self.max_abs_, max_abs)`.

Fix: merge with NaN-ignoring operations (`np.fmin`/`np.fmax`, or `xp.where(isnan(...), ...)` for array API namespaces without `fmin`). A feature that is all-NaN in every batch seen so far still gets `nan` statistics, as with `fit`.

`StandardScaler` has the same symptom with a different cause, see `utils-incremental-mean-var-nan-only-batch.md` (upstream #35070).

## Versions

```
sklearn: 1.10.dev0, commit f570497251 (code identical on upstream/main 289c4d5cfa)
python: 3.13.14, numpy: 2.5.3, scipy: 1.18.1, platform: linux
```
