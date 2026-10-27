# Unweighted ROC/PR curves store counts in `y_score.dtype`: float16 scores overflow and give wrong AP (regression on main)

```json
{
    "estimators": ["confusion_matrix_at_thresholds", "precision_recall_curve", "average_precision_score", "roc_curve", "roc_auc_score", "det_curve"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 2,
    "difficulty": 1,
    "decision": 0,
    "decision_tags": []
}
```

## Classification

- **Priority (2)**: unreleased regression from #34817 (merged 2026-08-31):
  with `float16` scores and no `sample_weight`, the TP/FP counts are cast to
  float16, so they are rounded above 2048 and overflow to `inf` above 65504.
  `average_precision_score` then silently drops from 0.753 to 0.653 on
  100k samples. float16 scores are not the common case, but they come
  straight out of mixed-precision deep learning models, and 1.9 handled them
  correctly. It should be fixed before 1.10 is released.
- **Difficulty (1)**: cast the integer counts to
  `_max_precision_float_dtype(xp, device)` (as the weighted path does) instead
  of `y_score.dtype`.
- **Decision (0)**: restores the 1.9 behavior; on float32-only devices
  `_max_precision_float_dtype` is float32 anyway, so the goal of #34817 is
  kept.

## Upstream search

Queries: "float16 roc_curve", "float16 roc_auc_score", "float16
precision_recall_curve", "float16 average_precision_score",
"confusion_matrix_at_thresholds float16", "confusion_matrix_at_thresholds
dtype", "dtype y_score cumulative_sum counts".

- #34817 (merged 2026-08-31): introduced the integer accumulation and the cast
  to `y_score.dtype`. Its discussion is about float32-only devices; float16
  inputs are not mentioned.
- #34813 (open): numerical stability of `confusion_matrix_at_thresholds` on
  float32-only devices, the motivation of #34817; different symptom.
- #34827 (open, ogrisel): stabilizes the weighted branch on float32-only
  devices; it does not change the `output_dtype` line of the unweighted
  branch.

No report of the float16 regression.

## Describe the bug and its user-facing impact

Since #34817, `confusion_matrix_at_thresholds` accumulates unweighted counts
in int64 and then casts them to the dtype of `y_score`
(`output_dtype = y_score.dtype` when it is a real floating dtype). For float16
`y_score`, the counts lose precision above 2048 and become `inf` above 65504.
Every function built on it is affected when `sample_weight=None`:

- `precision_recall_curve` / `average_precision_score`: `tps + fps` overflows
  (`RuntimeWarning: overflow encountered in add`), precision becomes 0 at low
  thresholds and AP is wrong by ~0.1 on 100k samples;
- `roc_curve` / `roc_auc_score` / `det_curve`: rounded counts, AUC off by
  ~3e-4 here, and `nan` rates as soon as one class has more than 65504
  samples (same data with `n = 150_000`: `roc_auc_score` returns `nan`
  instead of 0.7635).

With `sample_weight` the counts are still accumulated in float64, so the
same data gives different results with and without unit weights. Before
#34817 both paths used float64 on NumPy.

On NumPy, float32 scores are also affected in a milder way: counts above
2**24 (~16.7M samples) are now rounded in float32 whereas 1.9 kept them
exact in float64.

## Steps/Code to Reproduce

```python
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, confusion_matrix_at_thresholds

rng = np.random.RandomState(0)
n = 100_000
y_true = rng.randint(0, 2, n)
y_score = (y_true + rng.randn(n)).astype(np.float16)

print("AP  float16:", average_precision_score(y_true, y_score))
print("AP  float64:", average_precision_score(y_true, y_score.astype(np.float64)))
print("AUC float16:", roc_auc_score(y_true, y_score))
print("AUC float64:", roc_auc_score(y_true, y_score.astype(np.float64)))
tns, fps, fns, tps, _ = confusion_matrix_at_thresholds(y_true, y_score)
print("tps.dtype:", tps.dtype, "total positives:", tps[-1], "expected:", y_true.sum())
```

## Expected Results

The same values for the float16 scores and the same scores upcast to float64
(the upcast is exact, so the ranking is identical), and an exact positive
count of 50063.

## Actual Results

```
/home/arthur/scikit-learn/sklearn/metrics/_ranking.py:1189: RuntimeWarning: overflow encountered in add
  ps = tps + fps
AP  float16: 0.6533167865127325
AP  float64: 0.7530069676313381
AUC float16: 0.7600348320481012
AUC float64: 0.7597416145657874
tps.dtype: float16 total positives: 5.005e+04 expected: 50063
```

## Root cause and possible fix

`sklearn/metrics/_ranking.py:1030-1036`:

```python
output_dtype = (
    y_score.dtype
    if hasattr(y_score, "dtype") and xp.isdtype(y_score.dtype, "real floating")
    else _max_precision_float_dtype(xp, device)
)
tps = xp.astype(tps_int, output_dtype)
fps = xp.astype(fps_int, output_dtype)
```

The dtype of the scores says nothing about the range of the counts. Using
`_max_precision_float_dtype(xp, device)` unconditionally fixes float16 and
restores float64 counts on NumPy, while still giving float32 on float32-only
devices (which is what #34817 targeted, see #34813). Open PR #34827 rewrites
the weighted branch only and does not change this line.

## Versions

```
sklearn: 1.10.dev0, commit 289c4d5cfa (upstream/main; sklearn/metrics/ sources identical to the tested checkout)
python: 3.13, numpy: 2.5.3, scipy: 1.18.1, platform: linux
```
