# RFECV memory grows quadratically with n_features since per-split ranking/support were added to cv_results_

```json
{
    "estimators": ["RFECV"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 2,
    "difficulty": 1,
    "decision": 1,
    "decision_tags": ["api-change"]
}
```

## Classification

- **Priority (2)**: with the default `step=1`, `RFECV.fit` peaks at ~40 bytes x n_features² per CV split and stores ~9 bytes x n_features² per split in `cv_results_`. At 5000 features and the default 5 folds that's ~5 GB of temporary memory and a ~1 GB fitted attribute, for an input that may weigh a few MB. This is a regression from 1.7 (#30179); before that, memory was linear in n_features.
- **Difficulty (1)**: storing NumPy copies instead of Python lists of NumPy scalars (2 lines) removes ~80% of the peak.
- **Decision (1)**: that first fix is free. Getting back to linear memory needs an `api-change`: `split{k}_ranking` and `split{k}_support` (shape `(n_steps, n_features)`) are fully determined by the final ranking of each split and could be stored in that compact form, or computed lazily.

## Upstream search

- `gh search issues ... RFECV memory`, `... RFECV memory step_support`, `... RFECV cv_results_ ranking support memory`, `... RFECV cv_results_`: no issue about memory.
- #30179 (merged, 1.7) added `split{k}_ranking` / `split{k}_support`; its review comments don't mention memory. #17782 (closed) is the feature request behind it.
- #35033 (open issue) and #35174 (open PR) fix the documented shapes of these keys (they omit the feature axis). Related, but about docs only.

## Describe the bug and its user-facing impact

`RFE._fit` with `step_score` (only used by `RFECV`) appends, at each elimination step, `list(support_)` and `list(ranking_)`: Python lists with one NumPy scalar object per feature (~32 bytes each for the int64 ranking, plus an 8-byte pointer). With `step=1` there are ~n_features steps, so each split holds ~n_features² Python objects until `RFECV.fit` converts them with `np.array`. The converted arrays are then stored in `cv_results_` as `split{k}_ranking` (int64) and `split{k}_support` (bool), still n_features² entries per split.

So both the peak memory of `fit` and the size of the fitted estimator (and its pickle) grow as n_features² x n_splits, while X itself is n_samples x n_features. The reproducer below uses 10 samples and a cheap estimator so that the bookkeeping dominates: doubling n_features multiplies the peak and `cv_results_` size by ~4. Extrapolating: 5000 features with `cv=5` gives ~5 GB peak and ~1.1 GB in `cv_results_`.

## Steps/Code to Reproduce

```python
import tracemalloc

import numpy as np
from sklearn.feature_selection import RFECV
from sklearn.linear_model import Ridge

rng = np.random.RandomState(0)
for n_features in [100, 200, 400]:
    X, y = rng.rand(10, n_features), rng.rand(10)
    tracemalloc.start()
    rfecv = RFECV(Ridge(), cv=2).fit(X, y)  # default step=1
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    cv_results_size = sum(v.nbytes for v in rfecv.cv_results_.values())
    print(
        f"n_features={n_features}: X {X.nbytes / 1e3:.0f} kB, "
        f"peak during fit {peak / 1e6:.1f} MB, cv_results_ {cv_results_size / 1e6:.1f} MB"
    )
```

## Expected Results

Memory linear in n_features (times the number of splits), as it was before 1.7: the per-step supports and rankings of a split carry no more information than the final ranking of that split (the support at the step with k remaining features is the set of features with rank 1 after k features remain, and the ranking at each step is the final ranking shifted and clipped at 1).

At the very least, the temporary peak should not be ~5x the size of what is stored.

## Actual Results

```
n_features=100: X 8 kB, peak during fit 1.2 MB, cv_results_ 0.2 MB
n_features=200: X 16 kB, peak during fit 4.1 MB, cv_results_ 0.7 MB
n_features=400: X 32 kB, peak during fit 15.9 MB, cv_results_ 2.9 MB
```

At 800 features I measured 63.2 MB peak and 11.6 MB in `cv_results_` for a 64 kB X.

## Root cause and possible fix

- `sklearn/feature_selection/_rfe.py:338-339` (`RFE._fit`):

  ```python
                self.step_support_.append(list(support_))
                self.step_ranking_.append(list(ranking_))
  ```

  `list()` of a NumPy array creates one NumPy scalar object per element. Replacing these with `support_.copy()` and `ranking_.copy()` keeps the same values at ~9 bytes per entry and makes `np.array(rankings)` / `np.array(supports)` (`_rfe.py:903-904`) a plain stack instead of a slow per-object conversion.
- `sklearn/feature_selection/_rfe.py:936-937`: `cv_results_` stores the full `(n_steps, n_features)` arrays per split. Making this linear needs an API decision: store the final per-split ranking and derive the per-step views (or expose them through a helper), or document the quadratic cost and suggest a larger `step` for many features.

## Versions

```
sklearn: 1.10.dev0, commit 289c4d5cfa
python: 3.13.14, numpy: 2.5.3, scipy: 1.18.1, platform: linux
```
