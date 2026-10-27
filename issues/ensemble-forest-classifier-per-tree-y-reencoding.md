# RandomForestClassifier re-validates and re-encodes the full y for every tree, which dominates fit time with small max_samples

```json
{
    "estimators": ["RandomForestClassifier", "ExtraTreesClassifier"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 1,
    "difficulty": 2,
    "decision": 0,
    "decision_tags": []
}
```

## Classification

- **Priority (1)**: only visible when each tree is cheap compared to n_samples, i.e. a small `max_samples` (or shallow trees) on a large dataset, which is the documented way to speed up forests on big data. Then the classifier is 5-8x slower than the regressor on the same data. With full-size bootstrap samples the overhead is a few percent of fit time.
- **Difficulty (2)**: the forest has to pass its already-encoded target and classes down to `BaseDecisionTree._fit` so the tree can skip `check_classification_targets` and `np.unique`. The change touches the forest/tree private interface, which is shared with other ensembles.
- **Decision (0)**: same trees, same results.

## Upstream search

- `gh search issues ... forest tree np.unique each tree classification overhead`, `... RandomForestClassifier max_samples slow large dataset`, `... check_input tree _fit redundant forest`, `... max_samples forest slow`: no match. #14191 (open) is about `BaggingClassifier` with `LinearSVC`, unrelated.
- Local `issues/` has several `max_samples` reports (bootstrap size, `min_samples_leaf`), none about fit time.

## Describe the bug and its user-facing impact

`BaseForest.fit` validates y and label-encodes it once (`_validate_y_class_weight`). Then each tree's `BaseDecisionTree._fit`, called with `check_input=False`, runs on the full y again: `check_classification_targets(y)` (which calls `type_of_target`), `np.copy(y)`, and `np.unique(y[:, k], return_inverse=True)` for each output. That is O(n_samples log n_samples) per tree, with n_samples the size of the full dataset, regardless of `max_samples`.

With `max_samples=1000` on 1M samples, a profile of `RandomForestClassifier(n_estimators=20).fit` shows ~75% of the time in `np.unique` and `type_of_target` called from `BaseDecisionTree._fit` (~50 ms per tree). `RandomForestRegressor` on the same X and y skips this and is 5-8x faster; both still pay O(n_samples) per tree for the bootstrap counts, which is a smaller constant. For 500 trees this is ~25 s of overhead vs ~3 s for the regressor (loaded laptop, `OMP_NUM_THREADS=2`, `n_jobs=None`). With `n_jobs > 1` the gap shrinks (I measured 2.4x at `n_jobs=4`) because part of the per-tree work runs in parallel threads.

## Steps/Code to Reproduce

```python
from time import perf_counter

import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

rng = np.random.RandomState(0)
for n_samples in [250_000, 500_000, 1_000_000]:
    X = rng.rand(n_samples, 5).astype(np.float32)
    y = (X[:, 0] > 0.5).astype(np.int64)
    times = {}
    for Est in [RandomForestClassifier, RandomForestRegressor]:
        tic = perf_counter()
        Est(n_estimators=20, max_samples=1000, random_state=0).fit(X, y)
        times[Est.__name__] = perf_counter() - tic
    print(f"n_samples={n_samples:>8}: " + ", ".join(f"{k} {v:.2f}s" for k, v in times.items()))
```

## Expected Results

The classifier and the regressor should have similar fit times when each tree sees only 1000 samples: the target is already validated and encoded by the forest, nothing per tree needs to look at all of y beyond drawing the bootstrap sample.

## Actual Results

```
n_samples=  250000: RandomForestClassifier 0.20s, RandomForestRegressor 0.04s
n_samples=  500000: RandomForestClassifier 0.37s, RandomForestRegressor 0.06s
n_samples= 1000000: RandomForestClassifier 1.02s, RandomForestRegressor 0.12s
```

(Two runs gave the same numbers within 0.02 s.)

## Root cause and possible fix

- `sklearn/ensemble/_forest.py:167` / `:176` (`_parallel_build_trees`) call `tree._fit(X, y, ..., check_input=False)` with the forest's encoded float64 y.
- `sklearn/tree/_classes.py:357-368` (`BaseDecisionTree._fit`) runs, whatever `check_input` is:

  ```python
            check_classification_targets(y)
            y = np.copy(y)
            ...
            for k in range(self.n_outputs_):
                classes_k, y_encoded[:, k] = np.unique(y[:, k], return_inverse=True)
  ```

  then converts the result back to a contiguous float64 array.

Possible fix: let ensembles pass the precomputed `classes_` / `n_classes_` (they already pass `categorical_counts` the same way to skip per-tree categorical encoding) and skip the target check and re-encoding when `check_input=False` and the classes are given. `GradientBoosting*` and `Bagging*` use `_fit` with regression trees or `check_input=True`, so they're not affected. Beyond this, the remaining O(n_samples) per tree (bootstrap `bincount` and the splitter scanning all samples for non-zero weights) could be avoided by passing the bootstrap indices, but that's a bigger change.

## Versions

```
sklearn: 1.10.dev0, commit 289c4d5cfa
python: 3.13.14, numpy: 2.5.3, scipy: 1.18.1, platform: linux
```
