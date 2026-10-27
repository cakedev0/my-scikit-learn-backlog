# TheilSenRegressor.fit uses O(max_subpopulation * n_samples) memory to store the random subsets

```json
{
    "estimators": ["TheilSenRegressor"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 2,
    "difficulty": 1,
    "decision": 0,
    "decision_tags": []
}
```

## Classification

- **Priority (2)**: default settings and any dataset large enough to take the random-subset path: peak memory grows as `10_000 * n_samples * 8` bytes (~8 GB at 100k samples) while the data actually needed is ~1.7 MB. TheilSen is a niche estimator, and users who hit it see a MemoryError or swapping rather than wrong results, so not 3.
- **Difficulty (1)**: one `.copy()` (or `np.asarray` stacking) on the line building the subsets.
- **Decision (0)**: same random numbers, same results, strictly less memory.

## Upstream search

Queries: "TheilSenRegressor memory", "TheilSen" (issues and PRs). Nothing about memory or large datasets; the hits are docs/example PRs and the original PR #2949. Two more queries ("TheilSenRegressor MemoryError", "TheilSenRegressor slow large") hit the secondary rate limit. `git log upstream/main -- sklearn/linear_model/_theil_sen.py` shows only maintenance commits.

## Describe the bug and its user-facing impact

`TheilSenRegressor.fit` draws `n_subpopulation_` random subsets (10,000 by default) of `n_features + 1` samples each. Each subset is a 21-element view, but it keeps alive the full permutation of `n_samples` it was sliced from. So the list of subsets uses `max_subpopulation * n_samples * 8` bytes instead of `max_subpopulation * n_subsamples * 8`.

With the default `max_subpopulation=1e4`, the peak is ~78 KiB per training sample: ~620 MiB for 8k samples, ~7.5 GiB for 100k samples, whatever the number of features. Users get a MemoryError (or heavy swapping) on datasets that are tiny in memory.

## Steps/Code to Reproduce

```python
import tracemalloc

from sklearn.datasets import make_regression
from sklearn.linear_model import TheilSenRegressor

for n_samples in [1_000, 2_000, 4_000, 8_000]:
    X, y = make_regression(n_samples=n_samples, n_features=20, random_state=0)
    tracemalloc.start()
    TheilSenRegressor(random_state=0).fit(X, y)
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    print(
        f"n_samples={n_samples:5d}: X is {X.nbytes / 2**20:.2f} MiB, "
        f"fit peak is {peak / 2**20:6.1f} MiB"
    )
```

## Expected Results

The subsets themselves take `10_000 * 21 * 8` bytes (~1.6 MiB), so the peak should stay around 10 MiB and not grow with `n_samples` beyond the size of `X`.

## Actual Results

```
n_samples= 1000: X is 0.15 MiB, fit peak is   86.7 MiB
n_samples= 2000: X is 0.31 MiB, fit peak is  163.0 MiB
n_samples= 4000: X is 0.61 MiB, fit peak is  315.6 MiB
n_samples= 8000: X is 1.22 MiB, fit peak is  620.7 MiB
```

Peak grows linearly with `n_samples` at ~400x the size of `X` (20 features).

## Root cause and possible fix

`sklearn/linear_model/_theil_sen.py:424-427` (commit below):

```python
indices = [
    random_state.choice(n_samples, size=n_subsamples, replace=False)
    for _ in range(self.n_subpopulation_)
]
```

With `replace=False`, the legacy `RandomState.choice` returns `self.permutation(pop_size)[:size]`, which is a view: `random_state.choice(100_000, size=21, replace=False).base.shape == (100_000,)`. The list holds 10,000 such views, so 10,000 full permutations stay alive until `np.array_split(indices, n_jobs)` runs.

Copying each draw fixes it without changing the random stream. I checked it by passing a `RandomState` subclass whose `choice` returns `.copy()`: the peak drops from 620.7 MiB to 10.9 MiB at `n_samples=8000`, with identical `coef_`.

```python
indices = np.asarray([
    random_state.choice(n_samples, size=n_subsamples, replace=False).copy()
    for _ in range(self.n_subpopulation_)
])
```

Generating 10,000 permutations of `n_samples` is also O(max_subpopulation * n_samples) in time. Fixing that (e.g. with `sample_without_replacement`) would change the random stream (`rng-order`, `output-change`), so I'd keep it out of this fix.

## Versions

```
sklearn: 1.10.dev0, commit 289c4d5cfa (upstream/main; local checkout f570497251, no diff on this file)
python: 3.13.14, numpy: 2.5.3, scipy: 1.18.1, platform: linux
```
