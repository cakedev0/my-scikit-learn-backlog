# `label_ranking_average_precision_score` is ~35x slower than needed (per-sample `rankdata` calls)

```json
{
    "estimators": ["label_ranking_average_precision_score", "label_ranking_loss"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 1,
    "difficulty": 2,
    "decision": 0,
    "decision_tags": []
}
```

## Classification

- **Priority (1)**: a niche multilabel metric, but it costs ~120 µs per sample (~2 min for 1M samples), where a row-wise version gives the same value ~35x faster. No workaround from the public API.
- **Difficulty (2)**: replace the loop with two calls to `scipy.stats.rankdata(..., axis=1)` and masked sums; the formula needs some care with ties and with the degenerate rows (no or all relevant labels), which existing tests cover.
- **Decision (0)**: same result (checked to 12 digits with ties), the dense `y_score` is already materialized.

## Upstream search

- "label_ranking_average_precision_score" (26 results, all reviewed), "ranking metric slow loop" (GitHub issue search).
- #22575 / #23442 (closed / merged): sparse `y_true` support, #13412 / #10845 / #10840: `sample_weight` and averaging semantics. None about speed.
- #17626 (open) "PERF Consider using argpartition in ndcg_score": another ranking metric, different function and cause.
- No match for the speed of LRAP or `label_ranking_loss`.

## Describe the bug and its user-facing impact

`label_ranking_average_precision_score` loops over samples in Python and calls `scipy.stats.rankdata` twice per sample. The per-call overhead dominates: ~120 µs per sample with 20 labels, so ~5 s for 40,000 samples and ~2 min for 1M samples. `rankdata` has accepted `axis` since SciPy 1.4, so both rankings can be computed for all rows at once, which gives the same score ~35x faster.

`label_ranking_loss` has the same structure (per-sample loop, `sklearn/metrics/_ranking.py:1630`), ~15 µs per sample. `coverage_error`, by contrast, is vectorized and takes ~1 µs per sample.

## Steps/Code to Reproduce

```python
from time import perf_counter

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import label_ranking_average_precision_score


def lrap_rowwise(y_true, y_score):
    # Same formula as scikit-learn, with rankdata applied to all rows at once.
    y_true = np.asarray(y_true, dtype=bool)
    n_labels = y_true.shape[1]
    n_relevant = y_true.sum(axis=1)
    # rank: number of labels scored at least as high as label j
    rank = rankdata(-y_score, "max", axis=1)
    # L: same count among relevant labels only (irrelevant ones pushed last)
    L = rankdata(np.where(y_true, -y_score, np.inf), "max", axis=1)
    aux = np.where(y_true, L / rank, 0).sum(axis=1) / np.maximum(n_relevant, 1)
    aux[(n_relevant == 0) | (n_relevant == n_labels)] = 1.0
    return aux.mean()


rng = np.random.RandomState(0)
for n_samples in [10_000, 20_000, 40_000]:
    y_true = rng.randint(2, size=(n_samples, 20))
    y_score = rng.rand(n_samples, 20).round(2)  # rounding creates ties
    t0 = perf_counter()
    a = label_ranking_average_precision_score(y_true, y_score)
    t1 = perf_counter()
    b = lrap_rowwise(y_true, y_score)
    t2 = perf_counter()
    print(f"n_samples={n_samples}: sklearn {t1 - t0:.2f}s ({a:.12f}), "
          f"row-wise rankdata {t2 - t1:.3f}s ({b:.12f})")
```

## Expected Results

The metric costs about as much as the row-wise version (a few µs per sample), like the vectorized `coverage_error`.

## Actual Results

Timings from a laptop loaded by other jobs:

```
n_samples=10000: sklearn 1.34s (0.567686195090), row-wise rankdata 0.034s (0.567686195090)
n_samples=20000: sklearn 2.58s (0.564697126412), row-wise rankdata 0.072s (0.564697126412)
n_samples=40000: sklearn 4.98s (0.565999651209), row-wise rankdata 0.115s (0.565999651209)
```

## Root cause and possible fix

`sklearn/metrics/_ranking.py:1455-1470` (on 289c4d5cfa):

```python
for i, (start, stop) in enumerate(zip(y_true.indptr, y_true.indptr[1:])):
    relevant = y_true.indices[start:stop]
    ...
        scores_i = y_score[i]
        rank = rankdata(scores_i, "max")[relevant]
        L = rankdata(scores_i[relevant], "max")
        aux = (L / rank).mean()
```

Possible fix, as in `lrap_rowwise` above: `rank = rankdata(y_score, "max", axis=1)` on the (already negated) scores, and `L = rankdata(np.where(y_true, y_score, np.inf), "max", axis=1)` (pushing the irrelevant labels past all relevant ones leaves the rank of each relevant label among the relevant ones unchanged). Then sum `L / rank` over the relevant labels, divide by their count, set the degenerate rows to 1 and apply `sample_weight` as today. `y_true` can be sparse, so it needs a `.toarray()` (or the mask built from `indptr`/`indices`), which is fine since `y_score` is dense with the same shape. `label_ranking_loss` can get the same treatment in a follow-up.

## Versions

```
sklearn: 1.10.dev0, commit 289c4d5cfa (upstream/main; checkout at f570497251, no diff in sklearn/metrics/_ranking.py)
python: 3.13, numpy: 2.5.3, scipy: 1.18.1, platform: linux
```
