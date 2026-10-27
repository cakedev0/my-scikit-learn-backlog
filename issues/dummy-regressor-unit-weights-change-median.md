# DummyRegressor median/quantile changes when passing unit sample_weight

```json
{
    "estimators": ["DummyRegressor"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 1,
    "difficulty": 1,
    "decision": 1,
    "decision_tags": ["output-change", "unclear-spec"]
}
```

## Classification

- **Priority (1)**: a baseline estimator gives a different constant with `sample_weight=np.ones(n)` than without weights (2.0 vs 2.5 below). Only `strategy="median"`/`"quantile"` with weights, and only when the cumulative weight lands exactly on the quantile, so mostly integer or equal weights.
- **Difficulty (1)**: for the median, pass `average=True` to `_weighted_percentile` (one line, open PR #34776). For other quantiles, there is no weighted equivalent of numpy's default `linear` method.
- **Decision (1)**: the median fix changes results for weighted users (`output-change`). For `strategy="quantile"`, someone must pick a quantile definition that both paths share (`unclear-spec`), e.g. switching the unweighted path to `np.percentile(..., method="inverted_cdf")` or `"averaged_inverted_cdf"`, which changes unweighted results too.

## Upstream search

- `gh issue list --search "DummyRegressor median sample_weight"`: #31367 (closed, NaN handling in `_weighted_percentile`), #3420 (closed, adding sample_weight support). No open issue.
- `gh pr list --search "DummyRegressor median sample_weight"`: #34776 (open, "FIX use averaged inverted CDF in weighted DummyRegressor median") fixes the median case only and leaves `strategy="quantile"` as is; it references the meta-issue #16298 on `sample_weight` equivalence.

## Describe the bug and its user-facing impact

Without weights, `DummyRegressor` uses `np.median` / `np.percentile` (linear interpolation). With weights it uses `_weighted_percentile` with the default `inverted_cdf` method. For an even number of equally weighted samples the two disagree, so unit weights are not a no-op and integer weights are not equivalent to repeating samples. A user comparing weighted and unweighted baselines (or a model to a weighted dummy) gets a shifted constant.

## Steps/Code to Reproduce

```python
import numpy as np
from sklearn.dummy import DummyRegressor

X = np.zeros((4, 1))
y = np.array([1.0, 2.0, 3.0, 4.0])
for sample_weight in [None, np.ones(4)]:
    median = DummyRegressor(strategy="median").fit(X, y, sample_weight=sample_weight)
    q50 = DummyRegressor(strategy="quantile", quantile=0.5).fit(X, y, sample_weight=sample_weight)
    print(f"sample_weight={sample_weight}: median {median.constant_.item()}, quantile(0.5) {q50.constant_.item()}")
```

## Expected Results

Same constant (2.5) with and without unit weights.

## Actual Results

```
sample_weight=None: median 2.5, quantile(0.5) 2.5
sample_weight=[1. 1. 1. 1.]: median 2.0, quantile(0.5) 2.0
```

## Root cause and possible fix

`sklearn/dummy.py:576-578` and `sklearn/dummy.py:590-592` call `_weighted_percentile(y, sample_weight, percentile_rank=...)` with the default `average=False`, while the unweighted branches use `np.median` and `np.percentile` with the default `linear` method. For the median, `average=True` matches `np.median` exactly (PR #34776). For `strategy="quantile"`, the two branches need a shared definition, which probably means a deprecation or a new parameter like `KBinsDiscretizer`'s `quantile_method`.

## Versions

```
sklearn: 1.10.dev0, commit 289c4d5cfa (upstream/main; reproduced on local checkout f570497251, no diff in sklearn/dummy.py)
python: 3.13, numpy: 2.5.3, scipy: 1.18.1, platform: linux
```
