# `IsotonicRegression(increasing="auto")` ignores `sample_weight` when choosing the direction

```json
{
    "estimators": ["IsotonicRegression"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 1,
    "difficulty": 2,
    "decision": 1,
    "decision_tags": ["unclear-spec", "output-change"]
}
```

## Classification

- **Priority (1)**: silent wrong fit, but it needs both `increasing="auto"` (not the default) and weights that change the sign of the correlation. Zero weights used to exclude samples (e.g. in `fit_params` routed by a meta-estimator) are the most likely trigger.
- **Difficulty (2)**: dropping zero-weight samples before the direction test is a one-line move; honoring non-zero weights needs a weighted Spearman correlation and a matching effective sample size for the Fisher-transform warning.
- **Decision (1)**: there is no standard weighted Spearman in scipy, so someone must pick a definition (`unclear-spec`, e.g. weighted Pearson correlation of the ranks); the chosen direction changes for some weighted fits (`output-change`).

## Upstream search

GraphQL searches "IsotonicRegression increasing auto sample_weight", "check_increasing sample_weight", "isotonic increasing auto weights". No match (only #4302, merged, about duplicate X handling).

## Describe the bug and its user-facing impact

With `increasing="auto"`, `IsotonicRegression.fit` picks the direction from the sign of the Spearman correlation between `X` and `y` (`check_increasing`). This is computed on all samples, unweighted, *before* the samples with zero weight are dropped. The rest of the fit (pooling, thresholds) uses the weights. So samples with zero weight (or tiny weight) can flip the direction, and the weighted fit is then forced to be monotone in the wrong direction, giving a constant prediction in the example below.

Expected per the `sample_weight` convention: a sample with zero weight should have the same effect as removing it, and integer weights the same effect as repeating samples.

## Steps/Code to Reproduce

```python
import numpy as np
from sklearn.isotonic import IsotonicRegression

X = np.arange(20.0)
y = np.r_[np.arange(10.0), 100.0 - 10 * np.arange(10, 20)]
sample_weight = np.r_[np.ones(10), np.zeros(10)]  # only the increasing part counts

weighted = IsotonicRegression(increasing="auto").fit(X, y, sample_weight=sample_weight)
subset = IsotonicRegression(increasing="auto").fit(X[:10], y[:10])
print("increasing_ with sample_weight :", weighted.increasing_)
print("increasing_ on non-zero weights:", subset.increasing_)
print("predict with sample_weight     :", weighted.predict(X[:10]))
print("predict on non-zero weights    :", subset.predict(X[:10]))
```

## Expected Results

Fitting with zero weights on the last 10 samples is equivalent to fitting on the first 10: `increasing_` is `True` and the predictions are `0, 1, ..., 9`.

## Actual Results

```
increasing_ with sample_weight : False
increasing_ on non-zero weights: True
predict with sample_weight     : [4.5 4.5 4.5 4.5 4.5 4.5 4.5 4.5 4.5 4.5]
predict on non-zero weights    : [0. 1. 2. 3. 4. 5. 6. 7. 8. 9.]
```

## Root cause and possible fix

`sklearn/isotonic.py:313` calls `check_increasing(X, y)` without the weights, before `sklearn/isotonic.py:320` removes zero-weight samples.

Minimal fix: move the direction test after the zero-weight filtering. Full fix: compute a weighted rank correlation (e.g. weighted Pearson correlation of the ranks) when `sample_weight` is given, and use `sum(sample_weight)` (or the number of non-zero weights) as the sample size in the Fisher-transform warning of `check_increasing`.

## Versions

```
sklearn: 1.10.dev0, commit 289c4d5cfa (upstream/main; local HEAD f570497251 has no diff in sklearn/isotonic.py)
python: 3.13, numpy: 2.5.3, scipy: 1.18.1, platform: linux
```
