# `IsolationForest` threshold for float `contamination` ignores `sample_weight`

```json
{
    "estimators": ["IsolationForest"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 2,
    "difficulty": 1,
    "decision": 1,
    "decision_tags": ["output-change"]
}
```

## Classification

- **Priority (2)**: silent wrong threshold whenever `contamination` is a float and `sample_weight` is not uniform. Zero weights make it obvious: excluded rows still define `offset_`, and the fraction of flagged (weighted) training rows can be far from `contamination`. Limited to users passing `sample_weight` to `IsolationForest`, which is not the most common use.
- **Difficulty (1)**: replace `np.percentile` with `sklearn.utils.stats._weighted_percentile` when `sample_weight` is given.
- **Decision (1)**: `output-change`: `offset_` changes for weighted fits with a float `contamination`.

## Upstream search

GraphQL search on scikit-learn/scikit-learn: "IsolationForest contamination sample_weight offset", "offset_ percentile sample_weight outlier", "IsolationForest sample_weight".

- #34673 (open, with PRs #34884 and #35058): zero `sample_weight` crashes `IsolationForest` when fewer non-zero weights than `max_samples`. Different symptom; with more non-zero weights than `max_samples` (as below), `fit` works and the bug here shows.
- #16298 (open): list of estimators with incorrect `sample_weight` handling, does not mention the `contamination` threshold.
- #27626 (closed): sparse input with float `contamination`, unrelated.

No match for this bug.

## Describe the bug and its user-facing impact

The docstring defines `contamination` as "the proportion of outliers in the data set", used "to define the threshold on the scores of the samples". With a float `contamination`, `fit` sets `offset_` to the unweighted percentile of the training scores. `sample_weight` is used to draw the subsamples of the trees, but not for this threshold. Rows with weight 0, which should not exist for the model, can then move the threshold arbitrarily. Non-uniform positive weights give a threshold that does not match the weighted proportion of outliers either.

In the reproducer, 1000 far-away rows get a zero weight. The trees ignore them, but they fill the lower tail of the scores, so none of the 1000 weighted rows is flagged instead of 10%.

## Steps/Code to Reproduce

```python
import numpy as np
from sklearn.ensemble import IsolationForest

rng = np.random.RandomState(0)
X = rng.randn(2000, 2)
# Rows 1000+ are far away and excluded with a zero weight.
X[1000:] += 20
sw = np.r_[np.ones(1000), np.zeros(1000)]

iso = IsolationForest(contamination=0.1, random_state=0).fit(X, sample_weight=sw)
print("fraction of weighted rows flagged as outliers:", np.mean(iso.predict(X[:1000]) == -1))
iso = IsolationForest(contamination=0.1, random_state=0).fit(X[:1000])
print("same, zero-weighted rows dropped             :", np.mean(iso.predict(X[:1000]) == -1))
```

## Expected Results

About 10% in both cases: zero-weighted rows are ignored, and `offset_` is the weighted `contamination` quantile of the training scores.

## Actual Results

```
fraction of weighted rows flagged as outliers: 0.0
same, zero-weighted rows dropped             : 0.1
```

## Root cause and possible fix

`sklearn/ensemble/_iforest.py:376`:

```python
self.offset_ = np.percentile(self._score_samples(X), 100.0 * self.contamination)
```

Fix: when `sample_weight` is not None, use `_weighted_percentile(scores, sample_weight, 100 * self.contamination)` (as done in other weighted quantile code). Other outlier detectors with a `contamination` parameter and `sample_weight` support may deserve the same check.

## Versions

```
sklearn: 1.10.dev0, commit f570497251 (sklearn/ensemble/_iforest.py identical to upstream/main 289c4d5cfa)
python: 3.13, numpy: 2.5.3, scipy: 1.18.1, platform: linux
```
