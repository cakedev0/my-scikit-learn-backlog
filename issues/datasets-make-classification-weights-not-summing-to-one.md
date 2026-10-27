# `make_classification` silently distorts class proportions when `weights` don't sum to 1

```json
{
    "estimators": ["make_classification"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 1,
    "difficulty": 1,
    "decision": 1,
    "decision_tags": ["unclear-spec", "output-change"]
}
```

## Classification

- **Priority (1)**: silent but needs `weights` that don't sum to 1, which most users avoid; the docstring describes a behavior (more than `n_samples` samples) that never happens.
- **Difficulty (1)**: normalize `weights_` (or raise) in a couple of lines, and update the docstring.
- **Decision (1)**: `unclear-spec`: normalize, raise, or only fix the docstring. `output-change`: normalizing changes the generated data for these inputs (only for them).

## Upstream search

Queries: "make_classification weights sum", "make_classification weights sum greater than 1".

- #12985 (open, "Needs Decision", "make_classification does not respect class weights as well as possible?"): related, same code path. It's about rounding with normalized weights and small `n_samples` (the deficit from `int()` is spread round-robin over clusters); @jnothman wasn't sure it was worth fixing. It doesn't mention unnormalized weights or the docstring promise of more than `n_samples` samples, where the distortion is large at any `n_samples`.

The sentence "More than ``n_samples`` samples may be returned if the sum of ``weights`` exceeds 1" was added in 224a4d593b (2014, "TST/COSMIT cleaner make_classification and fix tests"), when `X` was already allocated as `np.zeros((n_samples, n_features))`, so it has never matched the code.

## Describe the bug and its user-facing impact

`weights` is documented as "the proportions of samples assigned to each class", and "More than `n_samples` samples may be returned if the sum of `weights` exceeds 1". In practice:

- the output always has exactly `n_samples` rows;
- if the sum exceeds 1, the per-cluster counts overflow `n_samples` and the last clusters are truncated, so `weights=[0.7, 0.7]` gives a 65/35 split instead of 50/50;
- if the sum is below 1, the missing samples are spread round-robin over the clusters, so `weights=[0.1, 0.3]` gives 40/60 instead of 25/75.

The data looks plausible, so a user passing unnormalized weights (e.g. class counts ratios like `[1, 3]`, or rounded proportions) gets a class balance that matches neither the weights nor their normalized version.

## Steps/Code to Reproduce

```python
import numpy as np
from sklearn.datasets import make_classification

for weights in [[0.25, 0.75], [0.1, 0.3], [0.7, 0.7], [0.3, 0.9]]:
    X, y = make_classification(n_samples=100, weights=weights, flip_y=0, random_state=0)
    print(weights, X.shape, np.bincount(y))
```

## Expected Results

Either the proportions of the normalized weights (`[25 75]`, `[50 50]`, `[25 75]` for the last three), more than 100 samples for the sums above 1 as the docstring says, or an error.

## Actual Results

```
[0.25, 0.75] (100, 20) [25 75]
[0.1, 0.3] (100, 20) [40 60]
[0.7, 0.7] (100, 20) [65 35]
[0.3, 0.9] (100, 20) [30 70]
```

## Root cause and possible fix

`sklearn/datasets/_samples_generator.py:270-274` (upstream/main 289c4d5cfa) computes `int(n_samples * weights_[k % n_classes] / n_clusters_per_class)` per cluster without normalizing `weights_`, then fills the clusters into an array of `n_samples` rows (line 279), so any surplus is cut from the last clusters and any deficit is spread evenly.

I'd normalize `weights_` by its sum when `len(weights) == n_classes` (and raise if the inferred last weight `1 - sum(weights)` is negative when `len(weights) == n_classes - 1`), and drop the "More than `n_samples`" sentence from the docstring.

## Versions

```
sklearn: 1.10.dev0, commit 289c4d5cfa (upstream/main; local build f570497251, no diff on sklearn/datasets/_samples_generator.py)
python: 3.13, numpy: 2.5.3, scipy: 1.18.1, platform: linux
```
