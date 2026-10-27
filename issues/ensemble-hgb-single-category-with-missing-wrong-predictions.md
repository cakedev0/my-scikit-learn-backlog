# HistGradientBoosting predicts wrong values for a categorical feature with a single category and missing values

```json
{
    "estimators": ["HistGradientBoostingClassifier", "HistGradientBoostingRegressor"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 2,
    "difficulty": 1,
    "decision": 0,
    "decision_tags": []
}
```

## Classification

- **Priority (2)**: silent wrong predictions, with the default `categorical_features="from_dtype"` on a plausible input (a pandas categorical flag column like "yes"/missing). The model learns the right split but `predict` routes the samples wrongly, so even the training score is bad. It needs a categorical feature with exactly one non-missing category, which is uncommon.
- **Difficulty (1)**: swap the order of two branches in `_fill_predictor_arrays` so that the categorical case is handled first.
- **Decision (0)**: no trade-off.

## Upstream search

Queries on open issues/PRs (core API dump of all open issues and of all issues labeled `module:ensemble`/`module:inspection`, grepped for categorical, missing, bitset, single category). No match. The full-text `gh search issues` queries ("HistGradientBoosting categorical single category missing values") failed on the shared search API rate limit, so this relies on titles only. #34691 (RFC on unknown categories vs missing values) and #34553 (doc on categorical missing values) are about other aspects.

## Describe the bug and its user-facing impact

When a categorical feature has a single known category plus missing values, a split on it (category vs missing) is turned into a broken predictor node: both the category and the missing values can be sent to the same child. `predict`, `predict_proba` and `score` are then wrong, silently. Training itself uses the right partition (raw predictions are updated from the grower's sample indices), so the model is fitted for a split that `predict` doesn't apply.

Whether the result is wrong depends on which side the missing values go: in the reproducer below the classifier gets 0.48 training accuracy on a perfectly separable problem, while encoding the same column as numerical gives 1.0. With several categorical splits in a tree, the node can also read the bitset of another split (`bitset_idx` stays 0).

A plausible way to hit it: a pandas categorical column used as a flag, with values `"yes"` or missing, and the default `categorical_features="from_dtype"`.

## Steps/Code to Reproduce

```python
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

rng = np.random.RandomState(0)
flag = np.where(rng.rand(1000) < 0.5, "yes", None)  # one category + missing values
X_cat = pd.DataFrame({"flag": pd.Categorical(flag)})
X_num = pd.DataFrame({"flag": np.where(flag == "yes", 0.0, np.nan)})
y = flag == "yes"

for X in [X_cat, X_num]:
    clf = HistGradientBoostingClassifier().fit(X, y)
    print(f"is_categorical_={clf.is_categorical_}: training accuracy {clf.score(X, y):.3f}")
```

## Expected Results

Training accuracy 1.0 in both cases: `y` is a deterministic function of whether `flag` is missing, and the numerical encoding of the same column gets it right.

## Actual Results

```
is_categorical_=[ True]: training accuracy 0.483
is_categorical_=None: training accuracy 1.000
```

## Root cause and possible fix

In `_fill_predictor_arrays` (`sklearn/ensemble/_hist_gradient_boosting/grower.py:778` on the commit below), the "split on nans" check comes before the categorical check:

```python
if split_info.bin_idx == n_bins_non_missing[feature_idx] - 1:
    # Split is on the last non-missing bin: it's a "split on nans".
    # All nans go to the right, the rest go to the left.
    # Note: for categorical splits, bin_idx is 0 and we rely on the bitset
    node["num_threshold"] = np.inf
elif split_info.is_categorical:
    ...  # sets bitset_idx and fills the binned and raw bitsets
```

For categorical splits, `bin_idx` is always 0. When the feature has a single category, `n_bins_non_missing[feature_idx] - 1 == 0`, so the categorical split takes the first branch: its `bitset_idx` is never set and its bitsets are left empty (or point to the bitset of another split). At predict time (`_predictor.pyx`, `_predict_one_from_raw_data` and `_predict_one_from_binned_data`), the category is looked up in that empty bitset and goes right, and missing values follow `missing_go_to_left`. When the split put the category on the left, everything ends up on the right.

Fix: test `split_info.is_categorical` first (or add `not split_info.is_categorical` to the first condition). The bug dates from the original categorical support (2020).

## Versions

```
sklearn: 1.10.dev0, commit 289c4d5cfa (upstream/main; run on a local editable build whose HGB grower/predictor code is identical to upstream/main)
python: 3.13, numpy: 2.5.3, scipy: 1.18.1, platform: linux
```
