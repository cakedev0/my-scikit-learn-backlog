# `CategoricalNB.n_categories_` only reflects the last `partial_fit` batch

```json
{
    "estimators": ["CategoricalNB"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 1,
    "difficulty": 1,
    "decision": 0,
    "decision_tags": []
}
```

## Classification

- **Priority (1)**: a fitted attribute is wrong after incremental fitting, but predictions are not affected (they use `category_count_` / `feature_log_prob_`, which are updated correctly). Users reading `n_categories_` (e.g. to build an encoder or check unseen categories) get a too small value.
- **Difficulty (1)**: take the maximum with the previous value (or derive it from `category_count_[i].shape[1]`).
- **Decision (0)**: no trade-off.

## Upstream search

GraphQL searches "CategoricalNB n_categories_ partial_fit", "CategoricalNB partial_fit categories". No match (#12569, merged, added `CategoricalNB`; #18925, merged, degenerate single-class case).

## Describe the bug and its user-facing impact

`CategoricalNB.n_categories_` is documented as "Number of categories for each feature. This value is inferred from the data or set by the minimum number of categories." With `partial_fit`, it is recomputed from the current batch only (`max(X) + 1`, or `min_categories`), so after a batch that contains only low category codes it shrinks, while `category_count_` and `feature_log_prob_` keep all the categories seen so far. After the same data, `fit` and a sequence of `partial_fit` calls disagree on `n_categories_`, and the attribute is inconsistent with `feature_log_prob_[i].shape[1]` of the same model.

## Steps/Code to Reproduce

```python
import numpy as np
from sklearn.naive_bayes import CategoricalNB

X = np.array([[0], [5], [3], [1], [0], [1]])
y = np.array([0, 1, 0, 1, 0, 1])

full = CategoricalNB().fit(X, y)
inc = CategoricalNB().partial_fit(X[:4], y[:4], classes=[0, 1])
inc.partial_fit(X[4:], y[4:])  # second batch only has categories 0 and 1

print("fit         n_categories_:", full.n_categories_, "feature_log_prob_ shape:", full.feature_log_prob_[0].shape)
print("partial_fit n_categories_:", inc.n_categories_, "feature_log_prob_ shape:", inc.feature_log_prob_[0].shape)
```

## Expected Results

`n_categories_` is `[6]` in both cases, matching `feature_log_prob_[0].shape[1]`.

## Actual Results

```
fit         n_categories_: [6] feature_log_prob_ shape: (2, 6)
partial_fit n_categories_: [2] feature_log_prob_ shape: (2, 6)
```

## Root cause and possible fix

`sklearn/naive_bayes.py:1521` in `CategoricalNB._count`: `self.n_categories_ = self._validate_n_categories(X, self.min_categories)` overwrites the attribute with the value of the current batch. Fix: after updating `category_count_`, set `self.n_categories_ = np.array([c.shape[1] for c in self.category_count_], dtype=np.int64)` (or `np.maximum` with the previous value on non-first calls).

## Versions

```
sklearn: 1.10.dev0, commit 289c4d5cfa (upstream/main; local HEAD f570497251 has no diff in sklearn/naive_bayes.py)
python: 3.13, numpy: 2.5.3, scipy: 1.18.1, platform: linux
```
