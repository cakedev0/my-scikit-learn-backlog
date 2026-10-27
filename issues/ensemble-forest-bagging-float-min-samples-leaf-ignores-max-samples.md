# Float `min_samples_leaf` / `min_samples_split` in forests and Bagging are relative to `X.shape[0]`, not to the bootstrap sample

```json
{
    "estimators": ["RandomForestRegressor", "RandomForestClassifier", "ExtraTreesRegressor", "ExtraTreesClassifier", "BaggingRegressor", "BaggingClassifier"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 2,
    "difficulty": 2,
    "decision": 1,
    "decision_tags": ["output-change"]
}
```

## Classification

- **Priority (2)**: silently degenerate trees (a handful of leaves) when a fractional `min_samples_leaf` or `min_samples_split` is combined with `max_samples` well below `n_samples`. Both settings are common for large datasets. With the default `max_samples=None` the effect is milder (the fraction applies to `n_samples` instead of ~63% unique rows).
- **Difficulty (2)**: the forest (or Bagging) has to resolve the fraction against the bootstrap size and pass an integer to each tree, or the tree must count sample weights instead of rows; the second option also changes what integer `min_samples_leaf` means.
- **Decision (1)**: `output-change` for all fits that use a fractional `min_samples_leaf`/`min_samples_split` in an ensemble.

## Upstream search

GraphQL search on scikit-learn/scikit-learn: "min_samples_leaf float max_samples random forest", "min_samples_split fraction bootstrap forest n_samples", "min_samples_leaf bootstrap unique samples sample_weight forest", "max_samples min_samples_split relative", "min_samples_leaf fraction relative to bootstrap". Only unrelated results (#9645, #3207 min_weight_fraction_leaf, #3359 the original float support). No match.

## Describe the bug and its user-facing impact

Forests and Bagging (when the base estimator accepts `sample_weight`) do not index rows for each tree: they pass the full `X` with the bootstrap counts as `sample_weight`. The tree then resolves fractional parameters with `n_samples = X.shape[0]`. So for `RandomForestRegressor(max_samples=500, min_samples_leaf=0.01)` on 10,000 rows, each tree requires 100 rows per leaf out of ~490 unique in-bag rows, instead of 5. Trees end up with 3 or 4 leaves, while the same trees fitted on the bootstrap samples directly have ~80 leaves.

This is also inconsistent with `min_weight_fraction_leaf`, which is relative to `sample_weight.sum()`, i.e. to the bootstrap size. A related detail: leaf sizes are counted in unique in-bag rows (`n_node_samples` ignores the counts), not in drawn samples.

## Steps/Code to Reproduce

```python
from sklearn.datasets import make_regression
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

X, y = make_regression(n_samples=10_000, n_features=5, noise=10, random_state=0)


def n_leaves(trees):
    return sorted({int(t.get_n_leaves()) for t in trees})


rf = RandomForestRegressor(
    n_estimators=20, max_samples=500, min_samples_leaf=0.01, random_state=0
).fit(X, y)
print("RF, leaves per tree             :", n_leaves(rf.estimators_))

# The same trees fitted directly on each bootstrap sample:
direct = [
    DecisionTreeRegressor(min_samples_leaf=0.01, random_state=0).fit(X[idx], y[idx])
    for idx in rf.estimators_samples_
]
print("trees fitted on bootstrap samples:", n_leaves(direct))

bag = BaggingRegressor(
    DecisionTreeRegressor(min_samples_leaf=0.01),
    n_estimators=20, max_samples=500, random_state=0,
).fit(X, y)
print("Bagging, leaves per tree        :", n_leaves(bag.estimators_))
```

## Expected Results

Each tree is grown on its bootstrap sample, so `min_samples_leaf=0.01` means `ceil(0.01 * 500) = 5` samples per leaf, as for a tree fitted on that sample directly (~80 leaves).

## Actual Results

```
RF, leaves per tree             : [3, 4]
trees fitted on bootstrap samples: [75, 76, 77, 79, 80, 81, 83]
Bagging, leaves per tree        : [3, 4]
```

## Root cause and possible fix

- `sklearn/ensemble/_forest.py:160` (upstream/main) and `sklearn/ensemble/_bagging.py:169` turn the bootstrap indices into counts passed as `sample_weight` on the full `X`.
- `sklearn/tree/_classes.py:343`, `:388` and `:393` (upstream/main) resolve the fractions with `n_samples = X.shape[0]`, while `:436` resolves `min_weight_fraction_leaf` against `np.sum(sample_weight)`.

Possible fix: in `_parallel_build_trees` and `_parallel_build_estimators`, set `min_samples_leaf` / `min_samples_split` on the tree to the integer `ceil(fraction * n_samples_bootstrap)` before fitting (the forest knows `n_samples_bootstrap`). GradientBoosting with `subsample < 1` has the same pattern (masked `sample_weight` on the full `X`), with a smaller effect since `subsample` is usually 0.5 or more.

## Versions

```
sklearn: 1.10.dev0, commit f570497251 (relevant code identical to upstream/main 289c4d5cfa; line numbers above are from upstream/main)
python: 3.13, numpy: 2.5.3, scipy: 1.18.1, platform: linux
```
