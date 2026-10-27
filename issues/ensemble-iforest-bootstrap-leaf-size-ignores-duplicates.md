# `IsolationForest(bootstrap=True)` ignores rows drawn several times in the leaf path-length correction

```json
{
    "estimators": ["IsolationForest"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 1,
    "difficulty": 1,
    "decision": 1,
    "decision_tags": ["output-change"]
}
```

## Classification

- **Priority (1)**: `bootstrap=True` is not the default and not part of the original algorithm; the scores are shifted by a few percent, and the ranking of samples changes only a little. It also applies to `sample_weight` combined with `bootstrap=True`, where frequency semantics are expected.
- **Difficulty (1)**: use `tree_.weighted_n_node_samples` instead of `tree_.n_node_samples` in `IsolationForest.fit`. Bagging always fits the trees with `sample_weight=bincount(drawn indices)`, so the weighted count is exactly the number of drawn rows.
- **Decision (1)**: `output-change`: `score_samples`, `decision_function`, `offset_` (with a float `contamination`) and `predict` change for `bootstrap=True`. Nothing changes for the default `bootstrap=False` without `sample_weight` (all weights are 1).

## Upstream search

Queries: `IsolationForest bootstrap n_node_samples`, `IsolationForest average path length sample_weight`, `IsolationForest weighted_n_node_samples`, `IsolationForest sample_weight repetition`, `IsolationForest bootstrap`.

- #16298 (open meta-issue on `sample_weight` handling): lists `IsolationForest` as unchecked, without details.
- #34673 (open, with PRs #34884, #35058): crash with zero `sample_weight` and `bootstrap=False`, different code path.

No match for this root cause.

## Describe the bug and its user-facing impact

`bootstrap=True` is documented as "individual trees are fit on random subsets of the training data sampled with replacement". Bagging implements it by fitting each `ExtraTreeRegressor` on all rows with `sample_weight = number of times each row was drawn`. The tree structure is the one that a fit on the repeated rows would give, but `tree_.n_node_samples` counts distinct rows with a non-zero weight. `IsolationForest` uses `n_node_samples` for the leaf correction `c(n)` added to the depth of each leaf (Liu et al. 2008, eq. 1 and 2). A leaf holding one row drawn 3 times gets `c(1) = 0` instead of `c(3)`.

So the scores depend on how the bootstrap sample is represented: an exact duplicate row in `X` (with `bootstrap=False`) gets `c(2)`, the same row drawn twice by the bootstrap gets `c(1)`. In the reproducer, 33 of the 50 leaves of the first tree have this mismatch, and all scores shift by ~0.02-0.03.

## Steps/Code to Reproduce

```python
import numpy as np
from sklearn.ensemble import IsolationForest


def c(n):
    # average path length of an unsuccessful BST search (Liu et al. 2008)
    n = np.asarray(n, dtype=float)
    out = np.zeros_like(n)
    out[n == 2] = 1.0
    big = n > 2
    out[big] = 2.0 * (np.log(n[big] - 1.0) + np.euler_gamma) - 2.0 * (n[big] - 1.0) / n[big]
    return out


def score(clf, X, leaf_size):
    depths = np.zeros(X.shape[0])
    for tree in clf.estimators_:
        t = tree.tree_
        leaves = tree.apply(X)
        n_leaf = t.weighted_n_node_samples if leaf_size == "drawn" else t.n_node_samples
        depths += t.compute_node_depths()[leaves] - 1.0 + c(n_leaf)[leaves]
    return -(2 ** (-depths / (len(clf.estimators_) * c([clf.max_samples_])[0])))


rng = np.random.RandomState(0)
X = rng.randn(256, 2)
X_test = np.array([[0.0, 0.0], [4.0, 4.0]])
clf = IsolationForest(n_estimators=200, bootstrap=True, random_state=0).fit(X)

t = clf.estimators_[0].tree_
leaf = t.children_left == -1
print("rows drawn in each leaf    :", t.weighted_n_node_samples[leaf][:8])
print("n_node_samples of the leaf :", t.n_node_samples[leaf][:8])
print("score_samples              :", clf.score_samples(X_test))
print("c(distinct rows in leaf)   :", score(clf, X_test, "distinct"))
print("c(drawn rows in leaf)      :", score(clf, X_test, "drawn"))
```

## Expected Results

The leaf correction uses the number of rows of the bootstrap sample in the leaf (`weighted_n_node_samples`), i.e. `score_samples` matches the last line, as it would if the trees were fitted on the repeated rows.

## Actual Results

```
rows drawn in each leaf    : [1. 2. 3. 2. 1. 3. 1. 7.]
n_node_samples of the leaf : [1 1 1 1 1 1 1 5]
score_samples              : [-0.43047565 -0.75756091]
c(distinct rows in leaf)   : [-0.43047565 -0.75756091]
c(drawn rows in leaf)      : [-0.40765267 -0.73091278]
```

## Root cause and possible fix

`sklearn/ensemble/_iforest.py:354-362`:

```python
_average_path_length(tree.tree_.n_node_samples),
```

Bagging passes the drawn rows as `sample_weight=np.bincount(indices)` (`sklearn/ensemble/_bagging.py:167-170`), so `n_node_samples` undercounts every leaf that contains a row drawn more than once. Replace it with `tree.tree_.weighted_n_node_samples`. `_average_path_length` already handles non-integer inputs. The user `sample_weight` is only used to draw the indices, never passed to the trees, so the weighted count is always an integer count of drawn rows.

## Versions

```
sklearn: 1.10.dev0, commit 289c4d5cfa
python: 3.13, numpy: 2.5.3, scipy: 1.18.1, platform: linux
```
