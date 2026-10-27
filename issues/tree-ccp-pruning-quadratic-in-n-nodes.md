# Minimal cost-complexity pruning is quadratic in the number of tree nodes

```json
{
    "estimators": ["DecisionTreeClassifier", "DecisionTreeRegressor", "ExtraTreeClassifier", "ExtraTreeRegressor", "RandomForestClassifier", "RandomForestRegressor", "GradientBoostingClassifier"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 2,
    "difficulty": 2,
    "decision": 0,
    "decision_tags": []
}
```

## Classification

- **Priority (2)**: `ccp_alpha` is not the default, but it is the documented way to prune trees and is commonly tuned with a grid search (one full pruning per fit). At 40k samples, pruning takes ~35x longer than growing the tree, and the gap doubles each time n doubles, so pruning becomes impractical well before the tree itself is slow to grow.
- **Difficulty (2)**: the weakest-link search must use a priority queue (with lazy invalidation of the ancestors whose alpha changes after a prune) instead of a full scan. It's a self-contained change in one Cython function, but the bookkeeping is easy to get subtly wrong.
- **Decision (0)**: same pruned trees and same pruning path, as long as ties are broken by node index like the current scan does (key the heap on `(alpha, node_idx)`).

## Upstream search

- `gh search issues ... ccp_alpha slow`, `... cost complexity pruning slow`, `... cost_complexity_pruning_path quadratic`, `... pruning ccp_alpha time`, `... _cost_complexity_prune`, plus `--match comments cost_complexity_pruning_path slow`: no issue or PR about pruning performance. Only the original implementation PR #12887 (merged) and doc PRs come up.
- Recent commits on `sklearn/tree/_tree.pyx` (categorical splits, `isnan` inlining) don't touch the pruning code.

## Describe the bug and its user-facing impact

When `ccp_alpha > 0`, `fit` grows the full tree, then prunes it with `_cost_complexity_prune`. Each pruning step scans **all** nodes to find the weakest link, and there is one step per pruned branch. With a fully grown tree (the default `max_depth=None`) the number of steps is of the order of the number of nodes, so pruning costs O(n_nodes²). `cost_complexity_pruning_path` always runs the full pruning sequence, so it is always quadratic.

On a noisy regression target with 3 features, growing the tree takes 0.31 s at 40k samples but pruning takes ~10 s, and it is x4 each time `n_samples` doubles (timings from a loaded laptop, `OMP_NUM_THREADS=2`). At 80k samples I measured 0.73 s to grow and ~41 s to prune. A `GridSearchCV` over `ccp_alpha`, or a `RandomForest*(ccp_alpha=...)` (each tree is pruned), multiplies this cost.

## Steps/Code to Reproduce

```python
from time import perf_counter

import numpy as np
from sklearn.tree import DecisionTreeRegressor

rng = np.random.RandomState(0)
for n_samples in [5_000, 10_000, 20_000, 40_000]:
    X = rng.rand(n_samples, 3)
    y = X[:, 0] + rng.rand(n_samples)  # noisy target: fully grown tree, ~2n nodes

    tic = perf_counter()
    tree = DecisionTreeRegressor(random_state=0).fit(X, y)
    t_grow = perf_counter() - tic

    tic = perf_counter()
    DecisionTreeRegressor(random_state=0, ccp_alpha=1e-5).fit(X, y)
    t_pruned = perf_counter() - tic

    tic = perf_counter()
    DecisionTreeRegressor(random_state=0).cost_complexity_pruning_path(X, y)
    t_path = perf_counter() - tic

    print(
        f"n_samples={n_samples:>6} n_nodes={tree.tree_.node_count:>6} | "
        f"fit: {t_grow:.2f}s | fit(ccp_alpha=1e-5): {t_pruned:.2f}s | "
        f"cost_complexity_pruning_path: {t_path:.2f}s"
    )
```

## Expected Results

Pruning should cost about as much as growing the tree, or less: the weakest-link pruning of Breiman et al. can be done in O(n_nodes log n_nodes) per path (times the depth for the ancestor updates) with a priority queue. Fit time with `ccp_alpha > 0` should scale like fit time without it.

## Actual Results

```
n_samples=  5000 n_nodes=  9999 | fit: 0.03s | fit(ccp_alpha=1e-5): 0.07s | cost_complexity_pruning_path: 0.09s
n_samples= 10000 n_nodes= 19999 | fit: 0.06s | fit(ccp_alpha=1e-5): 0.52s | cost_complexity_pruning_path: 0.55s
n_samples= 20000 n_nodes= 39999 | fit: 0.13s | fit(ccp_alpha=1e-5): 2.46s | cost_complexity_pruning_path: 2.54s
n_samples= 40000 n_nodes= 79999 | fit: 0.31s | fit(ccp_alpha=1e-5): 10.34s | cost_complexity_pruning_path: 10.49s
```

## Root cause and possible fix

`sklearn/tree/_tree.pyx:1834-1844` (`_cost_complexity_prune`):

```cython
        while candidate_nodes[0]:
            # computes ccp_alpha for subtrees and finds the minimal alpha
            effective_alpha = max_float64
            for i in range(n_nodes):
                if not candidate_nodes[i]:
                    continue
                subtree_alpha = (r_node[i] - r_branch[i]) / (n_leaves[i] - 1)
                ...
```

Each iteration of the outer loop prunes one branch and the inner loop is a full O(n_nodes) scan, hence O(n_nodes × n_prunes). It is called from `BaseDecisionTree._prune_tree` (`sklearn/tree/_classes.py:838`, every `fit` with `ccp_alpha > 0`) and from `cost_complexity_pruning_path` (`sklearn/tree/_classes.py:879`, via `ccp_pruning_path`).

Possible fix: keep the candidate nodes in a min-heap keyed by `(subtree_alpha, node_idx)`. After pruning a branch, the code already walks up the ancestors to update `n_leaves` and `r_branch` (lines 1877-1882); push the updated `(alpha, node)` for each ancestor there and skip stale heap entries when popping (compare with the current alpha, or a per-node version counter, and with `candidate_nodes`). That makes each step O(depth × log n_nodes). Keying on `(alpha, node_idx)` keeps the current tie-breaking (smallest index wins), so results don't change.

A second, smaller term is the initial "bubble up" from every leaf to the root (lines 1819-1828), O(n_leaves × depth). It's fine for balanced trees and could be replaced by a single post-order pass if needed.

## Versions

```
sklearn: 1.10.dev0, commit 289c4d5cfa
python: 3.13.14, numpy: 2.5.3, scipy: 1.18.1, platform: linux
```
