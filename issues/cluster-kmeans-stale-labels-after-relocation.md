# KMeans returns stale labels_ and inertia_ when it converges right after an empty-cluster relocation

```json
{
    "estimators": ["KMeans", "k_means", "BisectingKMeans"],
    "existing_issue": "https://github.com/scikit-learn/scikit-learn/issues/34074",
    "reproduced": true,
    "priority": 1,
    "difficulty": 1,
    "decision": 0,
    "decision_tags": []
}
```

## Classification

- **Priority (1)**: `labels_`/`inertia_` inconsistent with `cluster_centers_`, silently, but only when an empty cluster gets relocated and the labels do not change in that iteration. In a fuzz over tiny integer datasets it only happened with `n_clusters` close to the number of distinct points or with zero sample weights. Maintainers called it a very edge case on #34074.
- **Difficulty (1)**: have the relocation report that it moved a center (or compare centers) and don't declare strict convergence in that iteration, so the final E-step runs.
- **Decision (0)**: no trade-off; at most one extra E-step in rare cases.

## Upstream search

Open issue #34074 describes this exact bug (labels `Bug`, `Needs Decision`; a maintainer leans toward fixing it). Two PRs attempted a fix and were closed without merge: #34134 (linter failure) and #34329 (labelled spam). Queries: `KMeans predict inconsistent labels_`, `kmeans empty clusters relocation`. Related: #28055 / #28165 (infinite loop in relocation, merged fix), and `issues/cluster-kmeans-empty-cluster-copies-unnormalized-center.md` (another relocation bug that combines with this one).

## Describe the bug and its user-facing impact

`_kmeans_single_lloyd` and `_kmeans_single_elkan` declare "strict convergence" when the labels did not change during the last iteration, and then skip the final E-step that makes `labels_` consistent with the returned centers. But `_relocate_empty_clusters_*` moves centers without touching `labels`. If an iteration relocates an empty cluster and the labels are the same as in the previous iteration, the loop stops with centers that the labels were not computed with.

Result: `labels_ != predict(X)` on the training data, `inertia_ != -score(X)`, and the `ConvergenceWarning` about "fewer distinct clusters" is based on the stale labels. With default parameters it reproduces on sparse input; dense input hits it too (different tie-breaking because dense `X` is centered), e.g. in a fuzz with `n_clusters` above the number of distinct points.

## Steps/Code to Reproduce

Adapted from #34074 (default parameters, sparse input):

```python
import numpy as np
from scipy import sparse
from sklearn.cluster import KMeans

X = sparse.csr_matrix(np.array([[1.0], [2.0], [3.0]]))
init = np.array([[4.0], [0.0], [1.0]])
for algorithm in ["lloyd", "elkan"]:
    km = KMeans(n_clusters=3, init=init, algorithm=algorithm).fit(X)
    print(algorithm, "centers:", km.cluster_centers_.ravel(), "labels_:", km.labels_,
          "predict:", km.predict(X), "inertia_:", km.inertia_, "-score:", -km.score(X))
```

## Expected Results

`labels_` equal to `predict(X)` and `inertia_` equal to `-score(X)`: the `labels_` produced by `fit` are meant to be reassigned against the final centers (that is the purpose of the final E-step).

## Actual Results

```
lloyd centers: [3. 2. 1.] labels_: [2 2 0] predict: [2 1 0] inertia_: 1.0 -score: 0.0
elkan centers: [3. 2. 1.] labels_: [2 2 0] predict: [2 1 0] inertia_: 1.0 -score: 0.0
```

(plus `ConvergenceWarning: Number of distinct clusters (2) found smaller than n_clusters (3)`, wrong since the final centers give 3 clusters).

## Root cause and possible fix

`sklearn/cluster/_kmeans.py:580` (Elkan) and `:718` (Lloyd) break with `strict_convergence = True` on `np.array_equal(labels, labels_old)`, which skips the E-step at `:599` / `:737`. The relocation in `sklearn/cluster/_k_means_common.pyx` (`_relocate_empty_clusters_dense` / `_sparse`) changes `centers_new` and `weight_in_clusters` but not `labels`. Fix: make the relocation return whether it moved anything and only declare strict convergence when it did not.

## Versions

```
sklearn: 1.10.dev0, commit 289c4d5cfa
python: 3.13, numpy: 2.5.3, scipy: 1.18.1, platform: linux
```
