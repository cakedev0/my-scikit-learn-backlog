# `trustworthiness` penalizes a perfect embedding when distances are tied, and depends on the row order

```json
{
    "estimators": ["trustworthiness"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 1,
    "difficulty": 1,
    "decision": 1,
    "decision_tags": ["output-change", "unclear-spec"]
}
```

## Classification

- **Priority (1)**: only data with exactly tied distances (duplicates, discrete or binary features) is affected, and the bias is small (~0.3-1% on the examples below), but it makes the score of an identity embedding < 1 and depend on the row order, which can flip comparisons between close embeddings.
- **Difficulty (1)**: rank input distances with ties resolved in favor of the embedding (e.g. `scipy.stats.rankdata(..., method="min")` per row, or count the points strictly closer).
- **Decision (1)**: `unclear-spec`: Venna & Kaski don't define ranks with ties; "min" rank (no penalty for a neighbor tied with the k-th input neighbor) is the natural choice since it gives 1 for the identity embedding. `output-change`: scores on tied data move up slightly.

## Upstream search

Queries: "trustworthiness", "trustworthiness ties" (comments). Hits are about the `n_neighbors < n_samples / 2` bound (#18567, #18832, #23033), parameter validation (#26276) and the docstring example (#28203). Nothing about ties. The memory issue is filed separately in `issues/manifold-trustworthiness-quadratic-memory.md`; a chunked rewrite should fix both.

## Describe the bug and its user-facing impact

`trustworthiness` ranks the input-space neighbors with `np.argsort(dist_X, axis=1)` and gets the embedding neighbors from `NearestNeighbors.kneighbors`. When several points are at the same distance, each of the two picks an arbitrary order among them, and the orders don't match. A point `j` that is among the `k` nearest neighbors in the embedding can then get an input rank `r(i, j) > k` although it is at the same input distance as the `k`-th input neighbor, and is penalized.

Consequences: `trustworthiness(X, X) < 1` as soon as `X` has tied distances (small integer features, binary features, duplicated rows), and shuffling the rows of both `X` and `X_embedded` changes the score.

## Steps/Code to Reproduce

```python
import numpy as np
from sklearn.manifold import trustworthiness

rng = np.random.RandomState(0)
X = rng.randint(0, 4, size=(200, 2)).astype(float)  # many tied distances
print("X_embedded = X:", trustworthiness(X, X, n_neighbors=5))
for seed in range(2):
    perm = np.random.RandomState(seed).permutation(len(X))
    print("same, rows shuffled:", trustworthiness(X[perm], X[perm], n_neighbors=5))
X = rng.randn(200, 2)
print("no ties, X_embedded = X:", trustworthiness(X, X, n_neighbors=5))
```

## Expected Results

`1.0` for the identity embedding, whatever the row order. With binary features (`X = rng.rand(200, 20) < 0.3`, `n_neighbors=10`), the identity embedding scores 0.99736 instead of 1.

## Actual Results

```
X_embedded = X: 0.992875
same, rows shuffled: 0.99171875
same, rows shuffled: 0.9915885416666667
no ties, X_embedded = X: 1.0
```

## Root cause and possible fix

`sklearn/manifold/_t_sne.py:532` (`ind_X = np.argsort(dist_X, axis=1)`) and the inverted index built from it assign distinct ranks to tied distances. Using the "min" rank (`r(i, j) = 1 + #{l : d(i, l) < d(i, j)}`) only penalizes embedding neighbors that are strictly farther in the input space than the `k`-th input neighbor, which gives 1 for the identity embedding and makes the score permutation invariant with respect to the input-space ties. Ties in the embedding space still make the choice of `N_i^k` arbitrary; using the input "min" rank removes the penalty only when the tie-break differs from the input one, so the identity case is fixed but not every tie in `X_embedded`.

## Versions

```
sklearn: 1.10.dev0, commit 289c4d5cfa
python: 3.13.14, numpy: 2.5.3, scipy: 1.18.1, platform: linux
```
