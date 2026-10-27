# SpectralEmbedding.fit on sparse input overwrites the affinity parameter with "rbf"

```json
{
    "estimators": ["SpectralEmbedding"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 1,
    "difficulty": 1,
    "decision": 1,
    "decision_tags": ["output-change"]
}
```

## Classification

- **Priority (1)**: a fit on sparse data with the default `affinity="nearest_neighbors"` permanently switches the estimator to `affinity="rbf"`: later fits on dense data, `get_params`, `clone` and `repr` all see `"rbf"`. It warns once, but the side effect on later fits is silent. Needs the same estimator instance to be fitted on sparse then on other data.
- **Difficulty (1)**: use a local variable instead of assigning `self.affinity`.
- **Decision (1)**: fixing the parameter mutation alone is a no-brainer. The fallback itself is outdated (`kneighbors_graph` supports sparse input and gives the same graph as for dense input), and removing it would be the better fix, but it changes the embedding (`output-change`) for users who fit sparse data today and get an rbf embedding.

## Upstream search

- `"SpectralEmbedding sparse nearest_neighbors"`: no match. `git log -L` shows the fallback dates back to the creation of the estimator (c797e01ddb, "ENH finish sketch for the estimator wrapper"), when neighbors did not support sparse input.

## Describe the bug and its user-facing impact

In `SpectralEmbedding._get_affinity_matrix`, when `affinity="nearest_neighbors"` and `X` is sparse, the code warns "Nearest neighbors affinity currently does not support sparse input, falling back to rbf affinity" and then executes `self.affinity = "rbf"`. Estimators must not change their constructor parameters in `fit`: after this, the same instance uses the rbf affinity for every subsequent fit, including on dense data, without any warning.

Also, the reason for the fallback no longer holds: `kneighbors_graph(X_sparse, ...)` works and returns the same graph as on the dense array.

## Steps/Code to Reproduce

```python
import numpy as np
from scipy import sparse
from sklearn.manifold import SpectralEmbedding

rng = np.random.RandomState(0)
X = rng.rand(100, 5)

se = SpectralEmbedding(n_components=2, random_state=0)  # affinity="nearest_neighbors"
E_dense = se.fit_transform(X)
se.fit(sparse.csr_matrix(X))  # warns and falls back to rbf
print("affinity param after sparse fit:", se.get_params()["affinity"])
E_dense_again = se.fit_transform(X)  # dense data again, same estimator
print("same embedding as first dense fit:", np.allclose(np.abs(E_dense), np.abs(E_dense_again)))
```

## Expected Results

`affinity` stays `"nearest_neighbors"` and the second dense fit gives the same embedding as the first one.

## Actual Results

```
sklearn/manifold/_spectral_embedding.py:690: UserWarning: Nearest neighbors affinity currently does not support sparse input, falling back to rbf affinity
  warnings.warn(
affinity param after sparse fit: rbf
same embedding as first dense fit: False
```

## Root cause and possible fix

`sklearn/manifold/_spectral_embedding.py:695` (commit 289c4d5cfa):

```python
if self.affinity == "nearest_neighbors":
    if sparse.issparse(X):
        warnings.warn(...)
        self.affinity = "rbf"
```

Minimal fix: keep the effective affinity in a local variable (or a fitted attribute) instead of overwriting the parameter. Better fix: drop the fallback and call `kneighbors_graph` on the sparse input directly; this changes the result for sparse inputs, which would get the documented nearest-neighbors affinity instead of rbf.

## Versions

```
sklearn: 1.10.dev0, commit 289c4d5cfa (upstream/main)
python: 3.13, numpy: 2.5.3, scipy: 1.18.1, platform: linux
```
