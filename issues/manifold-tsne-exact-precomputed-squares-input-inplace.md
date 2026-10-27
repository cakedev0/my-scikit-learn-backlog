# TSNE(method="exact", metric="precomputed") squares the user's distance matrix in place

```json
{
    "estimators": ["TSNE"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 2,
    "difficulty": 1,
    "decision": 0,
    "decision_tags": []
}
```

## Classification

- **Priority (2)**: silent corruption of user data: after `fit`, the user's distance matrix holds squared distances, so any later use of it (another `fit`, a clustering with an `eps`, a plot) is wrong. Needs `method="exact"` (not the default) and a dense float32/float64 precomputed matrix.
- **Difficulty (1)**: square a copy instead of the input.
- **Decision (0)**: no trade-off beyond one `(n_samples, n_samples)` copy, which the exact method already allocates several times.

## Upstream search

- `"TSNE precomputed modifies"`, `"TSNE precomputed exact"`: no match. `git log -L` shows the in-place squaring has been there since #17662 / #21592 (squaring of non-Euclidean distances), and the `distances = X` alias since the original exact implementation.

## Describe the bug and its user-facing impact

With `method="exact"` and `metric="precomputed"`, `TSNE._fit` sets `distances = X` (no copy, since `validate_data` is called with the default `copy=False`), then does `distances **= 2` because the metric is not `"euclidean"`. The user's matrix is squared in place.

Consequences: the input array is silently modified, and fitting the same estimator twice on the same variable gives different embeddings (the second fit sees distances to the 4th power). The `barnes_hut` method is not affected (it goes through `kneighbors_graph`, which allocates a new matrix), nor is `method="exact"` with a non-precomputed metric (`pairwise_distances` returns a new array).

## Steps/Code to Reproduce

```python
import numpy as np
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances

rng = np.random.RandomState(0)
D = pairwise_distances(rng.randn(20, 3))
D_orig = D.copy()

tsne = TSNE(method="exact", metric="precomputed", init="random",
            perplexity=5, random_state=0)
E1 = tsne.fit_transform(D)
print("D was squared in place:", np.allclose(D, D_orig**2))
E2 = tsne.fit_transform(D)  # same estimator, same input variable
print("second fit gives the same embedding:", np.allclose(E1, E2))
```

## Expected Results

```
D was squared in place: False
second fit gives the same embedding: True
```

## Actual Results

```
D was squared in place: True
second fit gives the same embedding: False
```

## Root cause and possible fix

`sklearn/manifold/_t_sne.py` (commit 289c4d5cfa):

```python
if self.metric == "precomputed":
    distances = X                      # line 903, no copy
...
if self.metric != "euclidean":
    distances **= 2                    # line 927, in place
```

Use `distances = distances**2` (or copy `X` in the precomputed branch). `_joint_probabilities` does not need to write to `distances`, so a single out-of-place squaring is enough.

## Versions

```
sklearn: 1.10.dev0, commit 289c4d5cfa (upstream/main)
python: 3.13, numpy: 2.5.3, scipy: 1.18.1, platform: linux
```
