# `LocallyLinearEmbedding` accepts `n_neighbors <= n_components` for `method="modified"` and `"ltsa"` and returns a meaningless embedding

```json
{
    "estimators": ["LocallyLinearEmbedding", "locally_linear_embedding"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 1,
    "difficulty": 1,
    "decision": 1,
    "decision_tags": ["new-error"]
}
```

## Classification

- **Priority (1)**: invalid parameter combinations, unlikely in practice (a neighborhood smaller than the target dimension). But the output is silently garbage (or an unrelated broadcasting error) instead of a clear message, unlike `method="hessian"` which validates its requirement.
- **Difficulty (1)**: change one comparison for MLLE and add the same check for LTSA.
- **Decision (1)**: `new-error`: calls that currently return a (meaningless) embedding would raise.

## Upstream search

Queries: "LocallyLinearEmbedding n_neighbors n_components ltsa", "modified LLE requires n_neighbors", "ltsa could not broadcast". Only #24353 (closed, MLLE example `modified_tol`), unrelated.

## Describe the bug and its user-facing impact

The user guide says MLLE "requires `n_neighbors > n_components`", and LTSA needs at least `n_components + 1` neighbors to estimate a `n_components`-dimensional tangent space. But:

- `method="modified"` only checks `n_neighbors < n_components`, so `n_neighbors == n_components` passes. Then the "almost null space" of each neighborhood is empty (`s_i = 0`), `alpha_i` is `0 / 0` (300 `RuntimeWarning: invalid value encountered in scalar divide` for 300 samples), the weight matrix `M` is all zeros and the returned embedding is an arbitrary basis of R^n (`reconstruction_error_ = 0`).
- `method="ltsa"` has no check at all. With `n_neighbors <= n_components` it silently returns an embedding from a non-PSD `M` (negative `reconstruction_error_`), or fails with `could not broadcast input array from shape (2,2) into shape (2,3)` when `n_neighbors < n_components <= n_features`.

## Steps/Code to Reproduce

```python
import warnings
from sklearn.datasets import make_swiss_roll
from sklearn.manifold import LocallyLinearEmbedding

X, _ = make_swiss_roll(n_samples=300, random_state=0)
warnings.simplefilter("ignore", RuntimeWarning)
for method, n_neighbors in [("modified", 2), ("ltsa", 2), ("ltsa", 1)]:
    lle = LocallyLinearEmbedding(
        n_neighbors=n_neighbors, n_components=2, method=method, eigen_solver="dense"
    )
    try:
        lle.fit(X)
        print(f"{method}, n_neighbors={n_neighbors}: reconstruction_error_ = "
              f"{lle.reconstruction_error_:.3g}")
    except ValueError as e:
        print(f"{method}, n_neighbors={n_neighbors}: ValueError: {e}")

X5 = make_swiss_roll(n_samples=300, random_state=0)[0] @ [[1, 0, 0, 1, 0], [0, 1, 0, 0, 1], [0, 0, 1, 1, 1]]
try:
    LocallyLinearEmbedding(n_neighbors=2, n_components=3, method="ltsa").fit(X5)
except ValueError as e:
    print(f"ltsa, n_neighbors=2, n_components=3: ValueError: {e}")
```

## Expected Results

A `ValueError` saying that `method="modified"` and `method="ltsa"` require `n_neighbors > n_components`, as already done for `method="hessian"` (`n_neighbors > n_components * (n_components + 3) / 2`).

## Actual Results

```
modified, n_neighbors=2: reconstruction_error_ = 0
ltsa, n_neighbors=2: reconstruction_error_ = -7.18
ltsa, n_neighbors=1: reconstruction_error_ = -12
ltsa, n_neighbors=2, n_components=3: ValueError: could not broadcast input array from shape (2,2) into shape (2,3)
```

## Root cause and possible fix

- `sklearn/manifold/_locally_linear.py:302`: `if n_neighbors < n_components:` should be `<=` (the message already says "requires n_neighbors >= n_components", it should say `>`).
- `sklearn/manifold/_locally_linear.py:408`: the `ltsa` branch has no validation; add the same `n_neighbors <= n_components` check.

Both turn silently wrong outputs into errors, hence the `new-error` tag.

## Versions

```
sklearn: 1.10.dev0, commit 289c4d5cfa
python: 3.13.14, numpy: 2.5.3, scipy: 1.18.1, platform: linux
```
