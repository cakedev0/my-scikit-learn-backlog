# Non-metric MDS does not treat zero dissimilarities as missing values, as documented: it pulls those pairs together

```json
{
    "estimators": ["MDS", "smacof"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 1,
    "difficulty": 2,
    "decision": 1,
    "decision_tags": ["output-change", "complexity"]
}
```

## Classification

- **Priority (1)**: silently distorted embedding for users relying on the documented missing-value handling, but it needs non-metric MDS (a minority of MDS uses) with off-diagonal zeros meant as missing, and a docstring fix removes the false promise (priority review, 2026-10-23: was 2)
- **Difficulty (2)**: a real fix needs weighted SMACOF (weights 0 for missing pairs, Guttman transform with the pseudo-inverse of the weighted `V` matrix instead of `1 / n_samples`, weighted stress). A doc-only fix is one line in three docstrings.
- **Decision (1)**: `output-change`: proper missing-value handling changes the embedding for every non-metric fit with off-diagonal zeros (including duplicate points). `complexity`: the weighted Guttman transform needs an `(n_samples, n_samples)` pseudo-inverse, so maintainers may prefer to just fix the docstring and keep weights for the open feature request #6828.

## Upstream search

- `"nonmetric MDS"`: #30514 (merged, 2025, "Fix multiple severe bugs in non-metric MDS") fixed the disparities symmetrization, the order of the isotonic regression and the stress, but not this. #27028 (closed by #30514). #30240 (closed, docs on Kruskal stress).
- #6828 (open, 2016, "Weighting the Smacof MDS algorithm") and #18413 (closed, stalled, "Weighting option in MDS") are the feature request for a weight matrix; they mention that missing distances need weights, but nobody points out that the current docstring already claims zeros are missing values.
- `"MDS missing values"`, `"MDS zero dissimilarities"`, `"smacof weights missing"`: no other match.

## Describe the bug and its user-facing impact

The docstrings of `MDS(metric_mds=...)`, `smacof(metric=...)` and `_smacof_single` say: "When `False` (i.e. non-metric MDS), dissimilarities with 0 are considered as missing values."

In the code, zeros are only excluded from the isotonic regression. The resulting disparity matrix has 0 at those positions, and both the Guttman transform (`B = -disparities / distances`) and the stress use these zeros as targets. So "missing" pairs are treated as "these two points coincide" and are pulled together, and the reported `stress_` includes them.

In the reproducer, 100 out of 435 pairs are marked missing; the 335 remaining pairs are more than enough to pin down a 2D configuration, so a correct missing-value handling should recover the same embedding as with all pairs. Instead the missing pairs end up at about half their true relative distance.

## Steps/Code to Reproduce

```python
import numpy as np
from sklearn.manifold import MDS
from sklearn.metrics import euclidean_distances

rng = np.random.RandomState(0)
X = rng.rand(30, 2)
D = euclidean_distances(X)

# Mark the 100 dissimilarities between points 0-9 and 20-29 as missing (0).
# The remaining 335 pairs still pin down the 2D configuration.
missing = np.zeros_like(D, dtype=bool)
missing[:10, 20:] = missing[20:, :10] = True
D_missing = np.where(missing, 0, D)
other = ~missing & ~np.eye(30, dtype=bool)

for name, dissim in [("all pairs", D), ("some missing", D_missing)]:
    mds = MDS(metric_mds=False, metric="precomputed", init="random",
              n_init=4, max_iter=1000, eps=1e-9, random_state=0)
    E = euclidean_distances(mds.fit_transform(dissim))
    # distances are only defined up to scale in non-metric MDS: compare ratios
    ratio = E[missing].mean() / E[other].mean()
    print(f"{name:13s} mean dist(missing pairs) / mean dist(other pairs) = {ratio:.3f}")

print(f"{'truth':13s} mean dist(missing pairs) / mean dist(other pairs) = "
      f"{D[missing].mean() / D[other].mean():.3f}")
```

## Expected Results

The "some missing" ratio close to the truth (1.138), as for "all pairs", since the missing dissimilarities should be ignored.

## Actual Results

```
all pairs     mean dist(missing pairs) / mean dist(other pairs) = 1.138
some missing  mean dist(missing pairs) / mean dist(other pairs) = 0.503
truth         mean dist(missing pairs) / mean dist(other pairs) = 1.138
```

## Root cause and possible fix

`sklearn/manifold/_mds.py` (commit 289c4d5cfa), `_smacof_single`:

- line 162: `disparities = np.zeros_like(distances_flat)`, then only the non-zero positions are filled from the isotonic regression, so missing pairs keep a target of 0;
- lines 172-175: the Guttman transform `X = 1 / n_samples * B @ X` is the unweighted one, valid only when all pairs have weight 1;
- line 179: the stress sums over all pairs, including missing ones.

Two options:

1. Implement the weighted SMACOF update for non-metric MDS (weight 0 for zero dissimilarities): `B` built from `w_ij * disparities_ij / distances_ij`, `X = pinv(V) @ B @ X` with `V = diag(W @ 1) - W`, and a weighted stress. This is also what #6828 asks for.
2. Remove the "considered as missing values" sentence from the three docstrings and say that zero dissimilarities are only excluded from the monotonic regression.

## Versions

```
sklearn: 1.10.dev0, commit 289c4d5cfa (upstream/main)
python: 3.13, numpy: 2.5.3, scipy: 1.18.1, platform: linux
```
