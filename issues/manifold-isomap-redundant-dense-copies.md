# Isomap.fit peaks at 3 (n_samples, n_samples) matrices and the fitted model keeps 2 of them

```json
{
    "estimators": ["Isomap"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 1,
    "difficulty": 2,
    "decision": 1,
    "decision_tags": ["complexity"]
}
```

## Classification

- **Priority (1)**: Isomap is inherently O(n_samples^2) in memory (`dist_matrix_` is needed by `transform`), but `fit` needs 3x that and the fitted model (and its pickle) 2x. For 20k samples: ~9.6 GB peak and a 6.4 GB pickle where ~6.4 GB and ~3.2 GB are enough. No wrong results.
- **Difficulty (2)**: the peak is a one-argument change (`copy_X=False` on the private `KernelPCA`). Not storing the second matrix needs `Isomap.transform` to stop going through `self.kernel_pca_.transform`, or `KernelPCA(kernel="precomputed")` to stop keeping the training kernel.
- **Decision (1)**: the first part is free. The second part either duplicates a few lines of `KernelPCA.transform` in `Isomap` (`complexity`), or changes what `KernelPCA.X_fit_` holds for precomputed kernels.

## Upstream search

Queries: "Isomap memory", "Isomap dist_matrix_" (no results), "Isomap" (25 results), "KernelPCA copy_X", "KernelPCA copy". Related but different:

- #31246 (open issue) / #31247 (open PR): faster (randomized) eigendecomposition for Isomap and KernelPCA. It adds an `eigen_solver` option and does not change how `G` is copied or stored.
- #29100 (merged): in-place centering in `KernelPCA.fit` when `copy_X=False`, which is what makes the first part of the fix possible.

No issue about the extra copies.

## Describe the bug and its user-facing impact

`Isomap.fit` computes the geodesic distances `dist_matrix_` (n x n), then `G = -0.5 * dist_matrix_**2` (n x n), then calls `self.kernel_pca_.fit_transform(G)` on a `KernelPCA(kernel="precomputed")` created with the default `copy_X=True`. `KernelPCA.fit` copies `G` (third n x n matrix), centers the copy in place and stores it as `kernel_pca_.X_fit_`.

So:

- `fit` peaks at 3 dense n x n matrices, although `G` is a private temporary that `KernelPCA` could own and center in place;
- the fitted `Isomap` keeps both `dist_matrix_` and `kernel_pca_.X_fit_`. With a precomputed kernel, `KernelPCA.transform` only uses `X_fit_` for a shape check in `pairwise_kernels(X, self.X_fit_, metric="precomputed")`; the centering uses `_centerer.K_fit_rows_` and `K_fit_all_`. Neither `Isomap.transform` nor `Isomap.reconstruction_error` needs it.

Isomap is used on a few thousand to a few tens of thousands of samples, where these n x n matrices are hundreds of MB to GBs each.

## Steps/Code to Reproduce

```python
import pickle
import tracemalloc

from sklearn.datasets import make_swiss_roll
from sklearn.manifold import Isomap

for n_samples in [1_000, 2_000, 4_000]:
    X, _ = make_swiss_roll(n_samples=n_samples, random_state=0)
    n2 = n_samples**2 * 8  # one (n_samples, n_samples) float64 matrix
    tracemalloc.start()
    iso = Isomap(n_neighbors=10, n_components=2).fit(X)
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    print(
        f"n_samples={n_samples}: fit peak {peak / n2:.2f}, pickle {len(pickle.dumps(iso)) / n2:.2f} "
        f"(in units of one n_samples x n_samples float64 matrix); "
        f"dist_matrix_ {iso.dist_matrix_.shape}, kernel_pca_.X_fit_ {iso.kernel_pca_.X_fit_.shape}"
    )
```

## Expected Results

A `fit` peak of ~2 matrices (`dist_matrix_` and `G`, centered in place) and a fitted model of ~1 matrix (`dist_matrix_`).

## Actual Results

```
n_samples=1000: fit peak 3.39, pickle 2.01 (in units of one n_samples x n_samples float64 matrix); dist_matrix_ (1000, 1000), kernel_pca_.X_fit_ (1000, 1000)
n_samples=2000: fit peak 3.04, pickle 2.00 (in units of one n_samples x n_samples float64 matrix); dist_matrix_ (2000, 2000), kernel_pca_.X_fit_ (2000, 2000)
n_samples=4000: fit peak 3.02, pickle 2.00 (in units of one n_samples x n_samples float64 matrix); dist_matrix_ (4000, 4000), kernel_pca_.X_fit_ (4000, 4000)
```

## Root cause and possible fix

- `sklearn/manifold/_isomap.py:296-303` (commit below) builds `KernelPCA(kernel="precomputed", ...)` without `copy_X=False`, so `sklearn/decomposition/_kernel_pca.py:430` copies `G`. Passing `copy_X=False` is safe since `G` is local to `_fit_transform`. I checked it by monkeypatching `KernelPCA` in `sklearn.manifold._isomap` with `functools.partial(KernelPCA, copy_X=False)`: the peak drops from 3.04 to 2.04 matrices at `n_samples=2000`, `embedding_` and `transform` outputs match to 1e-13.
- `sklearn/decomposition/_kernel_pca.py:444` stores `self.X_fit_ = X`, i.e. the centered `G` for Isomap. To keep only `dist_matrix_` in the fitted Isomap, `Isomap.transform` could do the centering and projection itself (`kernel_pca_._centerer.transform(G_X) @ (eigenvectors_ / sqrt(eigenvalues_))`, as in `KernelPCA.transform`) and drop `kernel_pca_.X_fit_` after fit. Alternatively `KernelPCA` could keep only `n_samples` for `kernel="precomputed"`, but `X_fit_` is a documented attribute.

See also `issues/decomposition-kernelpca-copy-x-false-mutates-input.md` for the in-place centering of `X_fit_`.

## Versions

```
sklearn: 1.10.dev0, commit 289c4d5cfa (upstream/main; local checkout f570497251, no diff on these files)
python: 3.13.14, numpy: 2.5.3, scipy: 1.18.1, platform: linux
```
