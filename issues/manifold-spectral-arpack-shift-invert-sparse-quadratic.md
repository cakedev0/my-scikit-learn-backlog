# SpectralClustering(affinity="nearest_neighbors") uses quadratic memory: ARPACK shift-invert factorizes the sparse Laplacian

```json
{
    "estimators": ["SpectralClustering", "SpectralEmbedding", "spectral_embedding", "spectral_clustering"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 2,
    "difficulty": 2,
    "decision": 1,
    "decision_tags": ["algo-change", "output-change", "perf-medium"]
}
```

## Classification

- **Priority (2)**: with the default `eigen_solver=None` (ARPACK), a sparse k-NN affinity graph with O(n_samples * n_neighbors) entries needs O(n_samples^2) memory and super-linear time: 295 MiB and 10.6 s for 8k samples, ~1 GiB and ~74 s for 16k, vs 15-24 MiB and ~1 s with `eigen_solver="lobpcg"`. This is the configuration users pick precisely to scale. Results are correct and `eigen_solver="lobpcg"`/`"amg"` is a workaround, hence not 3.
- **Difficulty (2)**: the change is local (the `eigsh` call for sparse Laplacians), but needs checking convergence and accuracy on graphs where plain Lanczos converges slowly (many near-zero eigenvalues, e.g. nearly disconnected components).
- **Decision (1)**: dropping shift-invert for sparse input is an `algo-change`; eigenvectors in near-degenerate subspaces can change, and so can cluster labels (`output-change`); some graphs may converge slower without the inversion (`perf-medium` risk). The dense path (`affinity="rbf"`) can keep shift-invert.

## Upstream search

Queries: "SpectralClustering memory", "spectral_embedding arpack slow", "SpectralClustering slow large" (no results), "SpectralClustering" (15 results). Related but different:

- #8834 (open, "Reduce memory usage for spectral clustering"): about the extra dense copy made by the Laplacian of a dense affinity matrix, not about the sparse LU fill-in.
- #8740 (open, "Spectral clustering for large matrix"): a 57k x 57k dense RBF affinity; @lobpcg suggests `eigen_solver="amg"`. Same symptom (large spectral clustering runs out of memory) but on a dense matrix, where O(n_samples^2) is expected.

Related: #33242 / #33262 (merged): changed the call to `eigsh(L, sigma=-1e-5, which="LM")`, timed on a dense `rbf` affinity. It keeps shift-invert, so the sparse factorization is unchanged.

## Describe the bug and its user-facing impact

For the ARPACK solver, `_spectral_embedding` calls `eigsh(laplacian, k=n_components, sigma=-1e-5, which="LM")`. With `sigma` set, scipy runs ARPACK in shift-invert mode and, for a sparse matrix, factorizes `L - sigma * I` with SuperLU. The fill-in of the LU factors of a k-NN graph Laplacian is far from sparse, so memory grows ~4x and time ~7x each time `n_samples` doubles, although the graph only has ~10 neighbors per sample.

`SpectralClustering(affinity="nearest_neighbors")` (and `SpectralEmbedding(affinity="nearest_neighbors", n_neighbors=...)`, `spectral_embedding` on a sparse adjacency) therefore cannot scale past a few tens of thousands of samples with default settings, while LOBPCG, or ARPACK without shift-invert, handle the same graph in a few MiB.

## Steps/Code to Reproduce

```python
import resource
import time

from sklearn.cluster import SpectralClustering
from sklearn.datasets import make_blobs


def max_rss_mib():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # KiB on Linux


baseline = max_rss_mib()
# Runs ordered by increasing peak memory, so that the increase of the max RSS
# over the baseline is the peak of the current run.
for eigen_solver, n_samples in [("lobpcg", 8_000), ("arpack", 2_000), ("arpack", 4_000), ("arpack", 8_000)]:
    X, _ = make_blobs(n_samples=n_samples, n_features=10, centers=5, cluster_std=5.0, random_state=0)
    tic = time.perf_counter()
    SpectralClustering(
        n_clusters=5, affinity="nearest_neighbors", eigen_solver=eigen_solver, random_state=0
    ).fit(X)
    duration = time.perf_counter() - tic
    print(
        f"eigen_solver={eigen_solver}, n_samples={n_samples}: "
        f"peak RSS increase {max_rss_mib() - baseline:4.0f} MiB, {duration:4.1f} s"
    )
```

(RSS rather than `tracemalloc`, since SuperLU allocates with `malloc`.)

## Expected Results

Memory linear in the number of graph edges, as with `eigen_solver="lobpcg"`.

## Actual Results

```
eigen_solver=lobpcg, n_samples=8000: peak RSS increase   15 MiB,  0.5 s
eigen_solver=arpack, n_samples=2000: peak RSS increase   19 MiB,  0.2 s
eigen_solver=arpack, n_samples=4000: peak RSS increase   73 MiB,  1.4 s
eigen_solver=arpack, n_samples=8000: peak RSS increase  295 MiB, 10.6 s
```

In separate processes at `n_samples=16000`: ARPACK 1033 MiB and 73.8 s, LOBPCG 24 MiB and 1.1 s.

## Root cause and possible fix

`sklearn/manifold/_spectral_embedding.py:368-370` (commit below): `eigsh(..., sigma=-1e-5, which="LM")` for sparse Laplacians too.

For sparse input, ARPACK without shift-invert on the shifted operator `2 I - L` (its largest eigenvalues are `2 - λ` for the smallest `λ` of the normalized Laplacian, whose spectrum is in [0, 2]) avoids any factorization. On the 8000-sample k-NN graph above (`kneighbors_graph(X, 10)`, symmetrized, normalized Laplacian, `tol=0`):

```
shift-invert  eigsh(L, k=5, sigma=..., which="LM"):     265 MiB, 11.3 s
plain Lanczos eigsh(2 I - L, k=5, which="LA"):            3 MiB,  0.0 s
```

with the same eigenvalues (`[0, 8.21239734e-03, 2.73731239e-02, 4.47197591e-02, 7.08143839e-02]`). Lanczos can converge slowly when many eigenvalues are clustered near 0, which is presumably why shift-invert was chosen, so this needs a benchmark on harder graphs. Another option is to default to `"lobpcg"` for sparse Laplacians.

## Versions

```
sklearn: 1.10.dev0, commit 289c4d5cfa (upstream/main; local checkout f570497251, no diff in sklearn/manifold/_spectral_embedding.py and sklearn/cluster/_spectral.py)
python: 3.13.14, numpy: 2.5.3, scipy: 1.18.1, platform: linux
```
