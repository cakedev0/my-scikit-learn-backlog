# `MDS` / `smacof` give different results for `n_jobs=1` and `n_jobs>1` with the same `random_state`

```json
{
    "estimators": ["MDS", "smacof"],
    "existing_issue": "https://github.com/scikit-learn/scikit-learn/issues/10119",
    "reproduced": true,
    "priority": 1,
    "difficulty": 1,
    "decision": 1,
    "decision_tags": ["rng-order", "output-change"]
}
```

## Classification

- **Priority (1)**: only with `init="random"` and `n_init > 1`, and both results are valid embeddings (best of `n_init` runs). But a parallelism setting should not change the output for a fixed `random_state`.
- **Difficulty (1)**: draw the per-init seeds in both code paths.
- **Decision (1)**: changes the `n_jobs=1` results for a given `random_state` (rng-order, output-change).

## Upstream search

Queries (issues and PRs): "MDS n_jobs random_state", "smacof n_jobs", "LatentDirichletAllocation n_jobs".

- #10119 (open since 2017): "Improve test for MDS parallel processing case - requires update in MDS". It asks for a test that `n_jobs > 1` gives results comparable to `n_jobs=1`, and the title says MDS itself needs a change for that. It does not spell out the cause; this report does.
- #1178 (closed, 2012): `n_jobs` was not used at all, unrelated.

## Describe the bug and its user-facing impact

`smacof` (and so `MDS.fit`) runs `n_init` SMACOF runs and keeps the best one. The sequential path passes the same `RandomState` object to every run, so they draw from one stream. The parallel path first draws `n_init` integer seeds from it and gives one seed per run. The initial positions differ between the two paths, so the selected embedding and `stress_` depend on whether `n_jobs` is 1 or more.

## Steps/Code to Reproduce

```python
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.manifold import MDS

X, _ = make_blobs(n_samples=30, n_features=4, random_state=0)
emb = {}
for n_jobs in [1, 2]:
    mds = MDS(n_init=4, init="random", n_jobs=n_jobs, random_state=0)
    emb[n_jobs] = mds.fit_transform(X)
    print(f"n_jobs={n_jobs}: stress_={mds.stress_:.6f}")
print("max abs diff of embeddings:", np.abs(emb[1] - emb[2]).max())
```

## Expected Results

The same embedding and `stress_` for `n_jobs=1` and `n_jobs=2`.

## Actual Results

```
n_jobs=1: stress_=74.482576
n_jobs=2: stress_=74.546716
max abs diff of embeddings: 11.45804429660495
```

## Root cause and possible fix

`sklearn/manifold/_mds.py:383-397`: the `effective_n_jobs(n_jobs) == 1` branch calls `_smacof_single(..., random_state=random_state)` in a loop, while the parallel branch (`_mds.py:400-414`) draws `seeds = random_state.randint(np.iinfo(np.int32).max, size=n_init)` and passes `random_state=seed`. Drawing the seeds before the branch and using them in both paths (or dropping the sequential special case and always going through `Parallel`) makes the result independent of `n_jobs`. It changes the `n_jobs=1` output for a given `random_state`.

## Versions

```
sklearn: 1.10.dev0, commit f570497251 (_mds.py identical to upstream/main 289c4d5cfa)
python: 3.13.14, numpy: 2.5.3, scipy: 1.18.1, joblib: 1.6.0, platform: linux
```
