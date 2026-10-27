# sparse_encode, SparseCoder and dictionary learning transforms return float64 for float32 input when n_jobs > 1

```json
{
    "estimators": ["sparse_encode", "SparseCoder", "DictionaryLearning", "MiniBatchDictionaryLearning"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 1,
    "difficulty": 1,
    "decision": 0,
    "decision_tags": []
}
```

## Classification

- **Priority (1)**: the values are right, only the output dtype changes with `n_jobs` (2x memory, and downstream float32 pipelines silently switch to float64). The estimators declare `preserves_dtype = ["float64", "float32"]`, but the common test runs with the default `n_jobs=None`.
- **Difficulty (1)**: allocate the output with `dtype=X.dtype`.
- **Decision (0)**: no trade-off.

## Upstream search

- `"SparseCoder n_jobs"`, `"sparse_encode dtype"`: blocked by the GitHub secondary rate limit (shared by 20 agents), to be rerun before opening upstream. `git log -S` shows the `np.empty((n_samples, n_components))` allocation is unchanged since the `n_atoms` -> `n_components` rename, long before float32 support was added.

## Describe the bug and its user-facing impact

`_sparse_encode` computes the code in a single call when `effective_n_jobs(n_jobs) == 1` (keeping the input dtype), and otherwise splits the samples across joblib workers and gathers the pieces into a buffer allocated with `np.empty((n_samples, n_components))`, i.e. float64. So `transform` of `SparseCoder`, `DictionaryLearning` and `MiniBatchDictionaryLearning`, and `sparse_encode`, return float64 for float32 input as soon as `n_jobs > 1`, for all algorithms except `"threshold"` (which never goes through the parallel path).

## Steps/Code to Reproduce

```python
import numpy as np
from sklearn.decomposition import SparseCoder

rng = np.random.RandomState(0)
dictionary = rng.randn(5, 8).astype(np.float32)
X = rng.randn(10, 8).astype(np.float32)

for n_jobs in [1, 2]:
    coder = SparseCoder(dictionary, transform_algorithm="omp", n_jobs=n_jobs)
    print(f"n_jobs={n_jobs}: {coder.transform(X).dtype}")
```

## Expected Results

`float32` for both, as declared by the `preserves_dtype` tag.

## Actual Results

```
n_jobs=1: float32
n_jobs=2: float64
```

## Root cause and possible fix

`sklearn/decomposition/_dict_learning.py:443` (commit 289c4d5cfa), in `_sparse_encode`:

```python
code = np.empty((n_samples, n_components))
```

Use `np.empty((n_samples, n_components), dtype=X.dtype)`.

## Versions

```
sklearn: 1.10.dev0, commit 289c4d5cfa (upstream/main)
python: 3.13, numpy: 2.5.3, scipy: 1.18.1, platform: linux
```
