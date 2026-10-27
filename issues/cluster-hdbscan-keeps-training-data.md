# Fitted HDBSCAN keeps a float64 copy of the training data, only to read its number of rows

```json
{
    "estimators": ["HDBSCAN"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 1,
    "difficulty": 1,
    "decision": 0,
    "decision_tags": []
}
```

## Classification

- **Priority (1)**: every fitted `HDBSCAN` (non-precomputed metric) holds the validated training data, so pickles are ~10x larger than needed for 50 features, and with float32, pandas or non-contiguous input it also keeps a private float64 copy alive in memory. No wrong results, and `HDBSCAN` has no `predict`, so fitted models are less often persisted than for other estimators.
- **Difficulty (1)**: replace `self._raw_data = X` by a local variable (or `n_samples`); it is only used for `.shape[0]` inside `fit`.
- **Decision (0)**: `_raw_data` is private and unused after `fit`.

## Upstream search

Queries: "HDBSCAN _raw_data" (rate limited), "HDBSCAN memory", "HDBSCAN pickle" (no results), "HDBSCAN" (40 results). Related but different:

- #31503 / #33364 / #27572 (open): MST algorithm speed and memory, not the stored data.
- #34947 (open): validate dense precomputed distances in row blocks. Does not touch `_raw_data`.
- #26888 (open, float32 support): keeps `self._raw_data = X` unchanged.

## Describe the bug and its user-facing impact

`HDBSCAN.fit` stores the validated input as `self._raw_data` and never deletes it. It is only used to read `self._raw_data.shape[0]` when remapping labels for non-finite samples, at the end of the same `fit` call.

As a result:

- the pickled model contains the whole training set: ~2x the size of a float32 `X`, ~10x the size of what the model needs (`labels_`, `probabilities_`, `_single_linkage_tree_`) for 50 features;
- `validate_data` converts to float64, so for float32 input (or a DataFrame, or a non-contiguous array) the fitted estimator keeps a private float64 copy of `X` alive for its whole lifetime.

## Steps/Code to Reproduce

```python
import pickle

import numpy as np
from sklearn.cluster import HDBSCAN
from sklearn.datasets import make_blobs

for n_features in [10, 50, 100]:
    X, _ = make_blobs(n_samples=5_000, n_features=n_features, random_state=0)
    X = X.astype(np.float32)
    hdb = HDBSCAN(copy=True).fit(X)
    print(
        f"n_features={n_features:3d}: X is {X.nbytes / 2**20:.1f} MiB, "
        f"pickled HDBSCAN is {len(pickle.dumps(hdb)) / 2**20:.1f} MiB, "
        f"of which hdb._raw_data ({hdb._raw_data.dtype}) is {hdb._raw_data.nbytes / 2**20:.1f} MiB"
    )
```

## Expected Results

A pickle size independent of `n_features` (~0.2 MiB here: per-sample labels, probabilities and the single linkage tree), and no reference to the training data after `fit`.

## Actual Results

```
n_features= 10: X is 0.2 MiB, pickled HDBSCAN is 0.6 MiB, of which hdb._raw_data (float64) is 0.4 MiB
n_features= 50: X is 1.0 MiB, pickled HDBSCAN is 2.1 MiB, of which hdb._raw_data (float64) is 1.9 MiB
n_features=100: X is 1.9 MiB, pickled HDBSCAN is 4.0 MiB, of which hdb._raw_data (float64) is 3.8 MiB
```

## Root cause and possible fix

`sklearn/cluster/_hdbscan/hdbscan.py:746` (commit below) sets `self._raw_data = X`; the only reads are `self._raw_data.shape[0]` at lines 884 and 890, in the same `fit` call. `git grep _raw_data sklearn/cluster` shows no other use.

Fix: keep `n_samples = X.shape[0]` in a local variable (before `X = X[finite_index]`) and use it at lines 884 and 890.

## Versions

```
sklearn: 1.10.dev0, commit 289c4d5cfa (upstream/main; local checkout f570497251, no diff on this file)
python: 3.13.14, numpy: 2.5.3, scipy: 1.18.1, platform: linux
```

See also: `issues/cluster-hdbscan-store-centers-non-finite.md` (`store_centers` with non-finite rows currently needs the raw data or `finite_index`).
