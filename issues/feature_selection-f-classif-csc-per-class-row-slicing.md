# `f_classif` on a CSC matrix is O(n_classes * nnz): ~20x slower than on the same matrix in CSR

```json
{
    "estimators": ["f_classif", "SelectKBest", "SelectPercentile", "GenericUnivariateSelect"],
    "existing_issue": "https://github.com/scikit-learn/scikit-learn/issues/1651",
    "reproduced": true,
    "priority": 1,
    "difficulty": 1,
    "decision": 0,
    "decision_tags": []
}
```

## Classification

- **Priority (1)**: only CSC input with many classes is affected (most scikit-learn transformers and vectorizers output CSR), and converting to CSR is a cheap workaround. But the slowdown grows with the number of classes and nothing tells the user.
- **Difficulty (1)**: the minimal fix is `accept_sparse="csr"` in `f_classif`'s `check_X_y` (one O(nnz) conversion). The better fix of #1651 (per-class sums without copies) is a bigger refactoring.
- **Decision (0)**: same F-values, one format conversion at most.

## Upstream search

- "f_classif sparse slow", "f_classif csc" (GitHub issue search).
- #1651 (open, labels Performance, help wanted) "Memory scalability issues in univariate feature selection": same root cause, the per-class copies `args = [X[safe_mask(X, y == k)] for k in np.unique(y)]`. It discusses memory; amueller suggested converting all sparse input to CSR there. This file adds a time symptom: with CSC input each class selection scans the whole matrix. Fixing #1651 fixes this too, hence `existing_issue`.
- #12468, #13244 (unrelated feature requests).

## Describe the bug and its user-facing impact

`f_classif` accepts CSR, CSC and COO, then builds one row subset per class with `X[safe_mask(X, y == k)]`. Row selection on a CSR matrix only touches the selected rows, but on a CSC matrix SciPy has to go through all stored values, so the total cost is O(n_classes * nnz) instead of O(nnz).

On a 200,000 x 2,000 matrix with 1% density, `f_classif` takes ~0.3 s in CSR and ~5.6 s in CSC with 400 classes (~1.9 s vs ~0.16 s with 100 classes). This also applies to `SelectKBest(f_classif)` (the default `score_func`) and `SelectPercentile` when the input arrives in CSC, e.g. from a user-built matrix or a custom transformer, and in every CV split.

## Steps/Code to Reproduce

```python
from time import perf_counter

import numpy as np
import scipy.sparse as sp
from sklearn.feature_selection import f_classif

rng = np.random.RandomState(0)
X = sp.random(200_000, 2_000, density=0.01, format="csr", random_state=0)
for n_classes in [10, 100, 400]:
    y = rng.randint(n_classes, size=X.shape[0])
    X_csc = X.tocsc()
    t0 = perf_counter()
    F_csr, _ = f_classif(X, y)
    t1 = perf_counter()
    F_csc, _ = f_classif(X_csc, y)
    t2 = perf_counter()
    np.testing.assert_allclose(F_csr, F_csc)
    print(f"n_classes={n_classes}: CSR {t1 - t0:.2f}s, CSC {t2 - t1:.2f}s")
```

## Expected Results

Similar timings for CSR and CSC (a CSC to CSR conversion is O(nnz)), and a cost that doesn't grow with `n_classes * nnz`.

## Actual Results

Timings from a laptop loaded by other jobs:

```
n_classes=10: CSR 0.15s, CSC 0.29s
n_classes=100: CSR 0.16s, CSC 1.89s
n_classes=400: CSR 0.27s, CSC 5.57s
```

## Root cause and possible fix

`sklearn/feature_selection/_univariate_selection.py:169-170` (on 289c4d5cfa):

```python
X, y = check_X_y(X, y, accept_sparse=["csr", "csc", "coo"])
args = [X[safe_mask(X, y == k)] for k in np.unique(y)]
```

Minimal fix: `accept_sparse="csr"`, so CSC (and COO) input is converted once. The fix discussed in #1651, computing the per-class sums and sums of squares with a one-hot class indicator product (`Y.T @ X`, as `chi2` does) instead of per-class copies, removes both the memory and the format issue.

## Versions

```
sklearn: 1.10.dev0, commit 289c4d5cfa (upstream/main; checkout at f570497251, no diff in sklearn/feature_selection/_univariate_selection.py)
python: 3.13, numpy: 2.5.3, scipy: 1.18.1, platform: linux
```
