# AdditiveChi2Sampler returns NaN for explicitly stored zeros in sparse input

```json
{
    "estimators": ["AdditiveChi2Sampler"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 1,
    "difficulty": 1,
    "decision": 0,
    "decision_tags": []
}
```

## Classification

- **Priority (1)**: silent NaN in the output (only RuntimeWarnings), but it needs sparse input with explicitly stored zeros, which is uncommon (it happens after some sparse arithmetic or in-place edits of `.data`).
- **Difficulty (1)**: mask the zeros in `_transform_sparse` like `_transform_dense` does, or work on a copy with `eliminate_zeros()`.
- **Decision (0)**: no trade-off.

## Upstream search

Queries: "AdditiveChi2Sampler", "AdditiveChi2Sampler sparse", "AdditiveChi2Sampler nan", "AdditiveChi2Sampler sparse zeros" (issues and PRs).

No match. The issue tracker only has maintenance PRs on this class (parameter validation, statelessness, `get_feature_names_out`). Other agents filed the same class of bug for other code (`issues/feature_extraction-tfidf-explicit-zeros.md`, `issues/metrics-canberra-csr-explicit-zeros.md`), with different root causes.

## Describe the bug and its user-facing impact

`AdditiveChi2Sampler._transform_dense` only transforms the non-zero entries (`non_zero = X != 0.0`), because the feature map involves `log(x)`. `_transform_sparse` assumes every stored value is non-zero and applies `np.log(X.data)` directly. For an explicitly stored zero, `log(0) = -inf`, then `0 * cos(-inf) = nan`, so the cos/sin components of that entry are NaN instead of 0. The dense and sparse paths disagree on the same matrix, and NaNs propagate to any downstream model.

## Steps/Code to Reproduce

```python
import numpy as np
import scipy.sparse as sp
from sklearn.kernel_approximation import AdditiveChi2Sampler

X = sp.csr_array((np.array([1.0, 0.0, 2.0]), np.array([0, 1, 1]), np.array([0, 2, 3])), shape=(2, 2))
print("stored data:", X.data, "(explicit zero at row 0, col 1)")
dense = AdditiveChi2Sampler().fit_transform(X.toarray())
sparse = AdditiveChi2Sampler().fit_transform(X).toarray()
print("dense input :\n", dense)
print("sparse input:\n", sparse)
```

## Expected Results

The same output for the sparse and dense inputs (zeros for the zero entry).

## Actual Results

```
/home/arthur/scikit-learn/sklearn/kernel_approximation.py:811: RuntimeWarning: divide by zero encountered in log
  log_step_nz = sample_interval * np.log(X.data)
/home/arthur/scikit-learn/sklearn/kernel_approximation.py:817: RuntimeWarning: invalid value encountered in cos
  data_step = factor_nz * np.cos(j * log_step_nz)
/home/arthur/scikit-learn/sklearn/kernel_approximation.py:828: RuntimeWarning: invalid value encountered in sin
  data_step = factor_nz * np.sin(j * log_step_nz)
stored data: [1. 0. 2.] (explicit zero at row 0, col 1)
dense input :
 [[0.70710678 0.         0.63129772 0.         0.         0.        ]
 [0.         1.         0.         0.8397064  0.         0.30326027]]
sparse input:
 [[0.70710678 0.         0.63129772        nan 0.                nan]
 [0.         1.         0.         0.8397064  0.         0.30326027]]
```

## Root cause and possible fix

`AdditiveChi2Sampler._transform_sparse`, `sklearn/kernel_approximation.py:811` on the commit below:

```python
log_step_nz = sample_interval * np.log(X.data)
```

Fix: start `_transform_sparse` with `X = X.copy(); X.eliminate_zeros()` (`transform` already validated `X`, which may return the user's matrix, so copy before mutating), or compute the log only where `X.data != 0` and set the other entries to 0.

## Versions

```
sklearn: 1.10.dev0, commit 289c4d5cfa (upstream/main; ran on f570497251, no diff in the files involved)
python: 3.13.14, numpy: 2.5.3, scipy: 1.18.1, platform: linux
```
