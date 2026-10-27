# `safe_sparse_dot(..., dense_output=True)` fails on two sparse matrices with float32 and float64 dtypes

```json
{
    "estimators": ["safe_sparse_dot"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 1,
    "difficulty": 1,
    "decision": 0,
    "decision_tags": []
}
```

## Classification

- **Priority (1)**: regression in 1.8 of a "private-public" helper used by downstream projects; a loud error with no wrong result. I found no scikit-learn estimator reaching it with mixed dtypes (the pairwise functions cast both inputs to a common dtype first).
- **Difficulty (1)**: add `a.dtype == b.dtype` to the condition of the fast path.
- **Decision (0)**: no trade-off, mixed dtypes just go through the previous `(a @ b).toarray()` path.

## Upstream search

Queries: "safe_sparse_dot dense_output dtype", "sparse_matmul_to_dense dtype". The fast path was added in #31952 (merged, 2025, "ENH add sparse_matmul_to_dense"), released in 1.8.0. Before that, `safe_sparse_dot` returned `(a @ b).toarray()` for any dtypes.

## Describe the bug and its user-facing impact

`safe_sparse_dot(a, b, dense_output=True)` with `a` and `b` both CSR/CSC dispatches to `sparse_matmul_to_dense` as soon as each dtype is float32 or float64, but that helper requires both dtypes to be equal. A float32 sparse matrix times a float64 one raises a `ValueError` instead of returning the float64 product, while `dense_output=False` and `a @ b` work.

## Steps/Code to Reproduce

```python
import numpy as np
from scipy import sparse
from sklearn.utils.extmath import safe_sparse_dot

a = sparse.random_array((5, 4), density=0.5, format="csr", dtype=np.float32, rng=0)
b = sparse.random_array((4, 3), density=0.5, format="csr", dtype=np.float64, rng=0)

print(safe_sparse_dot(a, b).toarray().dtype)  # dense_output=False works
print(safe_sparse_dot(a, b, dense_output=True))
```

## Expected Results

A dense float64 array equal to `(a @ b).toarray()`, as in scikit-learn 1.7.

## Actual Results

```
float64
Traceback (most recent call last):
  File "repro_ssd.py", line 9, in <module>
    print(safe_sparse_dot(a, b, dense_output=True))
          ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/arthur/scikit-learn/sklearn/utils/extmath.py", line 226, in safe_sparse_dot
    return sparse_matmul_to_dense(a, b)
  File "/home/arthur/scikit-learn/sklearn/utils/sparsefuncs.py", line 774, in sparse_matmul_to_dense
    raise ValueError(msg)
ValueError: Dtype of A and B must be the same, either both float32 or float64.
```

## Root cause and possible fix

`sklearn/utils/extmath.py:216-226` (upstream/main 289c4d5cfa) checks that each dtype is a float type, while `sparse_matmul_to_dense` (`sklearn/utils/sparsefuncs.py:772-774`) requires `A.dtype == B.dtype`. Replace the two dtype conditions with `a.dtype == b.dtype and a.dtype in (np.float32, np.float64)`, or cast the float32 operand to float64 before calling the fast path.

## Versions

```
sklearn: 1.10.dev0, commit 289c4d5cfa (upstream/main; local build f570497251, no diff on the files involved)
python: 3.13, numpy: 2.5.3, scipy: 1.18.1, platform: linux
```
