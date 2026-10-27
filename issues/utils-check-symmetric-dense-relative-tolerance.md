# `check_symmetric` uses a relative tolerance of 1e-5 for dense arrays, ignoring `tol`

```json
{
    "estimators": ["check_symmetric", "spectral_embedding", "SpectralEmbedding", "MDS", "smacof"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 1,
    "difficulty": 1,
    "decision": 1,
    "decision_tags": ["unclear-spec", "output-change", "new-warning", "new-error"]
}
```

## Classification

- **Priority (1)**: the public helper does not do what its docstring says and is inconsistent between dense and sparse input, but the asymmetries it lets through are small (relative 1e-5), so the downstream numerical impact is limited.
- **Difficulty (1)**: pass `rtol=0` to `np.allclose`.
- **Decision (1)**: `output-change`: slightly asymmetric dense inputs get symmetrized (tiny changes in `spectral_embedding`, MDS); `new-warning`/`new-error`: they also start warning, or raising in `smacof`/`MDS(dissimilarity="precomputed")`, which use `raise_exception=True`, for float noise above 1e-10 (float32 distance matrices would hit this). `unclear-spec`: a dtype- and scale-aware tolerance is probably needed (see #11159), to be agreed on.

## Upstream search

Queries: "check_symmetric tolerance", "check_symmetric allclose".

- #11159 (closed, 2018-2019, "check_symmetric: adapt tolerance to data type"): asked for a tolerance depending on the dtype (float32 dissimilarities failing `raise_exception=True` in MDS). It was closed as fixed by #13554 (float32 precision of `euclidean_distances`), without touching `check_symmetric`. It did not discuss the dense/sparse inconsistency or the hidden `rtol`.

The dense check is unchanged since the function was created in 7794d90471 (2014, "MAINT: create ensure_symmetric utility function to check matrix symmetry").

## Describe the bug and its user-facing impact

The docstring says `tol` is the "Absolute tolerance for equivalence of arrays. Default = 1E-10". For sparse input, the check is `abs(A - A.T).data < tol`, as documented. For dense input, it is `np.allclose(array, array.T, atol=tol)`, which keeps NumPy's default `rtol=1e-5`: any asymmetry below `1e-5 * |A[j, i]|` passes, whatever `tol` is.

So a dense matrix of distances around 1000 with an asymmetry of 5e-3 is:

- returned unchanged (not symmetrized, no warning), while the same matrix in sparse format is symmetrized with a warning;
- accepted by `raise_exception=True` (used by `smacof` and `MDS(dissimilarity="precomputed")`), while the sparse version raises.

Callers relying on the "symmetrized version of the input array" (e.g. `spectral_embedding` feeding the matrix to symmetric eigensolvers) can get a non-symmetric matrix.

## Steps/Code to Reproduce

```python
import warnings
import numpy as np
from scipy import sparse
from sklearn.utils.validation import check_symmetric

A = np.array([[0.0, 1000.0], [1000.005, 0.0]])  # asymmetry of 5e-3, tol=1e-10

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    dense = check_symmetric(A)
    print("dense :", dense[0, 1], dense[1, 0], "warnings:", len(w))
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    sp = check_symmetric(sparse.csr_array(A)).toarray()
    print("sparse:", sp[0, 1], sp[1, 0], "warnings:", len(w))

try:
    check_symmetric(A, raise_exception=True)
    print("dense : no exception")
except ValueError as e:
    print("dense :", e)
try:
    check_symmetric(sparse.csr_array(A), raise_exception=True)
    print("sparse: no exception")
except ValueError as e:
    print("sparse:", e)
```

## Expected Results

Same behavior for dense and sparse input, following `tol`: the dense array is symmetrized to `1000.0025` with a warning, and `raise_exception=True` raises.

## Actual Results

```
dense : 1000.0 1000.005 warnings: 0
sparse: 1000.0025 1000.0025 warnings: 1
dense : no exception
sparse: Array must be symmetric
```

## Root cause and possible fix

`sklearn/utils/validation.py:1569` (upstream/main 289c4d5cfa): `np.allclose(array, array.T, atol=tol)` should be `np.allclose(array, array.T, rtol=0, atol=tol)` to match the docstring and the sparse branch at line 1567. If maintainers think a purely absolute 1e-10 is too strict for float noise in `smacof`, the alternative is to document a relative tolerance and use it in both branches.

## Versions

```
sklearn: 1.10.dev0, commit 289c4d5cfa (upstream/main; local build f570497251, no diff near check_symmetric)
python: 3.13, numpy: 2.5.3, scipy: 1.18.1, platform: linux
```
