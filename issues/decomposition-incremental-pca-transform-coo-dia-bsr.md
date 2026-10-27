# `IncrementalPCA.transform` and `fit_transform` crash on COO, DIA and BSR sparse input that `fit` accepts

```json
{
    "estimators": ["IncrementalPCA"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 1,
    "difficulty": 1,
    "decision": 0,
    "decision_tags": []
}
```

## Classification

- **Priority (1)**: loud crash, easy workaround (`X.tocsr()`), and `coo_array` (the default COO container in recent scipy) is not affected. Still, `fit_transform` on a `coo_matrix` is a plain call that fails.
- **Difficulty (1)**: validate (convert to CSR) in the sparse branch of `transform`.
- **Decision (0)**: no trade-off.

## Upstream search

Queries (issues and PRs): "IncrementalPCA transform sparse coo", "IncrementalPCA sparse transform". No match. #29512 (open, ARPACK solver for sparse `IncrementalPCA`) touches the same estimator but not this code path.

## Describe the bug and its user-facing impact

`IncrementalPCA.fit` accepts any sparse format (it converts to CSR through `validate_data(accept_sparse=["csr", "csc", "lil"])`). `IncrementalPCA.transform` does not validate sparse input: it slices `X[batch]` directly on what the user passed. That fails for formats without row slicing:

| input            | `fit` | `transform` / `fit_transform`             |
|------------------|-------|-------------------------------------------|
| `coo_matrix`     | ok    | `TypeError: 'coo_matrix' object is not subscriptable` |
| `dia_matrix`, `dia_array` | ok | `TypeError: ... is not subscriptable` |
| `bsr_matrix`, `bsr_array` | ok | `NotImplementedError` |
| `coo_array`, `dok_matrix`, `lil_matrix`, CSR, CSC | ok | ok |

Since `fit_transform` is `fit(X).transform(X)`, `IncrementalPCA().fit_transform(coo_matrix)` crashes right after a successful fit. The common test `check_estimator_sparse_matrix` does not catch it because it only calls `predict`/`predict_proba` after `fit`.

## Steps/Code to Reproduce

```python
import scipy.sparse as sp
from sklearn.decomposition import IncrementalPCA

X = sp.coo_matrix(sp.random_array((50, 5), density=0.5, random_state=0))
IncrementalPCA(n_components=2).fit(X)  # works
IncrementalPCA(n_components=2).fit_transform(X)
```

## Expected Results

The projected data, as for the same matrix in CSR format.

## Actual Results

```
Traceback (most recent call last):
  ...
  File "/home/arthur/scikit-learn/sklearn/base.py", line 969, in fit_transform
    return self.fit(X, **fit_params).transform(X)
  ...
  File "/home/arthur/scikit-learn/sklearn/decomposition/_incremental_pca.py", line 410, in transform
    output.append(super().transform(X[batch].toarray()))
                                    ~^^^^^^^
  File ".../scipy/sparse/_coo.py", line 1945, in __getitem__
    raise TypeError("'coo_matrix' object is not subscriptable")
TypeError: 'coo_matrix' object is not subscriptable
```

## Root cause and possible fix

`sklearn/decomposition/_incremental_pca.py:404-411`: the sparse branch of `transform` loops over `gen_batches` and indexes `X[batch]` without converting `X`. Converting first fixes it, e.g. `X = check_array(X, accept_sparse=["csr", "csc", "lil"])` (or a plain `X.tocsr()` when the format is not row-sliceable) before the loop; the per-batch `super().transform` keeps doing the feature checks.

## Versions

```
sklearn: 1.10.dev0, commit f570497251 (_incremental_pca.py identical to upstream/main 289c4d5cfa)
python: 3.13.14, numpy: 2.5.3, scipy: 1.18.1, platform: linux
```
