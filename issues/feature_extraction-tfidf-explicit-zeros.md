# `TfidfTransformer` counts explicit zeros as occurrences, and `sublinear_tf=True` fails on them

```json
{
    "estimators": ["TfidfTransformer"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 1,
    "difficulty": 1,
    "decision": 0,
    "decision_tags": []
}
```

## Classification

- **Priority (1)**: only sparse count matrices with explicitly stored zeros are affected. `CountVectorizer` never produces them, but they appear easily when users threshold or edit a count matrix in place (`X.data[X.data < k] = 0`). The idf error is silent, the `sublinear_tf` one is a confusing "Input contains infinity" error.
- **Difficulty (1)**: call `eliminate_zeros()` on a copy, or count `X.data != 0` in `_document_frequency`, and skip zeros in the `sublinear_tf` log.
- **Decision (0)**: the result should not depend on the storage of zeros.

## Upstream search

Queries: "TfidfTransformer explicit zeros", "sublinear_tf divide by zero", "TfidfTransformer document frequency explicit zero".

- #3637 (closed in 2017): "HashingVectorizer + TfidfTransformer fails because of a stored zero", same `sublinear_tf` symptom, triggered by a hash collision with `non_negative=True`. It was closed when `non_negative` was replaced by `alternate_sign` (#7565), but `TfidfTransformer` itself still does not handle explicit zeros: the error is still reproducible (with `HashingVectorizer`, see below, or any matrix with stored zeros), and the wrong `idf_` was not discussed there.

## Describe the bug and its user-facing impact

`TfidfTransformer.fit` computes the document frequency as the number of stored entries per column, so an explicitly stored zero counts as an occurrence of the term. The `idf_` of such terms is underestimated, and the result differs from passing the same matrix densified.

With `sublinear_tf=True`, `transform` applies `log` to every stored value, so an explicit zero becomes `-inf` and `normalize` then raises `ValueError: Input contains infinity`.

## Steps/Code to Reproduce

```python
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfTransformer

counts = np.array([[2.0, 1.0, 0.0], [0.0, 1.0, 3.0]])
X = sparse.csr_array(counts)
X.data[X.data < 2] = 0  # drop rare counts; this keeps explicit zeros
print(X.toarray(), "nnz:", X.nnz)

print("idf_ from sparse:", TfidfTransformer().fit(X).idf_)
print("idf_ from dense :", TfidfTransformer().fit(X.toarray()).idf_)
TfidfTransformer(sublinear_tf=True).fit_transform(X)
```

## Expected Results

The same `idf_` for the sparse and dense inputs, and a tf-idf matrix with `sublinear_tf=True` (zeros stay zeros, as for implicit zeros).

## Actual Results

```
RuntimeWarning: divide by zero encountered in log
  np.log(X.data, X.data)
[[2. 0. 0.]
 [0. 0. 3.]] nnz: 4
idf_ from sparse: [1.40546511 1.         1.40546511]
idf_ from dense : [1.40546511 2.09861229 1.40546511]
Traceback (most recent call last):
  File ".../F.py", line 12, in <module>
    TfidfTransformer(sublinear_tf=True).fit_transform(X)
  ...
  File "/home/arthur/scikit-learn/sklearn/feature_extraction/text.py", line 1726, in transform
    X = normalize(X, norm=self.norm, copy=False)
  ...
  File "/home/arthur/scikit-learn/sklearn/utils/validation.py", line 183, in _assert_all_finite_element_wise
    raise ValueError(msg_err)
ValueError: Input contains infinity or a value too large for dtype('float64').
```

## Root cause and possible fix

- `_document_frequency` (`sklearn/feature_extraction/text.py:924-929`) returns `np.bincount(X.indices)` (CSR) or `np.diff(X.indptr)` (CSC), i.e. the number of stored entries, not the number of non-zero entries.
- `TfidfTransformer.transform` (`sklearn/feature_extraction/text.py:1716-1718`) does `np.log(X.data, X.data)` on all stored values.

Fix: in `fit`, count only non-zero stored values (e.g. `np.bincount(X.indices, weights=X.data != 0, ...)`, or `eliminate_zeros()` on a copy); in `transform`, `X` is already a copy by default, so `X.eliminate_zeros()` before the log is cheap. `HashingVectorizer` can also produce explicit zeros when two tokens of a document hash to the same column with opposite signs (`alternate_sign=True`), so a `HashingVectorizer` + `TfidfTransformer(sublinear_tf=True)` pipeline can hit a similar error (I got `Input contains NaN` with `HashingVectorizer(n_features=4, norm=None)` on the document `"aa ab"`).

## Versions

```
sklearn: 1.10.dev0, commit f570497251 (code identical on upstream/main 289c4d5cfa)
python: 3.13.14, numpy: 2.5.3, scipy: 1.18.1, platform: linux
```

See also: `feature_extraction-hashing-vectorizer-binary-cancelled-collision.md` (stored zeros from `FeatureHasher` turned into 1 by `HashingVectorizer(binary=True)`).
