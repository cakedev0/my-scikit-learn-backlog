# `LabelBinarizer.inverse_transform` returns wrong labels on sparse multiclass scores whose row maximum is an implicit zero

```json
{
    "estimators": ["LabelBinarizer"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 1,
    "difficulty": 1,
    "decision": 0,
    "decision_tags": []
}
```

## Classification

- **Priority (1)**: silent wrong labels, but only for sparse score matrices with negative entries, which is an unusual input (decision functions are rarely sparse). Dense input is correct.
- **Difficulty (1)**: one line stores a class label where a column index is expected.
- **Decision (0)**: plain fix.

## Upstream search

Queries (`gh search issues --include-prs`): "inverse_binarize_multiclass", "LabelBinarizer inverse_transform sparse". Only the array API PRs (#32582, #28626) and unrelated old PRs. The sparse branch dates from 2bfca14c49 ("ENH sparse matrix support in label binarization") and was not touched since, apart from formatting.

## Describe the bug and its user-facing impact

For a multiclass `LabelBinarizer`, `inverse_transform` picks the class with the largest score in each row. With a sparse `Y`, when a row stores only negative values, its maximum is one of the implicit zeros. That case is handled by a dedicated loop, which writes the class label of the first non-stored column into the array of column indices, then indexes `classes_` a second time with it.

- With classes `[0, 1, ..., n-1]` the double indexing is a no-op, which is why the existing test (`np.arange(3)` as classes) passes.
- With other integer classes, the result is silently wrong (`[1, 2, 3]` returns `3` instead of `2`), or raises an `IndexError` when the label is out of range.
- With string classes it raises `ValueError: invalid literal for int()`.
- If the row stores an explicit `0.0` as its maximum and no implicit zero, `np.setdiff1d(...)[0]` raises `IndexError` on an empty array.

## Steps/Code to Reproduce

```python
import numpy as np
from scipy import sparse
from sklearn.preprocessing import LabelBinarizer

lb = LabelBinarizer().fit([1, 2, 3])
# decision scores: row 0 is maximal at the 2nd column (score 0, the others < 0)
scores = np.array([[-1.0, 0.0, -2.0],
                   [-1.0, 0.0, -2.0],
                   [0.5, 0.1, 0.0]])
print("dense: ", lb.inverse_transform(scores))
print("sparse:", lb.inverse_transform(sparse.csr_array(scores)))

lb = LabelBinarizer().fit(["a", "b", "c"])
print("dense: ", lb.inverse_transform(scores))
print("sparse:", lb.inverse_transform(sparse.csr_array(scores)))
```

## Expected Results

The sparse result matches the dense one: `[2 2 1]` and `['b' 'b' 'a']`.

## Actual Results

```
dense:  [2 2 1]
sparse: [3 3 1]
dense:  ['b' 'b' 'a']
Traceback (most recent call last):
  File "lb_sparse_inv.py", line 15, in <module>
    print("sparse:", lb.inverse_transform(sparse.csr_array(scores)))
                     ~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/arthur/scikit-learn/sklearn/preprocessing/_label.py", line 442, in inverse_transform
    y_inv = _inverse_binarize_multiclass(Y, self.classes_, xp=xp)
  File "/home/arthur/scikit-learn/sklearn/preprocessing/_label.py", line 732, in _inverse_binarize_multiclass
    y_i_argmax[i] = classes[np.setdiff1d(outputs, ind)][0]
    ~~~~~~~~~~^^^
ValueError: invalid literal for int() with base 10: np.str_('b')
```

With an explicitly stored zero as the row maximum (`sparse.csr_array((np.array([-1., 0., -2.]), np.array([0, 1, 2]), np.array([0, 3])), shape=(1, 3))` and classes `[0, 1, 2]`): `IndexError: index 0 is out of bounds for axis 0 with size 0`.

## Root cause and possible fix

`sklearn/preprocessing/_label.py:729-734` (commit below):

```python
samples = np.arange(n_samples)[(row_nnz > 0) & (row_max.ravel() == 0)]
for i in samples:
    ind = y.indices[y.indptr[i] : y.indptr[i + 1]]
    y_i_argmax[i] = classes[np.setdiff1d(outputs, ind)][0]

return classes[y_i_argmax]
```

`y_i_argmax` holds column indices, so the line should be `y_i_argmax[i] = np.setdiff1d(outputs, ind)[0]`. The loop should also only override rows that have an implicit zero (`row_nnz < n_outputs`); rows whose maximum is a stored zero are already handled by the vectorized argmax above.

## Versions

```
sklearn: 1.10.dev0, commit 289c4d5cfa
python: 3.13.14, numpy: 2.5.3, scipy: 1.18.1, platform: linux
```
