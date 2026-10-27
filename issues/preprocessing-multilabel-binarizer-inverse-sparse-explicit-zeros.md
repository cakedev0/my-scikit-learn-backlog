# `MultiLabelBinarizer.inverse_transform` returns labels for explicitly stored zeros in a sparse indicator

```json
{
    "estimators": ["MultiLabelBinarizer"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 1,
    "difficulty": 1,
    "decision": 0,
    "decision_tags": []
}
```

## Classification

- **Priority (1)**: silent wrong label sets, but only for sparse indicators with stored zeros. `MultiLabelBinarizer.transform` never produces them, but thresholding a sparse score matrix in place (`Y.data = (Y.data > 0.5).astype(int)`) does.
- **Difficulty (1)**: filter on `yt.data` (or call `eliminate_zeros()` on a copy) before taking `classes_`.
- **Decision (0)**: the result should not depend on how zeros are stored; dense and sparse should agree.

## Upstream search

Queries (`gh search issues --include-prs`): "MultiLabelBinarizer inverse_transform sparse", "MultiLabelBinarizer explicit zeros". Only unrelated PRs (#3203 sparse label binarizer, release PRs). No match.

## Describe the bug and its user-facing impact

`MultiLabelBinarizer.inverse_transform` accepts sparse indicators and checks that they only contain 0s and 1s, so stored zeros are explicitly allowed. It then returns `classes_[indices]` for every stored entry of each row, including entries whose value is 0. The same matrix densified gives a different, correct answer.

## Steps/Code to Reproduce

```python
import numpy as np
from scipy import sparse
from sklearn.preprocessing import MultiLabelBinarizer

mlb = MultiLabelBinarizer().fit([["a", "b"], ["c"]])
Y = sparse.csr_array(np.array([[1, 1, 0], [0, 0, 1]]))
Y.data[0] = 0  # e.g. after thresholding scores in place
print(Y.toarray())
print("sparse:", mlb.inverse_transform(Y))
print("dense: ", mlb.inverse_transform(Y.toarray()))
```

## Expected Results

`[('b',), ('c',)]` for both.

## Actual Results

```
[[0 1 0]
 [0 0 1]]
sparse: [('a', 'b'), ('c',)]
dense:  [('b',), ('c',)]
```

## Root cause and possible fix

`sklearn/preprocessing/_label.py:1047-1054` (commit below): after validating `yt.data`, the sparse branch uses `yt.indices[start:end]` without looking at the stored values. Possible fix: work on `yt = yt.tocsr(copy=True)` then `yt.eliminate_zeros()` before building the tuples (or mask the indices with `yt.data[start:end] != 0`). `LabelBinarizer.inverse_transform` already calls `eliminate_zeros()` in its sparse thresholding path.

## Versions

```
sklearn: 1.10.dev0, commit 289c4d5cfa
python: 3.13.14, numpy: 2.5.3, scipy: 1.18.1, platform: linux
```
