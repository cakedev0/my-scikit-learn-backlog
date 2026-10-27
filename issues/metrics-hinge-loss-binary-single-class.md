# `hinge_loss` treats all samples as negative when `y_true` has a single class, even with `labels`

```json
{
    "estimators": ["hinge_loss"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 1,
    "difficulty": 1,
    "decision": 1,
    "decision_tags": ["new-error"]
}
```

## Classification

- **Priority (1)**: silent wrong value, but only when the evaluated batch
  contains a single class (small folds, per-batch evaluation, a positive-only
  subset). The docstring asks for two unique values in the binary case.
- **Difficulty (1)**: fit the `LabelBinarizer` on `labels` when given, and
  raise (or warn) when only one class is known.
- **Decision (1)**: raising for single-class `y_true` without `labels` is a
  new error for calls that currently return a (wrong) number.

## Upstream search

Queries: "hinge_loss", "hinge_loss single class", "hinge_loss binary one
class", "hinge_loss LabelBinarizer", "hinge_loss one class".

- #17630 (closed): multiclass `hinge_loss` with only two classes present in
  `y_true` and `labels` given; fixed for the multiclass branch only, this bug
  is in the binary branch.
- #33054 (merged): docstring update stating the binary `y_true` should contain
  two unique values; documents the requirement but nothing enforces it.
- #1883 (closed): old `neg_label` bug, unrelated code.

No match.

## Describe the bug and its user-facing impact

In the binary case, `hinge_loss` encodes `y_true` with
`LabelBinarizer(neg_label=-1).fit_transform(y_true)`. When `y_true` holds a
single class, `LabelBinarizer` maps it to `-1` whatever that class is, so a
batch of only positive samples is scored as if they were all negative. The
`labels` parameter does not help: it is only used in the multiclass branch.

A batch of positives with large positive margins (perfect predictions) gets a
loss of `1 + margin` per sample instead of 0. No warning or error is raised.

## Steps/Code to Reproduce

```python
import numpy as np
from sklearn.metrics import hinge_loss

d = np.array([2.0, 2.0, 2.0])  # confident, correct decisions for class 1
print(hinge_loss([1, 1, 1], d))                   # expected 0.0
print(hinge_loss([1, 1, 1], d, labels=[-1, 1]))   # expected 0.0
print(hinge_loss([-1, -1, -1], -d))               # 0.0, correct
print(hinge_loss([1, -1, 1], np.array([2.0, -2.0, 2.0])))  # 0.0, correct
```

## Expected Results

`0.0` for the first two calls: each sample has margin 2 on the right side.
At minimum, the `labels=[-1, 1]` call should use the declared labels, and the
call without `labels` should raise instead of guessing.

## Actual Results

```
3.0
3.0
0.0
0.0
```

## Root cause and possible fix

`sklearn/metrics/_classification.py:3597-3598`:

```python
lbin = LabelBinarizer(neg_label=-1)
y_true = lbin.fit_transform(y_true)[:, 0]
```

`LabelBinarizer` fitted on a single class returns `neg_label` for every
sample. `labels` is computed into `y_true_unique` at line 3551 but ignored in
this branch. Fix: `lbin.fit(labels if labels is not None else y_true)`, and
raise a `ValueError` asking for `labels` when fewer than two classes are
known (as `log_loss` does). Extending the `labels` docstring to the binary
case goes with it.

## Versions

```
sklearn: 1.10.dev0, commit 289c4d5cfa (upstream/main; sklearn/metrics/ sources identical to the tested checkout)
python: 3.13, numpy: 2.5.3, scipy: 1.18.1, platform: linux
```
