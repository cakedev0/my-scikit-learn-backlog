# Binary `LabelBinarizer.transform` returns 2 columns when `y` contains an unseen label and both known classes

```json
{
    "estimators": ["LabelBinarizer", "label_binarize"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 1,
    "difficulty": 1,
    "decision": 1,
    "decision_tags": ["unclear-spec", "new-error"]
}
```

## Classification

- **Priority (1)**: needs labels at `transform` time that were not seen in `fit`, on a binary problem. The output shape (and the meaning of the unseen label) then depends on the content of the batch, which can break downstream code in a data-dependent way, but this is an unusual use.
- **Difficulty (1)**: in `label_binarize`, decide the binary/multiclass layout from `classes` instead of `type_of_target(y)`.
- **Decision (1)**: a 1-column output cannot represent an unseen label (0 already means the first class), so the fix has to pick between raising (`new-error`) and keeping the current "encode as negative" behavior consistently (`unclear-spec`). Either is better than a shape that depends on the batch.

## Upstream search

Queries (`gh search issues --include-prs`): "label_binarize binary unseen shape", "LabelBinarizer unseen label binary columns", "label_binarize shape binary"; `gh issue list --search` "LabelBinarizer unseen", "label_binarize two classes".

- #3462 (closed, 0.15.1): regression in how `LabelBinarizer` handles unseen labels in the multiclass case (all-zeros rows), tested by `test_label_binarizer_unseen_labels`. The binary case was not discussed.
- #4450 (open): `label_binarize` sorts classes lexicographically, unrelated.

No match.

## Describe the bug and its user-facing impact

The docstrings of `LabelBinarizer.transform` and `label_binarize` say the output has shape `(n_samples, 1)` for binary problems. After fitting on two classes, `transform` behaves in two different ways when `y` contains a label that was not seen in `fit`:

- if `y` does not contain both known classes, the unseen label is silently encoded as the negative class (`0`) in a single column;
- if `y` contains both known classes plus an unseen one, `type_of_target(y)` says "multiclass" and the output has 2 columns, with an all-zero row for the unseen label.

So the number of output columns, and whether an unseen label is distinguishable from the negative class, depend on which labels happen to be in the batch. `inverse_transform` of the 2-column output maps the unseen label to the negative class.

## Steps/Code to Reproduce

```python
from sklearn.preprocessing import LabelBinarizer

lb = LabelBinarizer().fit(["no", "yes"])
print(lb.transform(["no", "yes"]).shape)           # (2, 1) as documented
print(lb.transform(["no", "maybe"]).ravel())       # "maybe" encoded as "no"
print(lb.transform(["no", "yes", "maybe"]))        # now 2 columns
print(lb.inverse_transform(lb.transform(["no", "yes", "maybe"])))
```

## Expected Results

A consistent behavior for a binary `LabelBinarizer`: always one column, and either an error for unseen labels or a documented encoding of them.

## Actual Results

```
(2, 1)
[0 0]
[[1 0]
 [0 1]
 [0 0]]
['no' 'yes' 'no']
```

## Root cause and possible fix

`sklearn/preprocessing/_label.py:561` (commit below) computes `y_type = type_of_target(y)` from the data being transformed, and line 602-611 only collapses to one column when `y_type == "binary"`. With 3 distinct values in `y` and 2 `classes`, `y_type` is "multiclass" and the binary reduction at line 688 is skipped.

Possible fix: when `y_type` is "multiclass" but `len(classes) == 2`, treat it as binary for the output layout. Then decide what an unseen label gives in the binary case: raising a `ValueError` is the safest since a single column cannot encode it, but it would also affect the "single known class in the batch" path that silently works today.

## Versions

```
sklearn: 1.10.dev0, commit 289c4d5cfa
python: 3.13.14, numpy: 2.5.3, scipy: 1.18.1, platform: linux
```
