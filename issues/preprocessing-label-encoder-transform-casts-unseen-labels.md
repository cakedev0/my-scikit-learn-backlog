# `LabelEncoder.transform` casts `y` to `classes_.dtype`, so unseen labels are silently encoded

```json
{
    "estimators": ["LabelEncoder"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 2,
    "difficulty": 1,
    "decision": 0,
    "decision_tags": []
}
```

## Classification

- **Priority (2)**: silent wrong output instead of the documented "unseen labels" error, with ordinary inputs: any string label longer than the longest fitted class is truncated, and float labels are truncated toward zero when the classes are integers. `LabelEncoder` is widely used to encode test targets by hand, where unseen labels are exactly what the error is meant to catch.
- **Difficulty (1)**: only cast when the cast is lossless (or only for object `classes_`, which is what the cast was added for), a few lines.
- **Decision (0)**: the fix restores the pre-1.2 error.

## Upstream search

Queries (`gh search issues --include-prs`): "LabelEncoder transform truncated", "LabelEncoder unseen labels dtype", "LabelEncoder float transform cast", "LabelEncoder string longer", "LabelEncoder silently wrong", "LabelEncoder transform unseen not raised"; `gh issue list --search` "LabelEncoder transform dtype", "LabelEncoder classes_ dtype cast".

- #22628 / #22629 (closed / merged, 1.2): NaN not encoded by `transform`. The fix added `dtype=self.classes_.dtype` to `column_or_1d` in `transform`, which introduced this bug.
- #17294 (open): mixed-type labels in `LabelEncoder`, a different issue.
- #9285 (closed): mixed strings and numbers in `inverse_transform`, different.

No report of the truncation.

## Describe the bug and its user-facing impact

`LabelEncoder.transform` converts `y` to the dtype of `classes_` before looking labels up. The conversion is lossy, so a label that was never seen during `fit` can be mapped to an existing class instead of raising `ValueError: y contains previously unseen labels`:

- fitted on numpy strings, longer strings are truncated to the fitted itemsize (`"category"` becomes `"cat"` when classes are `["cat", "dog"]`);
- fitted on integers, floats are truncated (`1.5` becomes `1`), and so are numeric strings (`"2"` becomes `2`).

The output looks valid, so users have no way to notice.

## Steps/Code to Reproduce

```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder().fit(["cat", "dog"])
# "category" and "doge" were never seen: they should raise "unseen labels"
print(le.transform(["category", "doge"]))

le = LabelEncoder().fit([1, 2, 3])
# 1.5 and 2.9 are not classes either
print(le.transform([1.5, 2.9]))
```

## Expected Results

`ValueError: y contains previously unseen labels: ...` in both cases, as with scikit-learn < 1.2 and as with any other unseen label (e.g. `le.transform(["cow"])`).

## Actual Results

```
[0 1]
[0 1]
```

## Root cause and possible fix

`sklearn/preprocessing/_label.py:136` (commit below):

```python
y = column_or_1d(y, dtype=self.classes_.dtype, warn=True)
```

The cast was added in #22629 so that `[np.nan]` (float) matches the NaN stored in an object `classes_`. Casting to a fixed-width `<U3` truncates strings, and casting floats to int truncates values.

Possible fix: only pass `dtype` when `classes_` is object (the NaN case), or cast and check that the round trip is lossless, otherwise let `_encode_labels` raise the unseen labels error. `inverse_transform` has no cast and is fine.

## Versions

```
sklearn: 1.10.dev0, commit 289c4d5cfa
python: 3.13.14, numpy: 2.5.3, scipy: 1.18.1, platform: linux
```
