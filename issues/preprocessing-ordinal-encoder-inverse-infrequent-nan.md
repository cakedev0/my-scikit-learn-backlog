# `OrdinalEncoder.inverse_transform` is wrong with infrequent categories and missing values in `fit`

```json
{
    "estimators": ["OrdinalEncoder"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 2,
    "difficulty": 1,
    "decision": 0,
    "decision_tags": []
}
```

## Classification

- **Priority (2)**: silent wrong output (infrequent code decoded as `nan`) or an `IndexError`, with a common combination: `min_frequency`/`max_categories` on a column that has missing values. Only `inverse_transform` is affected, which is less used than `transform`.
- **Difficulty (1)**: two index computations in `inverse_transform` ignore the `nan` category, a few lines to fix.
- **Decision (0)**: the current output is plainly wrong.

## Upstream search

Queries (issues and PRs, open and closed): "OrdinalEncoder inverse_transform infrequent", "OrdinalEncoder inverse_transform nan infrequent", "OrdinalEncoder min_frequency nan inverse_transform".

- #35155 (closed, "Needs Triage", closed by its author 2026-10-18 without any comment): "OrdinalEncoder.inverse_transform swaps infrequent and missing values". Same root cause, with one infrequent category (rows swapped silently); this report adds the `IndexError` with two infrequent categories. The bug is still present on main, so the closure is not a fix.
- #35158 (closed by the bot because the issue was not triaged): a fix reading the infrequent code from `_default_to_infrequent_mappings[i].max()` and remapping missing values after filtering. It was never reviewed.
- #25677 (merged): added infrequent categories to `OrdinalEncoder`, where this code comes from.

## Describe the bug and its user-facing impact

When `OrdinalEncoder` is fitted with infrequent categories enabled (`min_frequency` or `max_categories`) on a column that contains missing values, `inverse_transform` is broken:

- the encoded value of the infrequent group is decoded as `nan` instead of `"infrequent_sklearn"` (silently wrong);
- an encoded missing value raises `IndexError`.

`transform` is correct. Any user calling `inverse_transform` (e.g. to inspect predictions of a model on encoded features, or through a pipeline) on data with missing values and rare categories hits it.

## Steps/Code to Reproduce

```python
import numpy as np
from sklearn.preprocessing import OrdinalEncoder

X = np.array([["a"] * 5 + ["b"] * 3 + ["c", "d"] + [np.nan] * 2], dtype=object).T
enc = OrdinalEncoder(min_frequency=2).fit(X)
print(enc.categories_, enc.infrequent_categories_)

X_test = np.array([["a"], ["b"], ["c"]], dtype=object)
X_trans = enc.transform(X_test)
print(X_trans.ravel())
print(enc.inverse_transform(X_trans).ravel())  # "c" should come back as "infrequent_sklearn"

enc.inverse_transform([[np.nan]])  # missing value should come back as nan
```

## Expected Results

```
['a' 'b' 'infrequent_sklearn']
```

and `[[nan]]` for the last call, as when there are no missing values in `fit`.

## Actual Results

```
[array(['a', 'b', 'c', 'd', nan], dtype=object)] [array(['c', 'd'], dtype=object)]
[0. 1. 2.]
['a' 'b' nan]
Traceback (most recent call last):
  File ".../A.py", line 13, in <module>
    enc.inverse_transform([[np.nan]])  # missing value should come back as nan
    ~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^
  File "/home/arthur/scikit-learn/sklearn/preprocessing/_encoders.py", line 1671, in inverse_transform
    X_tr[rows_to_update, i] = categories[labels_int]
                              ~~~~~~~~~~^^^^^^^^^^^^
IndexError: index 4 is out of bounds for axis 0 with size 3
```

## Root cause and possible fix

In `fit`, the `nan` category is excluded from the infrequent grouping (`return_and_ignore_missing_for_infrequent=True`), so the encoding is `a=0, b=1, infrequent=2` and missing values get `encoded_missing_value`. `inverse_transform` (`sklearn/preprocessing/_encoders.py:1636-1671`) does not account for this:

- line 1650 computes `infrequent_encoding_value = len(categories) - len(infrequent_indices[i])`, i.e. `5 - 2 = 3` because `categories` still contains `nan`. The real code is `2`, so the infrequent rows are not detected and are looked up in the frequent categories `['a', 'b', nan]`, giving `nan`.
- line 1643 maps encoded missing values back to `self._missing_indices[i]` (the index of `nan` in `categories_`, here `4`), which is out of bounds of the reduced `categories` array of size 3.

Fix: when `i in self._missing_indices`, subtract one from `infrequent_encoding_value`, and handle the missing rows separately (exclude them from `rows_to_update` and set them to `categories_[i][-1]` at the end), as is done for unknown values.

## Versions

```
sklearn: 1.10.dev0, commit f570497251 (code identical on upstream/main 289c4d5cfa)
python: 3.13.14, numpy: 2.5.3, scipy: 1.18.1, platform: linux
```
