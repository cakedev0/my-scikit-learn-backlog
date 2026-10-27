# `OneHotEncoder(drop=[...])`: dropping an infrequent string category raises `AttributeError` instead of the intended `ValueError`

```json
{
    "estimators": ["OneHotEncoder"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 1,
    "difficulty": 1,
    "decision": 0,
    "decision_tags": []
}
```

## Classification

- **Priority (1)**: the call is invalid anyway, but the user gets `AttributeError: 'str' object has no attribute 'item'` instead of the explanation that the category is infrequent. Happens for object-dtype categories, i.e. pandas string columns and object arrays, the common case.
- **Difficulty (1)**: one line in an error message.
- **Decision (0)**: no trade-off.

## Upstream search

Queries: "OneHotEncoder drop infrequent AttributeError", "Unable to drop category infrequent", "OneHotEncoder drop min_frequency object dtype".

No match. #27042 (merged) added the `.item()` call for NEP 51; #25550 (closed) is about the `drop_idx_` description with infrequent categories, unrelated.

## Describe the bug and its user-facing impact

When `drop` names a category that ends up infrequent (`min_frequency` or `max_categories`), `OneHotEncoder.fit` is meant to raise `ValueError: Unable to drop category 'c' from feature 0 because it is infrequent`. Building that message calls `.item()` on the category, which only works for numpy scalars. With object-dtype categories (pandas DataFrames, `dtype=object` arrays), the category is a Python `str` and the message construction itself fails.

The existing tests pass because they use `np.array([["a", ...]])`, whose categories are `numpy.str_`.

## Steps/Code to Reproduce

```python
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

X = pd.DataFrame({"city": ["a"] * 5 + ["b"] * 3 + ["c", "d"]})
OneHotEncoder(drop=["c"], min_frequency=2).fit(X)
```

## Expected Results

```
ValueError: Unable to drop category 'c' from feature 0 because it is infrequent
```

## Actual Results

```
Traceback (most recent call last):
  File ".../M.py", line 5, in <module>
    OneHotEncoder(drop=["c"], min_frequency=2).fit(X)
  File "/home/arthur/scikit-learn/sklearn/preprocessing/_encoders.py", line 996, in fit
    self._set_drop_idx()
  File "/home/arthur/scikit-learn/sklearn/preprocessing/_encoders.py", line 867, in _set_drop_idx
    self._map_drop_idx_to_infrequent(feature_idx, drop_idx[0])
  File "/home/arthur/scikit-learn/sklearn/preprocessing/_encoders.py", line 798, in _map_drop_idx_to_infrequent
    f"Unable to drop category {categories[drop_idx].item()!r} from"
                               ^^^^^^^^^^^^^^^^^^^^^^^^^
AttributeError: 'str' object has no attribute 'item'
```

## Root cause and possible fix

`sklearn/preprocessing/_encoders.py:798`. The `.item()` call was added to keep the message independent of the NEP 51 scalar repr (numpy >= 2). Using `categories[drop_idx:drop_idx + 1].tolist()[0]` (or `np.asarray(categories[drop_idx]).item()`) gives a Python scalar for every dtype. A test with `dtype=object` input would cover it.

## Versions

```
sklearn: 1.10.dev0, commit f570497251 (code identical on upstream/main 289c4d5cfa)
python: 3.13.14, numpy: 2.5.3, scipy: 1.18.1, pandas: 3.0.6, platform: linux
```
