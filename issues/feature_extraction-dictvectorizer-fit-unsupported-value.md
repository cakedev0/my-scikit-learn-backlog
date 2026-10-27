# `DictVectorizer.fit` silently skips unsupported values (or raises `UnboundLocalError`), unlike `fit_transform`

```json
{
    "estimators": ["DictVectorizer"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 1,
    "difficulty": 1,
    "decision": 1,
    "decision_tags": ["new-error"]
}
```

## Classification

- **Priority (1)**: `fit` and `fit_transform` disagree on the same input; `fit` either drops the feature silently or fails with an unrelated `UnboundLocalError`. Only for values of unsupported types (dates, custom objects), and `transform` raises a clear error afterwards.
- **Difficulty (1)**: add the missing `else: raise TypeError(...)` branch in `fit`, mirroring `_transform`.
- **Decision (1)**: `new-error`: `fit` alone currently "succeeds" on such input.

## Upstream search

Queries: "DictVectorizer UnboundLocalError", "DictVectorizer fit unsupported type", "DictVectorizer fit transform inconsistent".

No match. #17367 (merged) added iterable values and the `TypeError` for mappings, but only `_transform` got a final `else` branch.

## Describe the bug and its user-facing impact

`DictVectorizer.fit` has no `else` branch for values that are neither `str`, `Number`, `None`, `Mapping` nor `Iterable`. For such a value, `feature_name` keeps its value from the previous item of the loop:

- if a previous item exists, the unsupported feature is silently ignored (its name is not added to the vocabulary, and the previous feature is just processed twice);
- if it is the first item, `fit` fails with `UnboundLocalError: cannot access local variable 'feature_name'`.

`fit_transform` and `transform` raise a clear `TypeError: Unsupported value Type ...` on the same input.

## Steps/Code to Reproduce

```python
import datetime
from sklearn.feature_extraction import DictVectorizer

D = [{"price": 1.0, "date": datetime.date(2020, 1, 1)}]
print(DictVectorizer().fit(D).feature_names_)  # "date" silently ignored
try:
    DictVectorizer().fit_transform(D)
except TypeError as e:
    print("fit_transform:", e)
DictVectorizer().fit([{"date": datetime.date(2020, 1, 1)}])
```

## Expected Results

`fit` raises the same `TypeError` as `fit_transform` in both cases.

## Actual Results

```
['price']
fit_transform: Unsupported value Type <class 'datetime.date'> for date: 2020-01-01.
<class 'datetime.date'> objects are not supported.
Traceback (most recent call last):
  File ".../G.py", line 10, in <module>
    DictVectorizer().fit([{"date": datetime.date(2020, 1, 1)}])
  File "/home/arthur/scikit-learn/sklearn/feature_extraction/_dict_vectorizer.py", line 178, in fit
    if feature_name is not None:
       ^^^^^^^^^^^^
UnboundLocalError: cannot access local variable 'feature_name' where it is not associated with a value
```

## Root cause and possible fix

`sklearn/feature_extraction/_dict_vectorizer.py:163-175`: the `if`/`elif` chain in `fit` has no final `else`, while `_transform` (lines 243-244) ends with `else: raise TypeError(...)`. Adding the same `else` branch to `fit` fixes both symptoms. A call to `fit` that currently returns would then raise (`new-error`), but `transform` on the same data already raises, so no working pipeline depends on it.

## Versions

```
sklearn: 1.10.dev0, commit f570497251 (code identical on upstream/main 289c4d5cfa)
python: 3.13.14, numpy: 2.5.3, scipy: 1.18.1, platform: linux
```
