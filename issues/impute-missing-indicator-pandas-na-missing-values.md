# `MissingIndicator(missing_values=pd.NA)` fails on any input containing `pd.NA`, unlike `SimpleImputer`

```json
{
    "estimators": ["MissingIndicator"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 1,
    "difficulty": 1,
    "decision": 0,
    "decision_tags": []
}
```

## Classification

- **Priority (1)**: loud errors, and the `MissingIndicator` docstring does not list `pd.NA` (it says to use `np.nan` for nullable integer columns). But the parameter validation accepts `pd.NA`, `SimpleImputer` supports it (including `add_indicator=True`, which uses `MissingIndicator` internally), and the errors are confusing: a raw `TypeError` from pandas, or "does not accept missing values encoded as NaN".
- **Difficulty (1)**: use the same `ensure_all_finite` rule as `SimpleImputer._validate_input`; `_get_mask` already handles `pd.NA`.
- **Decision (0)**: only turns errors into the result `SimpleImputer` already gives.

## Upstream search

Queries (`gh search issues --include-prs`): "boolean value of NA is ambiguous", "MissingIndicator pd.NA".

- #27349, #29291 (open): `SimpleImputer` with the default `missing_values=np.nan` fails on pandas string columns holding `pd.NA` (same `TypeError`, raised by `_object_dtype_isnan`). That also affects `MissingIndicator()` with the default, and is a different trigger: here the user explicitly asks for `pd.NA`, which `SimpleImputer` handles.
- #32312 (closed): `SimpleImputer(missing_values=pd.NA)` repr crash, fixed.
- #25078 (closed): `check_array` and `pd.NA`.

No report about `MissingIndicator(missing_values=pd.NA)`.

## Describe the bug and its user-facing impact

`MissingIndicator` validates `X` with `ensure_all_finite=True` whenever `missing_values` is not a float NaN, so with `missing_values=pd.NA`:

- an object or pandas `string` column with `pd.NA` crashes in the finiteness check with `TypeError: boolean value of NA is ambiguous`;
- a nullable `Int64` column (converted to float NaN by `check_array`) raises `ValueError: Input X contains NaN. MissingIndicator does not accept missing values encoded as NaN natively...`.

`SimpleImputer(missing_values=pd.NA, add_indicator=True)` on the same data works and returns the expected indicator, because `SimpleImputer` allows NaN when `missing_values` is `pd.NA` and passes a precomputed mask to its internal `MissingIndicator`. Users who build the indicator standalone (for instance in a `ColumnTransformer` next to another imputer) can't use the same `missing_values`.

## Steps/Code to Reproduce

```python
import pandas as pd
from sklearn.impute import MissingIndicator, SimpleImputer

X = pd.DataFrame({"city": ["Paris", None, "Lyon"]}).convert_dtypes()  # string[NA]

imputer = SimpleImputer(
    missing_values=pd.NA, strategy="most_frequent", add_indicator=True
)
print(imputer.fit_transform(X).tolist())

MissingIndicator(missing_values=pd.NA).fit_transform(X)
```

## Expected Results

`[[False], [True], [False]]` from `MissingIndicator`, consistent with the indicator column of `SimpleImputer`.

## Actual Results

```
[['Paris', False], ['Lyon', True], ['Lyon', False]]
Traceback (most recent call last):
  File "mi_na.py", line 11, in <module>
    MissingIndicator(missing_values=pd.NA).fit_transform(X)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^
  ...
  File "/home/arthur/scikit-learn/sklearn/impute/_base.py", line 995, in _fit
    X = self._validate_input(X, in_fit=True)
  File "/home/arthur/scikit-learn/sklearn/impute/_base.py", line 936, in _validate_input
    X = validate_data(
  ...
  File "/home/arthur/scikit-learn/sklearn/utils/validation.py", line 118, in _assert_all_finite
    if _object_dtype_isnan(X).any():
       ~~~~~~~~~~~~~~~~~~~^^^
  File "/home/arthur/scikit-learn/sklearn/utils/fixes.py", line 41, in _object_dtype_isnan
    return X != X
  File "pandas/_libs/missing.pyx", line 415, in pandas._libs.missing.NAType.__bool__
TypeError: boolean value of NA is ambiguous
```

With `X = pd.DataFrame({"n": pd.array([1, None, 3], dtype="Int64")})` instead: `ValueError: Input X contains NaN. MissingIndicator does not accept missing values encoded as NaN natively...`.

## Root cause and possible fix

`sklearn/impute/_base.py:931-934` (commit below):

```python
if not is_scalar_nan(self.missing_values):
    ensure_all_finite = True
else:
    ensure_all_finite = "allow-nan"
```

`SimpleImputer._validate_input` (`sklearn/impute/_base.py:340`) uses `is_pandas_na(self.missing_values) or is_scalar_nan(self.missing_values)`. Using the same condition here, plus listing `pandas.NA` in the `missing_values` docstring, fixes both cases (`_get_mask` already maps `pd.NA` to NaN for numeric arrays, `sklearn/impute/_base.py:32`).

## Versions

```
sklearn: 1.10.dev0, commit 289c4d5cfa
python: 3.13.14, numpy: 2.5.3, scipy: 1.18.1, platform: linux
pandas: 3.0.6
```
