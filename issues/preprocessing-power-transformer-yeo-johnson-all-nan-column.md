# `PowerTransformer(method="yeo-johnson")` fails with an unhelpful unpacking error on an all-NaN feature

```json
{
    "estimators": ["PowerTransformer", "power_transform"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 1,
    "difficulty": 1,
    "decision": 1,
    "decision_tags": ["unclear-spec"]
}
```

## Classification

- **Priority (1)**: a crash on an edge case (a column with no observed value, e.g. after a train/test split or in a `ColumnTransformer` over many columns), with the default `method`. The error message gives no hint about which column or why.
- **Difficulty (1)**: a check in `_yeo_johnson_optimize`, as `_box_cox_optimize` already does.
- **Decision (1)**: two reasonable fixes: raise the same clear error as box-cox, or keep the column as NaN like `StandardScaler` and `QuantileTransformer` do (`lambdas_[i] = 1`, identity, as for constant features). Someone has to pick.

## Upstream search

Queries (issues and PRs): "PowerTransformer all nan column", "boolean value of NA is ambiguous".

- #26303 (closed): same problem for `method="box-cox"`, fixed by #26400 (merged), which added `ValueError("Column must not be all nan.")` to `_box_cox_optimize` only. The yeo-johnson path was not changed.
- #27499 (closed): constant columns, different cause.

## Describe the bug and its user-facing impact

`PowerTransformer` accepts NaN (`allow_nan` tag) and ignores it during fit. With `method="yeo-johnson"` (the default), a feature with only NaN reaches `scipy.stats.yeojohnson` as an empty array, which returns an empty tuple, and fit fails with `ValueError: not enough values to unpack (expected 2, got 0)`. With `method="box-cox"`, the same input raises a clear error since #26400, and the other NaN-tolerant scalers keep the column as NaN.

## Steps/Code to Reproduce

```python
import numpy as np
from sklearn.preprocessing import PowerTransformer, QuantileTransformer, StandardScaler

rng = np.random.RandomState(0)
X = np.c_[rng.rand(20, 2), np.full(20, np.nan)]  # 3rd feature entirely missing

for est in [StandardScaler(), QuantileTransformer(n_quantiles=10)]:
    print(type(est).__name__, "->", est.fit_transform(X)[0])
for method in ["box-cox", "yeo-johnson"]:
    try:
        PowerTransformer(method=method).fit(X)
    except Exception as e:
        print(f"PowerTransformer(method={method!r}) -> {type(e).__name__}: {e}")
```

## Expected Results

Either NaN passed through for the empty column (like the other scalers), or at least the clear error that box-cox raises.

## Actual Results

```
StandardScaler -> [0.20760848 0.2542155         nan]
QuantileTransformer -> [0.55555556 0.55555556        nan]
PowerTransformer(method='box-cox') -> ValueError: Column must not be all nan.
PowerTransformer(method='yeo-johnson') -> ValueError: not enough values to unpack (expected 2, got 0)
```

(RuntimeWarnings from `StandardScaler`/`QuantileTransformer` about empty slices trimmed.)

## Root cause and possible fix

`sklearn/preprocessing/_data.py:3587-3588` (`_yeo_johnson_optimize`):

```python
x = x[~np.isnan(x)]
_, lmbda = stats.yeojohnson(x, lmbda=None)
```

has no guard for an empty `x`, unlike `_box_cox_optimize` (`_data.py:3569-3571`). The minimal fix is the same `if np.all(mask): raise ValueError("Column must not be all nan.")`. Alternatively, `_fit` (`_data.py:3420-3426`) could treat an all-NaN column like a constant one (`lambdas_[i] = 1.0`, identity), which matches `StandardScaler` and `QuantileTransformer`.

## Versions

```
sklearn: 1.10.dev0, commit f570497251 (sklearn/preprocessing/_data.py identical to upstream/main 289c4d5cfa)
python: 3.13.14, numpy: 2.5.3, scipy: 1.18.1, platform: linux
```
