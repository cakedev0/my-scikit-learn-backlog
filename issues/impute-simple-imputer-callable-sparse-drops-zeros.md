# `SimpleImputer(strategy=callable)` ignores zeros on sparse input

```json
{
    "estimators": ["SimpleImputer"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 2,
    "difficulty": 1,
    "decision": 0,
    "decision_tags": []
}
```

## Classification

- **Priority (2)**: silent wrong imputation values on sparse input, where zeros are by definition the most common value. Callable strategies are newer (1.5) and less used than the string ones.
- **Difficulty (1)**: the callable branch needs the zeros added back before calling the function.
- **Decision (0)**: the docstring defines the behavior and the dense path follows it.

## Upstream search

Queries: "SimpleImputer callable sparse", "SimpleImputer callable strategy", "SimpleImputer sparse custom strategy zeros".

No match. #28053 (merged) added callable strategies; its tests only cover dense input.

## Describe the bug and its user-facing impact

The `strategy` docstring says: "If an instance of Callable, then replace missing values using the scalar statistic returned by running the callable over a dense 1d array containing non-missing values of each column."

On sparse input, the callable only receives the non-zero, non-missing values: both implicit and explicit zeros are dropped. So `SimpleImputer(strategy=np.mean)` and `SimpleImputer(strategy="mean")` give different statistics on sparse data, and dense and sparse inputs give different results for the same callable. For a typical sparse column (mostly zeros), the imputed value can be far off.

## Steps/Code to Reproduce

```python
import numpy as np
from scipy import sparse
from sklearn.impute import SimpleImputer

X = np.array([[0.0], [0.0], [0.0], [4.0], [np.nan]])
imp = SimpleImputer(strategy=np.mean)
print("dense :", imp.fit(X).statistics_)
print("sparse:", imp.fit(sparse.csr_array(X)).statistics_)
print("strategy='mean', sparse:", SimpleImputer(strategy="mean").fit(sparse.csr_array(X)).statistics_)
```

## Expected Results

`[1.]` in the three cases.

## Actual Results

```
dense : [1.]
sparse: [4.]
strategy='mean', sparse: [1.]
```

## Root cause and possible fix

In `SimpleImputer._sparse_fit` (`sklearn/impute/_base.py:487-513`), the zeros are removed from `column` (line 494) so that the string strategies can handle them via `n_zeros`. The callable branch (line 513) then calls `self.strategy(column)` on this reduced array.

Fix: in the callable branch, call the function on the column with the zeros put back, e.g. `self.strategy(np.concatenate([column, np.zeros(n_zeros, dtype=column.dtype)]))`. This densifies one column at a time, which is what the docstring promises ("a dense 1d array"). The existing test (`sklearn/impute/tests/test_impute.py`, `strategy=np.min`) only covers dense input.

## Versions

```
sklearn: 1.10.dev0, commit f570497251 (code identical on upstream/main 289c4d5cfa)
python: 3.13.14, numpy: 2.5.3, scipy: 1.18.1, platform: linux
```
