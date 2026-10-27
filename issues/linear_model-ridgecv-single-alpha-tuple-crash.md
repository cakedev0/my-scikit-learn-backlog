# RidgeCV/RidgeClassifierCV crash on `alphas=(a,)` and mutate a one-element `alphas` list in place

```json
{
    "estimators": ["RidgeCV", "RidgeClassifierCV"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 1,
    "difficulty": 1,
    "decision": 0,
    "decision_tags": []
}
```

## Classification

- **Priority (1)**: a crash with an unhelpful error on valid input, but a one-element grid is an unusual choice for a CV estimator, and passing a list works.
- **Difficulty (1)**: validate a local variable instead of assigning into `self.alphas`.
- **Decision (0)**: no trade-off.

## Upstream search

Queries (issues and PRs): "RidgeCV alphas tuple", "RidgeCV single alpha", "RidgeCV alphas". No match.

## Describe the bug and its user-facing impact

`_BaseRidgeCV.fit` validates a one-element `alphas` by writing the validated value back into the user's parameter: `self.alphas[0] = check_scalar_alpha(self.alphas[0], "alphas")` (`sklearn/linear_model/_ridge.py:2470` on upstream/main). This fails for a tuple (`TypeError`), which is accepted for longer grids, and for a 0-d array (`IndexError`), which passes the `"array-like"` parameter constraint. For lists and 1-element arrays it modifies the estimator's parameter in place during `fit`, which estimators should not do.

A one-element grid can show up when the grid is built programmatically (e.g. `alphas=tuple(candidates)` after filtering).

## Steps/Code to Reproduce

```python
import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import RidgeCV

X, y = make_regression(n_samples=30, n_features=3, random_state=0)
print(RidgeCV(alphas=(1.0, 10.0)).fit(X, y).alpha_)  # works
print(RidgeCV(alphas=[1.0]).fit(X, y).alpha_)  # works
for alphas in [(1.0,), np.array(1.0)]:
    try:
        RidgeCV(alphas=alphas).fit(X, y)
    except Exception as e:
        print(f"alphas={alphas!r}: {type(e).__name__}: {e}")
```

## Expected Results

`alpha_ == 1.0` in all cases, as for `alphas=[1.0]`.

## Actual Results

```
1.0
1.0
alphas=(1.0,): TypeError: 'tuple' object does not support item assignment
alphas=array(1.): IndexError: too many indices for array: array is 0-dimensional, but 1 were indexed
```

## Root cause and possible fix

`sklearn/linear_model/_ridge.py:2464-2471` (upstream/main):

```python
if isinstance(self.alphas, (np.ndarray, list, tuple)):
    n_alphas = 1 if np.ndim(self.alphas) == 0 else len(self.alphas)
    if n_alphas != 1:
        for index, alpha in enumerate(self.alphas):
            alpha = check_scalar_alpha(alpha, f"alphas[{index}]")
    else:
        self.alphas[0] = check_scalar_alpha(self.alphas[0], "alphas")
```

The `n_alphas == 1` branch indexes `[0]` although `n_alphas` is also 1 for a 0-d array, and assigns into the parameter. Fix: validate `np.ravel(self.alphas)` element-wise in both branches without assignment (the result is already discarded in the `n_alphas != 1` branch).

## Versions

```
sklearn: 1.10.dev0, commit 289c4d5cfa (upstream/main; reproduced on a local build whose _ridge.py only differs in docstrings)
python: 3.13, numpy: 2.5.3, scipy: 1.18.1, platform: linux
```
