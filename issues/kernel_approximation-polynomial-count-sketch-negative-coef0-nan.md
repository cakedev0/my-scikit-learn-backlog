# PolynomialCountSketch accepts a negative coef0 and returns all-NaN features

```json
{
    "estimators": ["PolynomialCountSketch"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 1,
    "difficulty": 1,
    "decision": 1,
    "decision_tags": ["new-error"]
}
```

## Classification

- **Priority (1)**: the output is entirely NaN, with only a numpy RuntimeWarning; but negative `coef0` is unusual for this transformer.
- **Difficulty (1)**: change the parameter constraint of `coef0` to `Interval(Real, 0, None, closed="left")` and document it.
- **Decision (1)**: `new-error`: negative `coef0` would raise at fit instead of returning NaN (it never gave useful results).

## Upstream search

Queries: "PolynomialCountSketch", "PolynomialCountSketch coef0", "PolynomialCountSketch coef0 negative", "PolynomialCountSketch nan".

No match. #34213 (merged) only fixed the documented type of `coef0` (int -> float). #24217 (merged) added the current parameter constraints. #34919/#34920/#34923 are about sparse performance and the `bitHash_` docstring.

## Describe the bug and its user-facing impact

`PolynomialCountSketch` approximates `(gamma * <x, y> + coef0)^degree` by appending a constant feature `sqrt(coef0)` to `X` before sketching. Its parameter constraints accept any real `coef0` (`Interval(Real, None, None, closed="neither")`), and the docstring does not restrict it either, so `coef0 < 0` passes validation and `np.sqrt(coef0)` produces NaN in every output feature. The kernel with a negative `coef0` is not positive semi-definite in general, so there is no real feature map to approximate; the transformer should reject it.

## Steps/Code to Reproduce

```python
import numpy as np
from sklearn.kernel_approximation import PolynomialCountSketch

X = np.random.RandomState(0).rand(3, 2)
print(PolynomialCountSketch(coef0=-1.0, n_components=4, random_state=0).fit_transform(X))
```

## Expected Results

An `InvalidParameterError` saying that `coef0` must be non-negative.

## Actual Results

```
/home/arthur/scikit-learn/sklearn/kernel_approximation.py:208: RuntimeWarning: invalid value encountered in sqrt
  [X_gamma, np.sqrt(self.coef0) * np.ones((X_gamma.shape[0], 1))]
[[nan nan nan nan]
 [nan nan nan nan]
 [nan nan nan nan]]
```

## Root cause and possible fix

`PolynomialCountSketch._parameter_constraints` (`sklearn/kernel_approximation.py:129`) allows any real `coef0`, and `transform` uses `np.sqrt(self.coef0)` (`:202` for sparse, `:208` for dense). Fix: `"coef0": [Interval(Real, 0, None, closed="left")]`, and say "must be non-negative" in the docstring.

## Versions

```
sklearn: 1.10.dev0, commit 289c4d5cfa (upstream/main; ran on f570497251, no diff in the files involved)
python: 3.13.14, numpy: 2.5.3, scipy: 1.18.1, platform: linux
```
