# r_regression / f_regression silently score an informative feature as 0 when it has a large offset

```json
{
    "estimators": ["r_regression", "f_regression", "SelectKBest", "SelectPercentile"],
    "existing_issue": "https://github.com/scikit-learn/scikit-learn/issues/11395",
    "reproduced": true,
    "priority": 1,
    "difficulty": 1,
    "decision": 1,
    "decision_tags": ["float-order", "perf-low"]
}
```

## Classification

- **Priority (1)**: silent, but it needs a feature whose spread is tiny compared to its mean (coefficient of variation below ~1e-7 in float64; float32 input was checked to stay accurate up to an offset of 1e6 with unit noise), which is very rare in practice. A `RuntimeWarning` from `sqrt` hints at it. (priority review, 2026-10-23: was 2)
- **Difficulty (1)**: for dense input, center X before computing norms; for sparse input, `mean_variance_axis` gives the variance without the raw-moment formula. Both are short changes.
- **Decision (1)**: slightly different floating-point results for all inputs (`float-order`); centering dense X needs an extra copy or pass (`perf-low`).

## Upstream search

- #11395 (open, "f_regression takes square root of negative values when constant columns are present", labels Bug, Needs Investigation): same root cause, but only discussed for exactly constant columns, where the NaN is mapped to 0 by `force_finite=True` and the result (r = 0) is right anyway. The new argument here: the same cancellation also hits **non-constant, informative** features and silently turns their correlation into 0, so it is not only a warning problem.
- PR #34834 (closed) tried a clamp `np.maximum(..., 0)`; it would not fix this case, the variance would still be garbage. Earlier attempts #33642 and #34197 were closed too.
- `gh issue list --search "r_regression precision"`, `"f_regression numerical precision"`: nothing else.
- See also `issues/feature_selection-f-classif-cancellation-large-offset.md` (same problem in `f_classif`).

## Describe the bug and its user-facing impact

`r_regression` computes the norm of each centered feature as `sqrt(sum(x**2) - n * mean(x)**2)`. When the mean is large compared to the standard deviation (here `1e8 + noise` with std ~1.4), both terms are ~1e19 and their difference is pure rounding noise, possibly negative. `sqrt` then returns NaN, `force_finite=True` replaces it with 0, and `f_regression` reports F = 0, p = 1. SelectKBest/SelectPercentile with `f_regression` drop the feature. With a slightly smaller offset the norm is wrong but finite, giving a wrong correlation instead.

## Steps/Code to Reproduce

```python
import numpy as np
from sklearn.feature_selection import SelectKBest, f_regression, r_regression

rng = np.random.RandomState(0)
y = rng.randn(1000)
x = y + rng.randn(1000)  # correlation ~0.7 with y
X = np.column_stack([1e8 + x, rng.randn(1000)])  # informative, noise

print("np.corrcoef:  ", np.array([np.corrcoef(X[:, j], y)[0, 1] for j in range(2)]).round(3))
print("r_regression: ", r_regression(X, y).round(3))
print("f_regression: ", f_regression(X, y))
print("SelectKBest(k=1) keeps feature:", SelectKBest(f_regression, k=1).fit(X, y).get_support(indices=True))
```

## Expected Results

`r_regression` returns ~[0.703, -0.038] like `np.corrcoef` (a constant shift of a feature doesn't change its correlation), and SelectKBest keeps feature 0.

## Actual Results

```
/home/arthur/scikit-learn/sklearn/feature_selection/_univariate_selection.py:375: RuntimeWarning: invalid value encountered in sqrt
  X_norms = np.sqrt(row_norms(X.T, squared=True) - n_samples * X_means**2)
/home/arthur/scikit-learn/sklearn/feature_selection/_univariate_selection.py:375: RuntimeWarning: invalid value encountered in sqrt
  X_norms = np.sqrt(row_norms(X.T, squared=True) - n_samples * X_means**2)
/home/arthur/scikit-learn/sklearn/feature_selection/_univariate_selection.py:375: RuntimeWarning: invalid value encountered in sqrt
  X_norms = np.sqrt(row_norms(X.T, squared=True) - n_samples * X_means**2)
np.corrcoef:   [ 0.703 -0.038]
r_regression:  [ 0.    -0.038]
f_regression:  (array([0.        , 1.41938508]), array([1.        , 0.23378789]))
SelectKBest(k=1) keeps feature: [1]
```

## Root cause and possible fix

`sklearn/feature_selection/_univariate_selection.py:375`:

```python
X_norms = np.sqrt(row_norms(X.T, squared=True) - n_samples * X_means**2)
```

Use `sqrt(n_samples * var)` with `var` from `np.var` (dense) or `sklearn.utils.sparsefuncs.mean_variance_axis` (sparse), which both subtract the mean before squaring. This also fixes the constant-column case of #11395 (variance exactly 0 up to rounding of the mean, never negative). A clamp to 0 alone hides the warning but keeps the wrong values.

## Versions

```
sklearn: 1.10.dev0, commit 289c4d5cfa (upstream/main; reproduced on local checkout f570497251, no diff in sklearn/feature_selection/_univariate_selection.py)
python: 3.13, numpy: 2.5.3, scipy: 1.18.1, platform: linux
```
