# f_classif returns negative or -inf F-statistics for features with a large offset (catastrophic cancellation)

```json
{
    "estimators": ["f_classif", "f_oneway", "SelectKBest", "SelectPercentile", "SelectFdr"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 2,
    "difficulty": 2,
    "decision": 1,
    "decision_tags": ["float-order", "perf-low"]
}
```

## Classification

- **Priority (2)**: silent wrong scores on the default score function of the univariate selectors: an informative feature gets F = -998 or -inf and is ranked last. It needs a feature whose standard deviation is tiny compared to its mean (coefficient of variation below ~1e-7, e.g. `1e8 + noise`), which is rare but plausible (raw IDs, timestamps in ns, sensor readings with a large constant offset), and the results look plausible.
- **Difficulty (2)**: compute the sums of squares on centered data. Easy for dense input; for sparse input, it needs `mean_variance_axis` style per-class computations instead of the raw moments, without densifying.
- **Decision (1)**: changes the floating-point results slightly for all inputs (`float-order`), and centering costs an extra pass over the data (`perf-low`).

## Upstream search

- `gh issue list --search "f_oneway"`, `"f_classif precision"`, `"f_classif large values"`, `"f_classif negative"`: no issue about this (#994, closed 2012, was about NaN scores in selectors).
- #11395 (open) reports the same kind of cancellation in `r_regression`/`f_regression` (constant columns giving `sqrt` of negative values); it does not mention `f_classif`, whose code path (`f_oneway`) is separate. See also `issues/feature_selection-r-regression-cancellation-large-offset.md`.

## Describe the bug and its user-facing impact

`f_oneway` (used by `f_classif`) computes the within- and between-class sums of squares from raw moments: `sum(x**2) - sum(x)**2 / n`. When the feature mean is large compared to its spread, both terms are ~`n * mean**2` and their difference loses all significant digits. The F-statistic drifts as the offset grows, then becomes negative, `-inf` or NaN. SelectKBest (default `score_func=f_classif`) then drops the most informative feature. Shifting a feature by a constant should not change its F-statistic at all.

float32 input breaks much earlier, because the sums are accumulated in the input dtype. With a unit-variance feature shifted by 1e3 (the scale of a year or a price), the F-statistic is 198.5 instead of 236.2. Shifted by 1e5, it is -998.0. float64 gives 236.2 in both cases (checked during the priority review, 2026-10-23).

## Steps/Code to Reproduce

```python
import numpy as np
from scipy import stats
from sklearn.feature_selection import SelectKBest, f_classif

rng = np.random.RandomState(0)
y = rng.randint(0, 2, 1000)
x = y + rng.randn(1000)  # informative feature
for offset in [0, 1e6, 1e7, 1e8]:
    X = (offset + x)[:, None]
    print(f"offset={offset:.0e}  f_classif: {f_classif(X, y)[0][0]:9.3f}  "
          f"scipy f_oneway: {stats.f_oneway(X[y == 0], X[y == 1]).statistic[0]:.3f}")

X = np.column_stack([1e8 + x, rng.randn(1000)])  # informative, noise
print("SelectKBest(k=1) keeps feature:", SelectKBest(f_classif, k=1).fit(X, y).get_support(indices=True))
print("f_classif scores, p-values:", f_classif(X, y))
```

## Expected Results

The F-statistic is ~338.658 for every offset, as with `scipy.stats.f_oneway`, and SelectKBest keeps feature 0.

## Actual Results

```
/home/arthur/scikit-learn/sklearn/feature_selection/_univariate_selection.py:111: RuntimeWarning: divide by zero encountered in divide
  f = msb / msw
/home/arthur/scikit-learn/sklearn/feature_selection/_univariate_selection.py:111: RuntimeWarning: divide by zero encountered in divide
  f = msb / msw
offset=0e+00  f_classif:   338.658  scipy f_oneway: 338.658
offset=1e+06  f_classif:   338.645  scipy f_oneway: 338.658
offset=1e+07  f_classif:   338.305  scipy f_oneway: 338.658
offset=1e+08  f_classif:  -998.000  scipy f_oneway: 338.658
SelectKBest(k=1) keeps feature: [1]
f_classif scores, p-values: (array([      -inf, 0.00027923]), array([       nan, 0.98667119]))
```

## Root cause and possible fix

`sklearn/feature_selection/_univariate_selection.py:94-98`:

```python
ss_alldata = sum(safe_sqr(a).sum(axis=0) for a in args)
...
sstot = ss_alldata - square_of_sums_alldata / float(n_samples)
```

and the same pattern for `ssbn`. Subtracting the global feature mean before computing the moments (dense case) removes the cancellation and doesn't change the statistic mathematically. For sparse input, centering would densify, but per-class means and variances from `mean_variance_axis` (which does not use the raw-moment formula) give `sswn = sum_k n_k * var_k` and `ssbn = sum_k n_k * (mean_k - mean)**2` without cancellation. That second code path is why I put difficulty 2.

## Versions

```
sklearn: 1.10.dev0, commit 289c4d5cfa (upstream/main; reproduced on local checkout f570497251, no diff in sklearn/feature_selection/_univariate_selection.py)
python: 3.13, numpy: 2.5.3, scipy: 1.18.1, platform: linux
```
