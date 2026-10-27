# `StandardScaler.partial_fit` returns NaN `mean_`/`var_` when a feature is missing in a whole batch

```json
{
    "estimators": ["StandardScaler"],
    "existing_issue": "https://github.com/scikit-learn/scikit-learn/issues/35070",
    "reproduced": true,
    "priority": 1,
    "difficulty": 1,
    "decision": 0,
    "decision_tags": []
}
```

## Classification

- **Priority (1)**: needs `partial_fit` with a batch where a feature is entirely NaN (mostly one-sample batches with missing values); the result is a visible NaN column with `RuntimeWarning`s, `fit` is unaffected, and upstream maintainers already labelled the later-batch case Low Priority in #35070. (priority review, 2026-10-23: was 2)
- **Difficulty (1)**: handle per-feature zero counts in the update formulas, a few `xp.where` in `_incremental_mean_and_var` and one condition in the Cython sparse helper.
- **Decision (0)**: no trade-off, the expected values are the ones of `fit`.

## Upstream search

- #35070 (open, labels "Bug", "Low Priority", "Needs Investigation"): reports the later-batch case (dense `var_` becomes NaN when a new batch has an all-NaN feature). @lesteve called it an edge case and marked it low priority; the fix PR #35072 was auto-closed because the issue is not triaged as ready.
- New arguments for this report: (1) the first-batch case is not covered by #35070 and is worse: `mean_` is NaN too, and the sparse Cython path is affected as well (it was fixed in #16466 only for the later-batch case); (2) the natural trigger is online learning with one sample per `partial_fit` call, where any missing value breaks the scaler for good, which is not that much of an edge case for a `partial_fit` API.
- Other queries: "_incremental_mean_and_var nan", "partial_fit nan var_ StandardScaler": nothing else. The dense NaN handling was added in #11206 (2018, "Ignore NaNs in StandardScaler and scale"), which handles `last_sample_count == 0` for the variance only.

## Describe the bug and its user-facing impact

`StandardScaler` ignores NaNs when computing `mean_` and `var_`, and `fit` works with any NaN pattern. With `partial_fit`, the running statistics become NaN as soon as one batch contains only NaNs for a feature:

- if that batch is the first one seen for the feature (its count is still 0), `mean_` and `var_` of the feature are NaN forever, for the dense and the sparse code paths;
- if it comes later, `var_` (dense input) becomes NaN forever.

`scale_` is then NaN, and `transform` returns NaN for the whole column, silently apart from `RuntimeWarning: invalid value encountered in divide`. The natural trigger is online learning with one sample per `partial_fit` call: any missing value breaks the scaler for good.

## Steps/Code to Reproduce

```python
import numpy as np
from scipy import sparse
from sklearn.preprocessing import StandardScaler

# Feature 0 is missing in the first sample, feature 1 in the last one.
X = np.array([[np.nan, 1.0], [1.0, 2.0], [3.0, 4.0], [5.0, 7.0], [2.0, np.nan]])

# Online learning: one sample per partial_fit call.
scaler = StandardScaler()
for row in X:
    scaler.partial_fit(row.reshape(1, -1))

sparse_scaler = StandardScaler(with_mean=False)
for row in X:
    sparse_scaler.partial_fit(sparse.csr_array(row.reshape(1, -1)))

ref = StandardScaler().fit(X)
print("fit                 mean_:", ref.mean_, "var_:", ref.var_)
print("partial_fit dense   mean_:", scaler.mean_, "var_:", scaler.var_)
print("partial_fit sparse  var_:", sparse_scaler.var_)
```

## Expected Results

`partial_fit` over the rows gives the same statistics as `fit` (up to rounding), as it does when there are no NaNs or when NaNs never fill a whole batch column:

```
fit                 mean_: [2.75 3.5 ] var_: [2.1875 5.25  ]
```

## Actual Results

```
/home/arthur/scikit-learn/sklearn/utils/extmath.py:1200: RuntimeWarning: invalid value encountered in divide
  updated_mean = (last_sum + new_sum) / updated_sample_count
/home/arthur/scikit-learn/sklearn/utils/extmath.py:1205: RuntimeWarning: invalid value encountered in divide
  T = new_sum / new_sample_count
/home/arthur/scikit-learn/sklearn/utils/extmath.py:1229: RuntimeWarning: invalid value encountered in divide
  new_unnormalized_variance -= correction**2 / new_sample_count
/home/arthur/scikit-learn/sklearn/utils/extmath.py:1205: RuntimeWarning: invalid value encountered in divide
  T = new_sum / new_sample_count
/home/arthur/scikit-learn/sklearn/utils/extmath.py:1229: RuntimeWarning: invalid value encountered in divide
  new_unnormalized_variance -= correction**2 / new_sample_count
fit                 mean_: [2.75 3.5 ] var_: [2.1875 5.25  ]
partial_fit dense   mean_: [nan 3.5] var_: [nan nan]
partial_fit sparse  var_: [ nan 5.25]
```

## Root cause and possible fix

Dense path, `_incremental_mean_and_var` in `sklearn/utils/extmath.py` (upstream/main 289c4d5cfa):

- line 1172: `last_sum = last_mean * last_sample_count`. After a first all-NaN batch, the stored mean is `0 / 0 = nan`, and `nan * 0` is still NaN, so the mean never recovers.
- lines 1205 and 1229: `T = new_sum / new_sample_count` and `correction**2 / new_sample_count` are `0 / 0` when the current batch has only NaNs for a feature, which makes `new_unnormalized_variance` NaN. Line 1249 only handles the opposite case (`last_sample_count == 0`).

Sparse path, `_incr_mean_variance_axis0` in `sklearn/utils/sparsefuncs_fast.pyx`: the `new_n[i] > 0` branch (line 458) is right, but `is_first_pass` (line 444) is global, so a feature whose `last_n[i]` is 0 while another feature has data goes through line 461 with `last_mean[i] = nan`.

Fix: per feature, when `new_sample_count == 0` keep the previous mean and variance, and when `last_sample_count == 0` take the new ones (for the mean too, e.g. `last_sum = xp.where(last_sample_count == 0, 0, last_mean * last_sample_count)`). In the Cython helper, test `last_n[i] == 0` per feature instead of the global first pass. A test with one-sample batches containing NaNs, compared against `fit`, would cover both paths.

## Versions

```
sklearn: 1.10.dev0, commit 289c4d5cfa (upstream/main; local build f570497251, no diff on the files involved)
python: 3.13, numpy: 2.5.3, scipy: 1.18.1, platform: linux
```

See also: upstream #35070 (open, "Low Priority") reports the later-batch `var_` case of this bug, with an auto-closed fix PR #35072; `preprocessing-minmax-maxabs-partial-fit-all-nan-batch.md` (upstream #35143) has the same symptom in `MinMaxScaler`/`MaxAbsScaler`.
