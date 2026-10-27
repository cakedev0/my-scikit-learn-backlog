# KernelDensity bandwidth="scott"/"silverman" ignores the scale of the data

```json
{
    "estimators": ["KernelDensity"],
    "existing_issue": "https://github.com/scikit-learn/scikit-learn/issues/25623",
    "reproduced": true,
    "priority": 3,
    "difficulty": 2,
    "decision": 1,
    "decision_tags": ["output-change", "unclear-spec"]
}
```

## Classification

- **Priority (3)**: the rule-of-thumb bandwidths are only right for unit-variance data; on any other scale the density is wildly over- or under-smoothed (1000x too small a bandwidth below), silently. The user guide presents them as the way to avoid tuning the bandwidth by hand.
- **Difficulty (2)**: multiply by the data spread; the subtle part is what to do in several dimensions with an isotropic kernel (scipy uses the full covariance, which `KernelDensity` can't).
- **Decision (1)**: `output-change` for everyone using these options (wanted, since current results are wrong unless data is standardized). `unclear-spec`: in d > 1, which spread to use (mean std, geometric mean of stds, ...) must be decided, see the discussion in #27971.

## Upstream search

- `KernelDensity scott bandwidth`: #25623 (open, this bug), #26658 (open, duplicate: "Automatic bandwidth calculation valid only for normalized data"), #27971 (open PR "FIX `KernelDensity` incorrectly handling bandwidth", towards both issues, reviewed but stalled), #34991 (open PR, unrelated input validation problem of the same code, see `neighbors-kde-scott-silverman-list-input.md`), #22993 (merged, added the feature).

Filed for the backlog since the issues and the PR are still open.

## Describe the bug and its user-facing impact

`bandwidth="scott"` sets `bandwidth_ = n ** (-1 / (d + 4))` and `"silverman"` a similar factor: these are the factors that should multiply the standard deviation of the data (as in `scipy.stats.gaussian_kde` and every reference on these rules), but `KernelDensity` uses them as absolute bandwidths. With data in meters or dollars, the estimated density is a sum of spikes; with data in a tiny unit it's flat.

## Steps/Code to Reproduce

```python
import numpy as np
from scipy.stats import gaussian_kde
from sklearn.neighbors import KernelDensity

rng = np.random.RandomState(0)
X = rng.normal(scale=1000.0, size=(500, 1))  # e.g. a distance in meters

kde = KernelDensity(bandwidth="scott").fit(X)
ref = gaussian_kde(X.ravel(), bw_method="scott")
print("sklearn bandwidth_:", kde.bandwidth_)
print("scipy bandwidth:   ", np.sqrt(ref.covariance[0, 0]))
x = np.array([[0.0], [10.0]])
print("sklearn density:", np.exp(kde.score_samples(x)))
print("scipy density:  ", ref(x.ravel()))
```

## Expected Results

A bandwidth of ~288 (Scott's factor times the std of ~1000) and densities close to scipy's ~3.7e-4 at both points.

## Actual Results

```
sklearn bandwidth_: 0.2885399811814427
scipy bandwidth:    288.2965786218972
sklearn density: [2.25964262e-40 6.16066899e-04]
scipy density:   [0.00037263 0.00037223]
```

## Root cause and possible fix

`sklearn/neighbors/_kde.py:219-225` computes only the sample-size factor. Fix: multiply it by a spread estimate of `X` (and of `sample_weight` when given), e.g. the std in 1D. For d > 1 with an isotropic kernel, there's no exact equivalent of scipy's covariance scaling; #27971 explores options. The current behavior is right only for standardized data, so the docstring should at least say so if the fix is delayed.

## Versions

```
sklearn: 1.10.dev0, commit f570497251 (no diff in sklearn/neighbors vs upstream/main 289c4d5cfa)
python: 3.13, numpy: 2.5.3, scipy: 1.18.1, platform: linux
```
