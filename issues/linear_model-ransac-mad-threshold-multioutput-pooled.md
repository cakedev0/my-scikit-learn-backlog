# RANSACRegressor default `residual_threshold` pools all targets, so multi-output fits can flag no outliers

```json
{
    "estimators": ["RANSACRegressor"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 2,
    "difficulty": 2,
    "decision": 1,
    "decision_tags": ["output-change", "unclear-spec"]
}
```

## Classification

- **Priority (2)**: silent wrong result under the default `residual_threshold=None` for any multi-output `y` whose targets have different locations or scales: RANSAC keeps every sample as an inlier and returns the plain least-squares fit. Multi-output RANSAC is not the common case, and passing an explicit `residual_threshold` is a cheap workaround.
- **Difficulty (2)**: centering per target is a one-line change, but choosing a threshold that matches the summed multi-output loss needs a small decision (see below).
- **Decision (1)**: `unclear-spec` because there is no documented definition of the MAD for 2D `y`; `output-change` because multi-output fits with the default threshold change.

## Upstream search

- `gh search issues ... RANSAC residual_threshold median`: #16147 (closed) is about the MAD being 0 when most `y` values are equal, a different problem on 1D `y`.
- `gh search issues ... RANSAC multi output`: nothing relevant.
- `gh search issues ... RANSACRegressor` (30 results): #15836 (open, loss ignores `sample_weight`), #32908 (open, `score` on too small inlier sets), doc PRs on `random_state`. None about the threshold with 2D `y`.

## Describe the bug and its user-facing impact

With `residual_threshold=None`, `RANSACRegressor` uses the MAD of `y` as threshold. For 2D `y`, the code computes `np.median(np.abs(y - np.median(y)))` on the flattened array, i.e. it subtracts one global median from all targets. If the targets have different means (or scales), the "MAD" mostly measures the gap between targets, the threshold becomes huge, and every sample is an inlier. RANSAC then silently degrades to `LinearRegression` on all data, outliers included.

In the reproducer, the same signal with 10% gross outliers is fitted as one target and as two targets (second one shifted by +1000). Single-output RANSAC flags the 20 outliers and recovers the slope 3. Multi-output RANSAC flags none and returns slope 5.4 for both targets.

## Steps/Code to Reproduce

```python
import numpy as np
from sklearn.linear_model import RANSACRegressor

rng = np.random.RandomState(0)
X = rng.randn(200, 1)
y = 3 * X[:, 0] + 0.1 * rng.randn(200)
y[:20] += 50  # 10% gross outliers
Y = np.column_stack([y, y + 1000])  # second target: same signal, shifted mean
r1 = RANSACRegressor(random_state=0).fit(X, y)
r2 = RANSACRegressor(random_state=0).fit(X, Y)
print("single-output: outliers flagged", (~r1.inlier_mask_).sum(), "coef", r1.estimator_.coef_)
print("2-output:      outliers flagged", (~r2.inlier_mask_).sum(), "coef", r2.estimator_.coef_.ravel())
```

## Expected Results

Adding a constant to one target should not change which samples are outliers: the two-target fit should flag the same 20 samples and return a slope close to 3 for both targets. The docstring describes the default threshold as "the MAD (median absolute deviation) of the target values `y`", and a MAD is location invariant.

## Actual Results

```
single-output: outliers flagged 20 coef [3.00391003]
2-output:      outliers flagged 0 coef [5.39566251 5.39566251]
```

## Root cause and possible fix

`sklearn/linear_model/_ransac.py:387`:

```python
residual_threshold = np.median(np.abs(y - np.median(y)))
```

`np.median(y)` without `axis` reduces over all targets. Centering per target (`np.median(y, axis=0)`) fixes the location problem shown above.

There is a second, smaller mismatch: for 2D `y` the built-in losses sum the per-target residuals (`np.sum(np.abs(y_true - y_pred), axis=1)`), so a sample's loss grows with the number of targets while the threshold does not. A consistent choice for `loss="absolute_error"` would be the sum of the per-target MADs. Someone has to pick the definition, hence `unclear-spec`; the per-target centering alone is already a clear improvement.

## Versions

```
sklearn: 1.10.dev0, commit 289c4d5cfa
python: 3.13, numpy: 2.5.3, scipy: 1.18.1, platform: linux
```
