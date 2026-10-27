# `GaussianMixture` with float32 input: wrong fit or crash for `tied`, `diag` and `spherical` when features are not centered

```json
{
    "estimators": ["GaussianMixture", "BayesianGaussianMixture"],
    "existing_issue": "https://github.com/scikit-learn/scikit-learn/issues/30382",
    "reproduced": true,
    "priority": 2,
    "difficulty": 2,
    "decision": 1,
    "decision_tags": ["float-order", "perf-low"]
}
```

## Classification

- **Priority (2)**: float32 data with a mean ~1000x its spread (a common situation: prices, years, sensor readings) gives a degraded fit for `tied`/`spherical` and a misleading "ill-defined empirical covariance" error for `diag`. Float32 inputs stay float32 since 1.7 (#30415), so this got more likely. Workarounds exist (center the data, or pass float64) and `covariance_type="full"` (the default) is not affected, hence 2.
- **Difficulty (2)**: replace the `E[x^2] - mean^2` expansions by centered computations in the M-step and in the log-probability, in several functions, while keeping the memory and speed acceptable.
- **Decision (1)**: `float-order`: results change at round-off level for well-conditioned data. `perf-low`: the centered formulas cost one extra pass over `X` per component (the `full` branch already does it).

## Upstream search

#30382 (open, "Diagonal covariance vectors might contain unreasonably negative values when the input datatype is np.float32") reports the `diag`/`spherical` M-step cancellation, with discussion of float64 upcasting vs clipping. #30401 (closed PR) tried clipping negative variances. #30415 (merged in 1.7) made all computations run in float32 for float32 input. #34277 (merged, unreleased) rewrote the `tied` log-probability as an expansion. Queries: "GaussianMixture float32", "GaussianMixture diag covariance negative", "GaussianMixture cancellation".

This file adds to #30382: the `tied` covariance estimate and the log-probabilities of `tied`/`diag`/`spherical` (`_estimate_log_gaussian_prob`) have the same cancellation, and the fix discussed there (a centered weighted variance) is exact, contrary to a doubt raised in the thread.

## Describe the bug and its user-facing impact

For `covariance_type` in `{"tied", "diag", "spherical"}`, `GaussianMixture` computes variances as `E[x^2] - mean^2` and squared Mahalanobis distances as `||x||^2 - 2 x.mu + ||mu||^2`, instead of from the centered values `x - mu`. When `|mean| >> std`, both terms are close and their difference loses `log10(mean^2 / std^2)` digits. In float32 (~7 digits), an offset of 1000 for a unit spread already loses everything: `diag` fails with `ValueError: Fitting the mixture model failed because some components have ill-defined empirical covariance` (negative variances), `tied` and `spherical` silently converge to a worse model. In float64 the same happens around an offset of 1e7 (1e8 fails for all three types). `covariance_type="full"` centers `X` explicitly and is fine.

## Steps/Code to Reproduce

```python
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.mixture import GaussianMixture

X, _ = make_blobs(n_samples=600, centers=3, n_features=3, random_state=0)
X += 1000  # e.g. a measurement around 1000 with unit spread
for cov in ["full", "tied", "diag", "spherical"]:
    res = []
    for dtype in [np.float64, np.float32]:
        gm = GaussianMixture(3, covariance_type=cov, random_state=0)
        try:
            res.append(f"{gm.fit(X.astype(dtype)).score(X):.4f}")
        except ValueError as e:
            res.append(f"ValueError: {str(e)[:60]}...")
    print(f"{cov:9s} float64: {res[0]}  float32: {res[1]}")
```

## Expected Results

Same average log-likelihood in float32 and float64 up to ~1e-4, as for `full`. A GMM is translation invariant, so the offset should not matter.

## Actual Results

```
full      float64: -5.3013  float32: -5.3013
tied      float64: -5.3107  float32: -5.3366
diag      float64: -5.3108  float32: ValueError: Fitting the mixture model failed because some components hav...
spherical float64: -5.3124  float32: -5.3742
```

## Root cause and possible fix

On `sklearn/mixture/_gaussian_mixture.py` at the commit below:

- M-step: `_estimate_gaussian_covariances_tied` (line 221, `X.T @ X - nk * means.T @ means`) and `_estimate_gaussian_covariances_diag` (line 250, `resp.T @ (X * X) / nk - means**2`, also used by `spherical`).
- E-step / `score_samples` / `predict`: `_estimate_log_gaussian_prob` expands the squared distance for `tied` (line 533), `diag` (line 541) and `spherical`. Taking the float64-fitted parameters, casting them and `X` to float32, and calling `score_samples` gives errors up to 0.34 (`tied`), 0.38 (`diag`) and 0.44 (`spherical`) nats per sample on the data above, vs 4e-4 for `full`.

Fix: compute both from centered data, as the `full` branch does: `sum_n resp[n, k] * (x_n - mu_k)**2 / nk[k]` is the exact weighted variance (the means are the resp-weighted means), and `((X - mu_k) * prec_chol_k)**2` for the log-probabilities. This needs a loop over components (or chunking) to avoid an `(n_samples, n_components, n_features)` temporary, with a modest slowdown. `BayesianGaussianMixture` uses the same helpers: on the data above, its float32 fit is off for `tied` (score -5.3727 vs -5.3447 in float64), while `diag` and `spherical` are saved by the covariance prior added to the estimate.

## Versions

```
sklearn: 1.10.dev0, commit 289c4d5cfa
python: 3.13.14, numpy: 2.5.3, scipy: 1.18.1, platform: linux
```
