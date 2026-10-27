# Lars, LassoLars and LassoLarsIC give wrong results with `copy_X=False` when no Gram matrix is used

```json
{
    "estimators": ["LassoLarsIC", "Lars", "LassoLars"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 1,
    "difficulty": 1,
    "decision": 0,
    "decision_tags": []
}
```

## Classification

- **Priority (1)**: silent wrong model (LassoLarsIC can select the empty model), but the trigger is rare: the non-default `copy_X=False` combined with a path without a Gram matrix (explicit `precompute=False`, or wide data with a user `noise_variance` in LassoLarsIC), on estimators that are not widely used. (priority review, 2026-10-23: was 2)
- **Difficulty (1)**: the callers only need a private copy of `X` (or keep using the copy made by `lars_path`) when they reuse `X` after the path; `orthogonal_mp` already does exactly this for multi-target `y`.
- **Decision (0)**: `copy_X=False` only promises that `X` may be overwritten, not that the fit is wrong. The fix costs one extra copy of `X` in a setting where the user asked to avoid it, but only for the multi-target Lars case and LassoLarsIC, where the current result is wrong anyway.

## Upstream search

Queries: "LassoLarsIC", "LassoLarsIC copy_X", "lars_path copy_X permute columns", "Lars copy_X=False wrong", "LassoLarsIC precompute", "Lars copy_X", "lars_path copy_X", "Lars multi-target".

- #12972 (closed): LassoLarsIC had both a `copy_X` init param and a `copy_X` fit param. Different problem (which value wins), fixed by making the fit param default to `None`.
- #14566, #17145 (closed, fixed by #21481): wrong AIC/BIC formula. Different problem.
- #27045 (open): alternative noise variance estimate. Unrelated.

- #17789 (closed): `lars_path` overwriting `X` with `copy_X=True` and `Gram='auto'`. Different (the input is modified despite `copy_X=True`; here the fit is wrong because of `copy_X=False`).

No issue about the column permutation.

## Describe the bug and its user-facing impact

When there is no Gram matrix, `lars_path(..., copy_X=False)` swaps the columns of `X` in place as features enter (and leave) the active set (`sklearn/linear_model/_least_angle.py:696` and `:869`). That's allowed by its docstring ("If `False`, `X` is overwritten"), but two callers reuse that `X` afterwards as if its columns were still in the original order:

- `LassoLarsIC.fit` computes the residuals of every model on the path with `np.dot(X, coef_path_)`, so the AIC/BIC curve is wrong and the selected `alpha_` and `coef_` are wrong. This happens with `copy_X=False` (init or fit param) and `precompute=False`, or with the default `precompute="auto"` when `n_samples <= n_features` (which requires passing `noise_variance`).
- `Lars.fit` / `LassoLars.fit` with a 2D `y` loop over targets and pass the same `X` to `lars_path`. With `copy_X=False` and `precompute=False`, the second target is fitted on the permuted `X` and its `coef_` row is permuted.

The results look plausible (finite coefficients, a valid alpha), so a user has no reason to suspect them.

## Steps/Code to Reproduce

```python
import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import Lars, LassoLarsIC

# LassoLarsIC: wide data with a known noise variance (precompute="auto" -> no Gram)
X, y = make_regression(n_samples=30, n_features=50, n_informative=5, noise=1,
                       random_state=0)
for copy_X in [True, False]:
    m = LassoLarsIC(noise_variance=1.0, copy_X=copy_X).fit(X.copy(), y)
    print(f"LassoLarsIC copy_X={copy_X}: alpha_={m.alpha_:.3g}, "
          f"n_nonzero={np.count_nonzero(m.coef_)}")

# Lars with 2 targets and no Gram: the 2nd target's coef_ is permuted
X, Y = make_regression(n_samples=40, n_features=6, n_targets=2, noise=1,
                       random_state=0)
for copy_X in [True, False]:
    m = Lars(precompute=False, copy_X=copy_X).fit(X.copy(), Y)
    print(f"Lars copy_X={copy_X}: coef_[1] =", np.round(m.coef_[1], 1))
```

## Expected Results

`copy_X` should only control whether the input array is overwritten, not the fitted model: the same `alpha_` and `coef_` for `copy_X=True` and `copy_X=False`.

## Actual Results

```
LassoLarsIC copy_X=True: alpha_=0.0572, n_nonzero=21
LassoLarsIC copy_X=False: alpha_=115, n_nonzero=0
Lars copy_X=True: coef_[1] = [79.1 13.2 45.2 30.5 38.  69.8]
Lars copy_X=False: coef_[1] = [69.8 13.2 45.2 38.  79.1 30.5]
```

With `copy_X=False`, LassoLarsIC selects the empty model, and the second row of Lars' `coef_` is a permutation of the right one.

## Root cause and possible fix

- `sklearn/linear_model/_least_angle.py:2261` (`LassoLarsIC.fit`): `residuals = y[:, np.newaxis] - np.dot(X, coef_path_)` uses the `X` that `lars_path` just permuted (called with `copy_X=copy_X` at line 2239). The OLS noise variance estimate is not affected (a column permutation does not change OLS residuals).
- `sklearn/linear_model/_least_angle.py:1098-1105` and `:1129-1136` (`Lars._fit`): the same `X` is passed with `copy_X=self.copy_X` to `lars_path` for each target. `_get_gram` (line 1070) forces a Gram matrix for multi-target `y` only when `precompute="auto"`, which hides the bug in the default setting.

Possible fix: in `LassoLarsIC.fit`, call `lars_path` with `copy_X=True` when no Gram matrix will be used (or compute the residuals before the path with a copy). In `Lars._fit`, pass `copy_X=True` for every target but the last one when `n_targets > 1`, like `orthogonal_mp` does (`sklearn/linear_model/_omp.py`: "subsequent targets will be affected").

## Versions

```
sklearn: 1.10.dev0, commit 289c4d5cfa (upstream/main; reproduced on a local build whose _least_angle.py is identical)
python: 3.13, numpy: 2.5.3, scipy: 1.18.1, platform: linux
```
