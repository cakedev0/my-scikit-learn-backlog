# `d2_tweedie_score` raises `ZeroDivisionError` when `y_true` is constant

```json
{
    "estimators": ["d2_tweedie_score"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 1,
    "difficulty": 1,
    "decision": 1,
    "decision_tags": ["unclear-spec"]
}
```

## Classification

- **Priority (1)**: loud crash on a rarely used metric, only for a constant `y_true` (e.g. a CV fold of all-zero counts); the pre-1.6 result (`-inf`/`nan`) was not useful either. (priority review, 2026-10-23: was 2)
- **Difficulty (1)**: guard the division at the end of `d2_tweedie_score`.
- **Decision (1)**: what to return must be decided: the pre-1.6 `nan`/`-inf`, or the `force_finite` convention of `r2_score` and `d2_pinball_score` (1.0 for perfect predictions, 0.0 otherwise). The docstring says `power=0` "corresponds to r2_score", which argues for the latter (`unclear-spec`).

## Upstream search

Queries: "d2_tweedie_score", "d2_tweedie_score constant", "d2 score zero division".

No issue. #34798 (closed, unmerged, "FIX Handle zero-weight samples in regression scores") mentions a `ZeroDivisionError` in `d2_tweedie_score` but for a different trigger (one non-zero weight among zero-weight samples), and it was closed. #11435 (open) is about single-sample inputs. #28106 (merged, 1.6.0) introduced the Python-float return of `mean_tweedie_deviance`.

## Describe the bug and its user-facing impact

When `y_true` is constant, the denominator of D² (deviance of the constant `mean(y_true)` prediction) is exactly 0. `mean_tweedie_deviance` and `_mean_tweedie_deviance` return Python floats since #28106 (1.6.0), so `1 - numerator / denominator` raises `ZeroDivisionError` instead of producing `-inf` or `nan` with a RuntimeWarning as in 1.5.

`r2_score` (the `power=0` case, according to the docstring) returns 0.0 for the same input, and `d2_pinball_score`/`d2_absolute_error_score` also return finite values through `force_finite=True`.

## Steps/Code to Reproduce

```python
from sklearn.metrics import d2_tweedie_score, r2_score

y_true = [1.0, 1.0, 1.0]
y_pred = [1.0, 1.0, 2.0]
print("r2_score:", r2_score(y_true, y_pred))
print("d2_tweedie_score(power=0):", d2_tweedie_score(y_true, y_pred, power=0))
```

## Expected Results

No exception. Either `0.0` like `r2_score` (and `1.0` when `y_pred == y_true`), or at least `-inf`/`nan` as before 1.6.

## Actual Results

```
r2_score: 0.0
Traceback (most recent call last):
  File "r_d2tw.py", line 6, in <module>
    print("d2_tweedie_score(power=0):", d2_tweedie_score(y_true, y_pred, power=0))
                                        ~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^
  File "sklearn/utils/_param_validation.py", line 218, in wrapper
    return func(*args, **kwargs)
  File "sklearn/metrics/_regression.py", line 1705, in d2_tweedie_score
    return 1 - numerator / denominator
               ~~~~~~~~~~^~~~~~~~~~~~~
ZeroDivisionError: float division by zero
```

## Root cause and possible fix

`sklearn/metrics/_regression.py:1705`: `numerator` and `denominator` are Python floats (`_mean_tweedie_deviance` returns `float(_average(...))` at line 1408), so the division raises. The simplest fix is to reuse `_assemble_fraction_of_explained_deviance(..., force_finite=True)` like `d2_pinball_score` does, which also aligns `power=0` with `r2_score`. Note that a constant `y_true` whose mean is not exactly representable (e.g. `0.1`) gives a tiny non-zero denominator and a huge negative score instead; see `metrics-r2-constant-y-true-rounding.md` for that part.

## Versions

```
sklearn: 1.10.dev0, commit f570497251 (no change vs upstream/main 289c4d5cfa in the files involved)
python: 3.13, numpy: 2.5, scipy: 1.18, platform: linux
```
