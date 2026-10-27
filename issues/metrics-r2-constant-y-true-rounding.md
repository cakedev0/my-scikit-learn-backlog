# `r2_score` returns ~-1e32 instead of 0.0 for a constant `y_true` like `0.1` (`force_finite` misses rounding)

```json
{
    "estimators": ["r2_score", "explained_variance_score", "d2_tweedie_score"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 2,
    "difficulty": 1,
    "decision": 1,
    "decision_tags": ["output-change"]
}
```

## Classification

- **Priority (2)**: default settings, very common metric (default `score` of every regressor), and the documented `force_finite=True` behavior silently fails depending on the value of the constant. Constant targets do happen in CV folds or groups. The value is absurd enough to be noticed, but it wrecks averages such as `cross_val_score(...).mean()`.
- **Difficulty (1)**: detect "constant `y_true`" exactly (e.g. `xp.all(y_true == y_true[0], axis=0)`) instead of relying on `denominator != 0`.
- **Decision (1)**: changes the returned value (huge negative -> 0.0 or 1.0) for inputs that are currently affected (`output-change`); this is what the docstring promises.

## Upstream search

Queries: "r2_score constant", "r2_score force_finite", "force_finite constant", "r2_score constant y_true floating point".

- #16857 (closed) / #17266 (merged): introduced `force_finite` for constant `y_true`; neither discusses floating point rounding of the mean.
- #34834, #34197, #33642 (closed): similar "constant but not exactly zero variance" problem in `r_regression`/`f_regression` (feature_selection), not in the regression metrics.
- #11435 (open) is about single-sample inputs, a different degenerate case.

## Describe the bug and its user-facing impact

The docstring says that with `force_finite=True` (default), a constant `y_true` gives 1.0 for perfect predictions and 0.0 otherwise. The implementation checks `denominator != 0`, where the denominator is `sum(w * (y_true - mean(y_true))**2)`. For many constants (`0.1`, `0.3`, ...), `mean(y_true)` is not exactly the constant after floating point rounding, so the denominator is ~1e-34 instead of 0, the special case is skipped, and the score is ~-1e32. With `y_true = 1.0` it works, so the outcome depends on the constant and on `n_samples`.

`explained_variance_score` has the same issue (its denominator is the weighted variance of `y_true`), and so does `d2_tweedie_score` (which has no `force_finite` at all).

## Steps/Code to Reproduce

```python
import numpy as np
from sklearn.metrics import explained_variance_score, r2_score

y_pred = np.array([0.2, 0.3, 0.4])
for c in [1.0, 0.1]:
    y_true = np.full(3, c)  # constant target
    print(f"y_true={c}: r2_score={r2_score(y_true, y_pred)}, "
          f"explained_variance_score={explained_variance_score(y_true, y_pred)}")
print(np.mean(np.full(3, 0.1)) == 0.1)
```

## Expected Results

`r2_score=0.0, explained_variance_score=0.0` for both constants, as documented for `force_finite=True`.

## Actual Results

```
y_true=1.0: r2_score=0.0, explained_variance_score=0.0
y_true=0.1: r2_score=-2.423071867316253e+32, explained_variance_score=-3.4615312390232204e+31
False
```

## Root cause and possible fix

`sklearn/metrics/_regression.py:972` (`_assemble_fraction_of_explained_deviance`): `nonzero_denominator = denominator != 0`, with the denominator computed at lines 1318-1321 (`r2_score`) and 1146-1148 (`explained_variance_score`). Rounding in `_average(y_true)` leaves a tiny positive residual. Passing an explicit "y_true is constant per output" mask (exact equality to the first row) to `_assemble_fraction_of_explained_deviance`, or computing the denominator from `y_true - y_true[0]` shifted data, would make the special case exact. A relative tolerance (`denominator <= eps * sum(y_true**2)`) is another option but harder to justify.

## Versions

```
sklearn: 1.10.dev0, commit f570497251 (no change vs upstream/main 289c4d5cfa in the files involved)
python: 3.13, numpy: 2.5, scipy: 1.18, platform: linux
```
