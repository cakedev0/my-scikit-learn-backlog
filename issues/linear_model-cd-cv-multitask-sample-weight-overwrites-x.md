# LassoCV, ElasticNetCV and MultiTaskElasticNet overwrite X with `fit_intercept=False` and `sample_weight`, despite `copy_X=True`

```json
{
    "estimators": ["LassoCV", "ElasticNetCV", "MultiTaskElasticNet", "MultiTaskLasso", "MultiTaskElasticNetCV", "MultiTaskLassoCV"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 2,
    "difficulty": 1,
    "decision": 0,
    "decision_tags": []
}
```

## Classification

- **Priority (2)**: silent corruption of the user's data (a pandas DataFrame is modified in place), which then gives wrong results in any later use of that data, e.g. refitting gives a different model. It needs `fit_intercept=False` plus `sample_weight` plus an F-contiguous float array or a DataFrame, all with the default `copy_X=True`.
- **Difficulty (1)**: copy before rescaling when `copy_X=True` and `sample_weight` is given, as `ElasticNet.fit` already does.
- **Decision (0)**: one extra copy in a case where the user asked for it (`copy_X=True`).

## Upstream search

Queries (issues and PRs): "LassoCV modifies X", "LassoCV copy_X", "LassoCV sample_weight copy_X", "ElasticNetCV modifies X", "ElasticNetCV fit_intercept=False sample_weight", "MultiTaskElasticNet copy_X", "MultiTaskElasticNet sample_weight copy_X", "copy_X sample_weight overwritten".

- #27172 (closed): LinearRegression with `copy_X=False` and `sample_weight`; expected behavior since the user asked for `copy_X=False`. Here `copy_X=True`.

No other match.

`sample_weight` support for `MultiTaskElasticNet` was added in #33440 (1.9). The CV classes have a `# TODO: Sample_weights?` next to the line that decides not to copy.

## Describe the bug and its user-facing impact

With `fit_intercept=False` and `sample_weight`, the coordinate descent estimators below rescale `X` in place by `sqrt(sample_weight)` (normalized to sum to `n_samples`), even though `copy_X=True` (the default):

- `LassoCV`, `ElasticNetCV`, `MultiTaskLassoCV`, `MultiTaskElasticNetCV`: the final refit on the whole data;
- `MultiTaskLasso`, `MultiTaskElasticNet`.

The input is overwritten when no copy happens during validation, i.e. for a float64/float32 F-contiguous array, or for a pandas DataFrame (whose single block converts to an F-contiguous view). C-contiguous arrays are copied to F order on the way and are not affected. `ElasticNet` and `Lasso` are not affected.

After `fit`, the user's `X` silently holds `sqrt(w) * X`: any later prediction, plot or refit on it is wrong. In the reproducer below, fitting the same estimator twice on the same array gives two different models.

## Steps/Code to Reproduce

```python
import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.linear_model import LassoCV, MultiTaskLasso

X, Y = make_regression(n_samples=50, n_features=5, n_targets=2, random_state=0)
sample_weight = np.random.RandomState(0).uniform(0.5, 2, size=50)

df = pd.DataFrame(X.copy())  # same with np.asfortranarray(X.copy())
LassoCV(fit_intercept=False).fit(df, Y[:, 0], sample_weight=sample_weight)
print("LassoCV modified X:", not np.array_equal(df.to_numpy(), X))

df = pd.DataFrame(X.copy())
MultiTaskLasso(fit_intercept=False).fit(df, Y, sample_weight=sample_weight)
print("MultiTaskLasso modified X:", not np.array_equal(df.to_numpy(), X))

# consequence: fitting twice on the same data gives a different model
Xf = np.asfortranarray(X.copy())
coef_1 = LassoCV(fit_intercept=False).fit(Xf, Y[:, 0], sample_weight=sample_weight).coef_
coef_2 = LassoCV(fit_intercept=False).fit(Xf, Y[:, 0], sample_weight=sample_weight).coef_
print(np.round(coef_1, 2))
print(np.round(coef_2, 2))
```

## Expected Results

With `copy_X=True`, "X will be copied; else, it may be overwritten" (docstring): `X` is left untouched, and two fits on the same data give the same model. This is what `ElasticNet`/`Lasso` do in the same setting.

## Actual Results

```
LassoCV modified X: True
MultiTaskLasso modified X: True
[ 5.47 17.89 77.59 25.85 58.52]
[ 7.9  18.82 73.24 22.73 57.92]
```

## Root cause and possible fix

`_preprocess_data` always rescales dense `X` in place when `sample_weight` is given, relying on the caller to have made a copy (`sklearn/linear_model/_base.py:216-217`: "inplace=True here regardless of copy"). Two callers only copy when `fit_intercept=True`:

- `LinearModelCV.fit` sets `copy_X = self.copy_X and self.fit_intercept  # TODO: Sample_weights?` (`sklearn/linear_model/_coordinate_descent.py:1798`) and passes it to the final model (`model.copy_X = copy_X`, line 1999). That model's `ElasticNet.fit` then sees `copy_X=False` and rescales the user's array. The CV folds are not affected since `X[train]` is a copy.
- `MultiTaskElasticNet.fit` validates with `copy=self.copy_X and self.fit_intercept` (line 2761) and then calls `_pre_fit(..., copy=False,  # TODO: improve)` (line 2804).

`ElasticNet.fit` handles this correctly with `should_copy = self.copy_X and not X_copied` passed to `_pre_fit`. The fix is to do the same in `MultiTaskElasticNet.fit`, and to take `sample_weight is not None` into account when computing `copy_X` in `LinearModelCV.fit`.

## Versions

```
sklearn: 1.10.dev0, commit 289c4d5cfa (upstream/main; reproduced on a local build whose _coordinate_descent.py only differs in docstrings)
python: 3.13, numpy: 2.5.3, scipy: 1.18.1, pandas: 3.0.6, platform: linux
```
