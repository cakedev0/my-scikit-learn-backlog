# RidgeCV with `scoring` fails on zero sample weights (leave-one-out predictions are NaN)

```json
{
    "estimators": ["RidgeCV"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 2,
    "difficulty": 2,
    "decision": 0,
    "decision_tags": []
}
```

## Classification

- **Priority (2)**: a crash on valid input (zero weights are accepted everywhere else, including `RidgeCV` with `scoring=None`) for a common estimator; there is a workaround (drop the zero-weight rows), and the error ("Input contains NaN") at least is loud.
- **Difficulty (2)**: the division has to be skipped for zero weights and the prediction for those rows computed another way (the full-data model prediction, which equals the leave-one-out one since such a row has no influence on the fit).
- **Decision (0)**: no trade-off, the current result is a crash.

## Upstream search

Queries: "RidgeCV sample_weight zero", "RidgeCV scoring sample_weight NaN", "_RidgeGCV sample_weight scoring". No match for this symptom.

## Describe the bug and its user-facing impact

With the default efficient leave-one-out CV (`cv=None`) and a `scoring` other than `None`, `RidgeCV.fit(X, y, sample_weight=sw)` raises `ValueError: Input contains NaN` as soon as one entry of `sw` is 0, with or without `store_cv_results=True`. `RidgeClassifierCV(scoring="accuracy")` goes through the same division and gets NaN decision values for those rows, but does not fail since NaN rows end up predicted as the first class and have zero weight in the score.

Zero weights are a common way to mask samples (e.g. in a pipeline that computes weights), and they work with `scoring=None` and with every other Ridge solver.

## Steps/Code to Reproduce

```python
import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import RidgeCV

X, y = make_regression(n_samples=50, n_features=5, noise=10, random_state=0)
sample_weight = np.ones(50)
sample_weight[:5] = 0  # e.g. masking out a few samples

print(RidgeCV().fit(X, y, sample_weight=sample_weight).alpha_)  # works
RidgeCV(scoring="r2").fit(X, y, sample_weight=sample_weight)  # fails
```

## Expected Results

Both fits succeed. A sample with zero weight has no influence on the fit, so its leave-one-out prediction is just the prediction of the model fitted on all data; in any case the weighted scorer gives it no weight.

## Actual Results

```
/home/arthur/scikit-learn/sklearn/linear_model/_ridge.py:2255: RuntimeWarning: invalid value encountered in divide
  predictions /= sqrt_sw
0.1
Traceback (most recent call last):
  File ".../repro_ridgecv_zero.py", line 10, in <module>
    RidgeCV(scoring="r2").fit(X, y, sample_weight=sample_weight)  # fails
  File "/home/arthur/scikit-learn/sklearn/linear_model/_ridge.py", line 2783, in fit
    super().fit(X, y, sample_weight=sample_weight, **params)
  File "/home/arthur/scikit-learn/sklearn/linear_model/_ridge.py", line 2507, in fit
    estimator.fit(
  File "/home/arthur/scikit-learn/sklearn/linear_model/_ridge.py", line 2262, in fit
    alpha_score = self._score(
  File "/home/arthur/scikit-learn/sklearn/linear_model/_ridge.py", line 2362, in _score
    _score = scorer(
  ...
  File "/home/arthur/scikit-learn/sklearn/utils/validation.py", line 183, in _assert_all_finite_element_wise
    raise ValueError(msg_err)
ValueError: Input contains NaN.
```

## Root cause and possible fix

`_RidgeGCV.fit` works on data rescaled by `sqrt(sample_weight)`. To give the scorer predictions on the original scale, it divides the rescaled leave-one-out predictions by `sqrt_sw` (`sklearn/linear_model/_ridge.py:2246-2252` on upstream/main):

```python
predictions = y - looe
if sample_weight is not None:
    ...
        predictions /= sqrt_sw
```

For a zero weight, both the rescaled target and `looe` are 0, so this is `0 / 0 = NaN`.

Possible fix: divide only where `sqrt_sw > 0`, and for the zero-weight rows use the prediction of the model fitted on all data for that alpha (`X_original[zero] @ coef + intercept`), which is the exact leave-one-out prediction for those rows. Filling them with any finite value would be enough for the score (the scorer receives the same zero weights), but not for `cv_results_` when `store_cv_results=True`.

## Versions

```
sklearn: 1.10.dev0, commit 289c4d5cfa (upstream/main; reproduced on a local build whose _ridge.py only differs in docstrings)
python: 3.13, numpy: 2.5.3, scipy: 1.18.1, platform: linux
```
