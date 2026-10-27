# `QuadraticDiscriminantAnalysis` raises "not full rank" on well-conditioned data with small feature scales

```json
{
    "estimators": ["QuadraticDiscriminantAnalysis"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 2,
    "difficulty": 1,
    "decision": 1,
    "decision_tags": ["api-change"]
}
```

## Classification

- **Priority (2)**: regression in 1.8.0: `fit` now raises `LinAlgError` on full-rank data as soon as a class variance is below the absolute `tol=1e-4` (features in small units), where it used to warn. The error is loud and `StandardScaler` or `tol=0` work around it, but the error message points users to `reg_param`, which then badly distorts the model on such data.
- **Difficulty (1)**: compare the eigenvalues with a threshold relative to the largest one (or to the data scale) instead of an absolute one, and update the `tol` docstring.
- **Decision (1)**: `tol` changes meaning from absolute to relative (`api-change`), which may need a changelog note. Its current docstring already says it only controls a warning and does not affect predictions, so few users rely on its exact value.

## Upstream search

GraphQL searches "QuadraticDiscriminantAnalysis reg_param eigen", "QuadraticDiscriminantAnalysis LinAlgError not full rank", "QuadraticDiscriminantAnalysis tol", "QDA covariance not full rank error 1.8".

- #32108 (merged, in 1.8.0) turned the rank-deficiency `LinAlgWarning` into a `LinAlgError` on purpose (rank-deficient classes used to fail later at predict time). The discussion is about truly rank-deficient classes; the absolute scale of `tol` is not discussed (searched the PR comments and review comments for "tol", "relative", "absolute", "scale").
- #14997 (open, "QDA Classifier: divide by zero if data not of full rank"): truly rank-deficient data, different case.
- #32560 (closed), #19731 (merged): warning / message wording, not the threshold.

## Describe the bug and its user-facing impact

QDA estimates the rank of each class covariance with `np.sum(scaling_class > self.tol)`, where `scaling_class` are the covariance eigenvalues (variances along the principal axes) and `tol` defaults to `1e-4`. This is an absolute threshold on a quantity in squared feature units. Any class whose smallest principal variance is below `1e-4` (features with standard deviation below 0.01, e.g. measurements in meters of millimeter-scale objects, rates, probabilities, ...) is reported as "not full rank" and, since 1.8.0, `fit` raises.

Two more problems make it worse:

- The `tol` docstring says: "This parameter does not affect the predictions. It controls when a warning is raised if the covariance matrix is not full rank." It now decides whether `fit` succeeds.
- The error message says "Increase the value of `reg_param`". For small-scale data, `reg_param` adds `reg_param` (in absolute units) to each variance, so e.g. `reg_param=0.01` swamps variances of order `1e-6` and the class covariances are lost. On a variant of the example below where the two classes have different covariances (`X[y == 1, 0] *= 4`), the training accuracy goes from 0.87 (`tol=0`) to 0.745 with `reg_param=0.01`.

The same data rescaled by `1e3` fits fine, so the result of `fit` depends on the units of the features, which is not expected from QDA (whose predictions are invariant to feature scaling).

## Steps/Code to Reproduce

```python
import numpy as np
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

rng = np.random.RandomState(0)
X = rng.normal(size=(200, 3))
y = np.repeat([0, 1], 100)
X[y == 1] += 1

print(QuadraticDiscriminantAnalysis().fit(X, y).score(X, y))
# Same data in other units (e.g. meters instead of millimeters): class
# variances ~1e-6, below the absolute `tol=1e-4`.
QuadraticDiscriminantAnalysis().fit(X * 1e-3, y)
```

## Expected Results

Both fits succeed with the same accuracy: the class covariances are full rank and well conditioned in both cases.

## Actual Results

```
0.765
Traceback (most recent call last):
  File "qda_tol.py", line 12, in <module>
    QuadraticDiscriminantAnalysis().fit(X * 1e-3, y)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^
  File "/home/arthur/scikit-learn/sklearn/base.py", line 1403, in wrapper
    return fit_method(estimator, *args, **kwargs)
  File "/home/arthur/scikit-learn/sklearn/discriminant_analysis.py", line 1138, in fit
    raise linalg.LinAlgError(
    ...<3 lines>...
    )
numpy.linalg.LinAlgError: The covariance matrix of class 0 is not full rank. Increase the value of `reg_param` to reduce the collinearity.
```

The same happens with `solver="eigen"` (message then suggests `shrinkage`).

## Root cause and possible fix

`sklearn/discriminant_analysis.py:1124`: `rank = np.sum(scaling_class > self.tol)`, followed by the `raise linalg.LinAlgError` at lines 1125-1145 (introduced by #32108; it was a `LinAlgWarning` before 1.8.0).

Possible fix: use a relative threshold, e.g. `scaling_class > self.tol * scaling_class.max()` (or `np.finfo` based, like `numpy.linalg.matrix_rank`), and update the `tol` docstring to say it controls a `LinAlgError`.

## Versions

```
sklearn: 1.10.dev0, commit 289c4d5cfa (upstream/main; local HEAD f570497251 has no diff in sklearn/discriminant_analysis.py)
python: 3.13, numpy: 2.5.3, scipy: 1.18.1, platform: linux
```
