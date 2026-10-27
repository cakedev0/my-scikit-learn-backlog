# LocalOutlierFactor(novelty=True) drops feature names and rejects NaN with nan_euclidean in predict/score_samples

```json
{
    "estimators": ["LocalOutlierFactor"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 2,
    "difficulty": 1,
    "decision": 0,
    "decision_tags": []
}
```

## Classification

- **Priority (2)**: with a DataFrame (the common case), every `predict`/`decision_function`/`score_samples` call warns spuriously, and a DataFrame with reordered columns is silently accepted, giving wrong scores. With `metric="nan_euclidean"`, novelty detection doesn't work at all. Only `novelty=True` is affected.
- **Difficulty (1)**: drop the `check_array` call in `score_samples` and let `kneighbors` validate `X` (it already calls `validate_data(reset=False)` with the right `ensure_all_finite`).
- **Decision (0)**: aligns LOF with all other estimators; the only visible changes are the removed warning and the new error on reordered columns, which is the standard behavior.

## Upstream search

- `LocalOutlierFactor feature names`: #25844 (closed, same warning for `IsolationForest`, different code path), nothing for LOF.
- `"valid feature names" LocalOutlierFactor`: only #25844.
- `LocalOutlierFactor novelty nan_euclidean`: no match. #25330 (merged) added `nan_euclidean` support to neighbors estimators and tests LOF only through `fit_predict`, which doesn't go through `score_samples`.

## Describe the bug and its user-facing impact

`LocalOutlierFactor.score_samples` (used by `predict` and `decision_function` when `novelty=True`) converts `X` with `check_array` before calling `kneighbors`. This has three consequences:

1. When fitted on a DataFrame, every call warns "X does not have valid feature names, but LocalOutlierFactor was fitted with feature names", although the user passed the right DataFrame.
2. A DataFrame with the columns in a different order is not rejected: the feature name check runs on the already converted ndarray. Predictions are silently wrong (0.84 vs 0.2 mean prediction below), whereas every other estimator raises "The feature names should match those that were passed during fit."
3. `check_array` uses the default `ensure_all_finite=True`, so with `metric="nan_euclidean"`, `fit` and `kneighbors` accept NaN but `predict`, `decision_function` and `score_samples` raise "Input contains NaN".

## Steps/Code to Reproduce

```python
import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor

rng = np.random.RandomState(0)
X = pd.DataFrame({"a": rng.randn(50), "b": 100 * rng.randn(50)})
lof = LocalOutlierFactor(novelty=True).fit(X)

print(lof.predict(X).mean())  # warns about missing feature names
print(lof.predict(X[["b", "a"]]).mean())  # swapped columns: no error

X_nan = np.array([[0, 1], [1, np.nan], [2, 3], [3, 5]])
lof = LocalOutlierFactor(n_neighbors=2, metric="nan_euclidean", novelty=True)
lof.fit(X_nan)  # fit and kneighbors accept NaN
lof.predict(X_nan)  # raises
```

## Expected Results

No warning on the first call, a `ValueError` about feature names order on the second one (as for `KNeighborsClassifier.predict`), and predictions for `X_nan` since the metric supports NaN.

## Actual Results

```
/home/arthur/scikit-learn/sklearn/utils/validation.py:2822: UserWarning: X does not have valid feature names, but LocalOutlierFactor was fitted with feature names
  warnings.warn(
/home/arthur/scikit-learn/sklearn/utils/validation.py:2822: UserWarning: X does not have valid feature names, but LocalOutlierFactor was fitted with feature names
  warnings.warn(
0.84
0.2
Traceback (most recent call last):
  ...
  File "/home/arthur/scikit-learn/sklearn/neighbors/_lof.py", line 467, in score_samples
  File "/home/arthur/scikit-learn/sklearn/utils/validation.py", line 1084, in check_array
  File "/home/arthur/scikit-learn/sklearn/utils/validation.py", line 134, in _assert_all_finite
  File "/home/arthur/scikit-learn/sklearn/utils/validation.py", line 183, in _assert_all_finite_element_wise
ValueError: Input contains NaN.
```

## Root cause and possible fix

`sklearn/neighbors/_lof.py:467`: `X = check_array(X, accept_sparse="csr")`, unchanged since LOF was added (#5279). The result is passed to `self.kneighbors(X, ...)`, whose `validate_data(..., reset=False)` then sees an ndarray.

Fix: remove the `check_array` call and pass the user's `X` to `kneighbors`, which already validates it with feature names and the metric-dependent `ensure_all_finite`. The only use of the converted `X` is the float32 check that casts `distances_X`; it can be done on the dtype of the input (e.g. `getattr(X, "dtype", None)`, or a small helper shared with `kneighbors`). The common estimator checks don't run with `novelty=True`, which is why the feature names check never caught it; adding a `novelty=True` instance to the common tests would.

## Versions

```
sklearn: 1.10.dev0, commit f570497251 (no diff in sklearn/neighbors vs upstream/main 289c4d5cfa)
python: 3.13, numpy: 2.5.3, scipy: 1.18.1, platform: linux
```
