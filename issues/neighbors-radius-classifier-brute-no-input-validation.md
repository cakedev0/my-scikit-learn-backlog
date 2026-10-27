# RadiusNeighborsClassifier.predict_proba skips input validation in its brute-force fast path

```json
{
    "estimators": ["RadiusNeighborsClassifier"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 1,
    "difficulty": 1,
    "decision": 0,
    "decision_tags": []
}
```

## Classification

- **Priority (1)**: missing validation for malformed queries (wrong number of features, NaN), i.e. user errors, in a niche path: `RadiusNeighborsClassifier` with a non-euclidean metric on the brute-force backend. Correct inputs give correct results. (priority review, 2026-10-23: was 2)
- **Difficulty (1)**: call `validate_data(self, X, reset=False, ...)` (or `_check_precomputed`) before the fast path, as `KNeighborsClassifier.predict_proba` does.
- **Decision (0)**: no trade-off, it restores the validation the other paths already do.

## Upstream search

`RadiusNeighborsClassifier n_features`: no report of this bug. Related: #26828 (merged in 1.4, "PERF Implement `PairwiseDistancesReduction` backend for `RadiusNeighbors.predict_proba`", which introduced the fast path without validation), #33048 (merged, fixed string labels in the same fast path). Local: `neighbors-radius-classifier-brute-outlier-label-index.md` is another bug in the same fast path (outlier label lookup), with a different root cause.

## Describe the bug and its user-facing impact

`RadiusNeighborsClassifier.predict_proba` (and `predict`, which calls it) passes the raw `X` to `RadiusNeighborsClassMode.compute` when `algorithm` resolves to `"brute"` (e.g. `algorithm="auto"` with more than 15 features or sparse input) and the metric is supported by the Cython backend (anything but euclidean, e.g. `manhattan`, `p=1`, `chebyshev`). `X` is never validated there:

- a query with a different number of features than the training data is not rejected: distances are computed on `X.shape[1]` features, so with more features they read past each training row (and past the end of the training buffer for the last row), and with fewer features they silently use only the first columns;
- NaN or inf in `X` is not rejected; a NaN query ends up as an outlier (no neighbor) and raises a misleading "No neighbors found" error, or silently gets `outlier_label`.

The tree algorithms and the KNN classifier raise the usual validation errors.

## Steps/Code to Reproduce

```python
import numpy as np
from sklearn.neighbors import RadiusNeighborsClassifier

rng = np.random.RandomState(0)
X = rng.rand(20, 3)
y = rng.randint(2, size=20)
X_bad = [rng.rand(2, 5), np.array([[np.nan, 0.0, 0.0]])]  # 5 features; NaN

for algorithm in ["kd_tree", "brute"]:
    clf = RadiusNeighborsClassifier(metric="manhattan", algorithm=algorithm)
    clf.fit(X, y)
    for X_test in X_bad:
        try:
            print(algorithm, clf.predict_proba(X_test))
        except ValueError as e:
            print(algorithm, "ValueError:", str(e).splitlines()[0])
```

## Expected Results

The same validation errors for `brute` as for `kd_tree`.

## Actual Results

```
kd_tree ValueError: X has 5 features, but RadiusNeighborsClassifier is expecting 3 features as input.
kd_tree ValueError: Input X contains NaN.
brute [[0.  1. ]
 [0.5 0.5]]
brute ValueError: No neighbors found for test samples array([0]), you can try using larger radius, giving a label for outliers, or considering removing them from your dataset.
```

## Root cause and possible fix

`sklearn/neighbors/_classification.py:788-811`: the fast path calls `RadiusNeighborsClassMode.compute(X=X, ...)` with the user's `X`; validation only happens later in `radius_neighbors`, on the fallback path. `KNeighborsClassifier.predict_proba` (`_classification.py:343-348`) validates `X` with `validate_data(self, X, accept_sparse="csr", reset=False, order="C")` (or `_check_precomputed`) before its own fast path; do the same here, and handle `X=None` (query on the training set) before validating. The common tests don't catch it since they use the default euclidean metric, which this backend excludes.

## Versions

```
sklearn: 1.10.dev0, commit f570497251 (no diff in sklearn/neighbors vs upstream/main 289c4d5cfa)
python: 3.13, numpy: 2.5.3, scipy: 1.18.1, platform: linux
```
