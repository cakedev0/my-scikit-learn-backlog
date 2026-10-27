# RadiusNeighborsClassifier with brute force assigns outliers to the wrong class

```json
{
    "estimators": ["RadiusNeighborsClassifier"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 2,
    "difficulty": 1,
    "decision": 0,
    "decision_tags": []
}
```

## Classification

- **Priority (2)**: silent wrong class for outliers, but it takes a stack of non-default settings on a niche estimator: `RadiusNeighborsClassifier` with `outlier_label` set, a non-euclidean metric (the default `p=2` skips this backend), the brute-force path, outliers at predict time, and integer labels that differ from their indices. (priority review, 2026-10-23: was 3)
- **Difficulty (1)**: pass the index of `outlier_label_` in `classes_` (or the fitted `outlier_label_`) instead of the raw `outlier_label` parameter: a couple of lines.
- **Decision (0)**: the fast path should give the same result as the tree path; no trade-off.

## Upstream search

- `outlier_label`: #9597 (merged, introduced `outlier_label="most_frequent"`), #26828 (merged, introduced the `RadiusNeighborsClassMode` backend), #1602, #6902, #9629 (closed, older and unrelated outlier problems).
- `RadiusNeighborsClassMode`: #33048 (merged in 1.9.0, "Fix *NeighborsClassifier with brute force for string labels"), which switched `unique_Y_labels` from `self.classes_` to `np.arange(n_classes)` without changing `outlier_label`. That's what turned the integer label case into a wrong class.
- `RadiusNeighborsClassifier outlier_label`: no match.

See also `neighbors-radius-classifier-brute-no-input-validation.md` (same fast path, missing input validation).

## Describe the bug and its user-facing impact

When `RadiusNeighborsClassifier` uses the brute-force `RadiusNeighborsClassMode` backend (`weights="uniform"`, single output, a metric other than euclidean, e.g. `manhattan`, `p=1`, `chebyshev`...), `predict_proba` compares the raw `outlier_label` parameter with the encoded class indices `0..n_classes-1` instead of with the class labels.

Consequences for samples without any neighbor in the radius:

- integer labels: `outlier_label=2` with `classes_=[1, 2, 3]` puts the probability mass on index 2, i.e. class `3`, and `predict` returns `3`. Silent wrong prediction. If the label doesn't match any index (e.g. `classes_=[10, 20]`), the probabilities are all 0 with a misleading "not in training classes" warning.
- `outlier_label="most_frequent"`: never matches, so `predict_proba` returns all zeros and warns "Outlier label most_frequent is not in training classes" although `outlier_label_` is a training class. `predict` is right in this case only because it falls back on `outlier_label_` for all-zero rows. This part already existed before 1.9.
- string labels: zero probabilities plus a misleading warning.

The tree-based algorithms give the expected result, so the output depends on `algorithm` (and on the number of features or the sparsity of `X` with `algorithm="auto"`).

## Steps/Code to Reproduce

```python
import numpy as np
from sklearn.neighbors import RadiusNeighborsClassifier

X = np.array([[0.0], [1.0], [2.0]])
y = np.array([1, 2, 3])
X_outlier = np.array([[100.0]])  # no training point within the radius

for algorithm in ["kd_tree", "brute"]:
    clf = RadiusNeighborsClassifier(
        radius=0.5, metric="manhattan", algorithm=algorithm, outlier_label=2
    ).fit(X, y)
    print(algorithm, clf.predict(X_outlier), clf.predict_proba(X_outlier))

for algorithm in ["kd_tree", "brute"]:
    clf = RadiusNeighborsClassifier(
        radius=0.5, metric="manhattan", algorithm=algorithm,
        outlier_label="most_frequent",
    ).fit(X, [1, 3, 3])
    print(algorithm, clf.predict(X_outlier), clf.predict_proba(X_outlier))
```

## Expected Results

Same output for all algorithms, as with `kd_tree`: outlier predicted as `2` with probabilities `[[0, 1, 0]]`, and for `"most_frequent"` predicted as `3` with probabilities `[[0, 1]]`, without warning. The docstring says outliers are given `outlier_label`, and that probabilities are 0 only when the label is not a training class.

## Actual Results

```
/home/arthur/scikit-learn/sklearn/metrics/_pairwise_distances_reduction/_dispatcher.py:731: UserWarning: Outlier label most_frequent is not in training classes. All class probabilities of outliers will be assigned with 0.
  return RadiusNeighborsClassMode64.compute(
kd_tree [2] [[0. 1. 0.]]
brute [3] [[0. 0. 1.]]
kd_tree [3] [[0. 1.]]
brute [3] [[0. 0.]]
```

## Root cause and possible fix

`sklearn/neighbors/_classification.py:800-801` passes `unique_Y_labels=np.arange(len(self.classes_))` together with `outlier_label=self.outlier_label` (the raw parameter, possibly `"most_frequent"`). `sklearn/metrics/_pairwise_distances_reduction/_radius_neighbors_classmode.pyx.tp:104-107` then looks for `outlier_label` among `unique_Y_labels`, which are class indices, not labels.

Fix: pass the fitted `self.outlier_label_[0]` converted to its index in `self.classes_` (or a sentinel such as `-1` when it's not a training class, keeping the warning), and add a test comparing brute and tree algorithms with a non-euclidean metric and `outlier_label` set to an integer label that differs from its index, and to `"most_frequent"`.

## Versions

```
sklearn: 1.10.dev0, commit f570497251 (no diff in sklearn/neighbors vs upstream/main 289c4d5cfa)
python: 3.13, numpy: 2.5.3, scipy: 1.18.1, platform: linux
```
