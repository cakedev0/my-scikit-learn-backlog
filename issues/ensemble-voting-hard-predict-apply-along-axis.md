# `VotingClassifier(voting="hard").predict` spends most of its time in a per-sample `np.apply_along_axis` vote

```json
{
    "estimators": ["VotingClassifier"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 1,
    "difficulty": 1,
    "decision": 0,
    "decision_tags": []
}
```

## Classification

- **Priority (1)**: performance only, with correct results: hard voting costs ~2.5 µs per sample on top of the base predictions, which stays well under a second for typical test sets and only becomes noticeable on hundreds of thousands of rows. (priority review, 2026-10-23: was 2)
- **Difficulty (1)**: replace `apply_along_axis` with a loop over the (few) estimators that accumulates weights into an `(n_samples, n_classes)` array, then `argmax`. Same tie-breaking.
- **Decision (0)**: same predictions; the weights are added in the same order (estimator by estimator), so even the floating-point sums are unchanged.

## Upstream search

- "VotingClassifier hard voting slow", "VotingClassifier apply_along_axis" (GitHub issue search).
- #5805 (merged) "parallelized VotingClassifier": parallel `fit`, not the vote.
- #15056, #11031 (closed): unrelated errors.
- No issue or PR about the speed of the hard vote.

## Describe the bug and its user-facing impact

With `voting="hard"`, `VotingClassifier.predict` gathers the base estimators' predictions into an `(n_samples, n_estimators)` array, then calls a Python lambda on each row through `np.apply_along_axis` (`np.bincount` + `np.argmax` per sample). That costs ~2.5 µs per sample, which is several times more than the base predictions for fast estimators (linear models, naive Bayes, small trees).

With three base estimators on 400,000 samples, `predict` takes ~1.3 s, of which ~0.3 s are the base estimators' `predict` and ~1 s is the vote, while a vectorized vote gives the same predictions in ~0.04 s. `score` calls `predict`, so cross-validation and grid searches over a `VotingClassifier` pay this on every split.

## Steps/Code to Reproduce

```python
from time import perf_counter

import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.naive_bayes import GaussianNB

X, y = make_classification(
    n_samples=400_000, n_features=20, n_informative=5, n_classes=3, random_state=0
)
estimators = [("lr", LogisticRegression()), ("ridge", RidgeClassifier()),
              ("nb", GaussianNB())]
clf = VotingClassifier(estimators, voting="hard").fit(X[:10_000], y[:10_000])
for n_samples in [100_000, 200_000, 400_000]:
    X_test = X[:n_samples]
    t0 = perf_counter()
    y_pred = clf.predict(X_test)
    t1 = perf_counter()
    # what VotingClassifier.predict does before the vote
    votes = np.column_stack([est.predict(X_test) for est in clf.estimators_])
    t2 = perf_counter()
    # vectorized vote, same tie-breaking (smallest class index wins)
    counts = np.zeros((n_samples, len(clf.classes_)))
    for column in votes.T:
        counts[np.arange(n_samples), column] += 1
    y_vectorized = clf.classes_[counts.argmax(axis=1)]
    t3 = perf_counter()
    print(f"n_samples={n_samples}: VotingClassifier.predict {t1 - t0:.2f}s, "
          f"base estimators' predict {t2 - t1:.2f}s, vectorized vote "
          f"{t3 - t2:.3f}s, same predictions: {np.array_equal(y_pred, y_vectorized)}")
```

## Expected Results

`predict` costs about the sum of the base estimators' `predict`, plus a vote that is negligible in comparison.

## Actual Results

Timings from a laptop loaded by other jobs:

```
n_samples=100000: VotingClassifier.predict 0.33s, base estimators' predict 0.08s, vectorized vote 0.007s, same predictions: True
n_samples=200000: VotingClassifier.predict 0.65s, base estimators' predict 0.14s, vectorized vote 0.016s, same predictions: True
n_samples=400000: VotingClassifier.predict 1.30s, base estimators' predict 0.29s, vectorized vote 0.038s, same predictions: True
```

## Root cause and possible fix

`sklearn/ensemble/_voting.py:414-420` (on 289c4d5cfa):

```python
else:  # 'hard' voting
    predictions = self._predict(X)
    maj = np.apply_along_axis(
        lambda x: np.argmax(np.bincount(x, weights=self._weights_not_none)),
        axis=1,
        arr=predictions,
    )
```

Possible fix: allocate `counts = np.zeros((n_samples, n_classes))` and, for each estimator `i`, do `counts[np.arange(n_samples), predictions[:, i]] += w_i` (with `w_i = 1` when `weights` is None), then `maj = counts.argmax(axis=1)`. `argmax` returns the first maximum like `np.argmax(np.bincount(...))`, and the per-row sums are accumulated in the same estimator order, so predictions are identical, ties included. Memory is `n_samples * n_classes` floats, the same order as `predict_proba` with soft voting.

## Versions

```
sklearn: 1.10.dev0, commit 289c4d5cfa (upstream/main; checkout at f570497251, no diff in sklearn/ensemble/_voting.py)
python: 3.13, numpy: 2.5.3, scipy: 1.18.1, platform: linux
```
