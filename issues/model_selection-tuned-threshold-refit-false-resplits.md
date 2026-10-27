# TunedThresholdClassifierCV with `refit=False` fits `estimator_` and tunes the threshold on different random splits

```json
{
    "estimators": ["TunedThresholdClassifierCV"],
    "existing_issue": "https://github.com/scikit-learn/scikit-learn/issues/33540",
    "reproduced": true,
    "priority": 2,
    "difficulty": 1,
    "decision": 1,
    "decision_tags": ["unclear-spec", "rng-order"]
}
```

## Classification

- **Priority (2)**: silent. With `cv` a float (or any single-split shuffling splitter) and `random_state=None` (the default), `estimator_` is trained on part of the validation set used to tune `best_threshold_`, and the threshold was tuned for another model. It needs the non-default `refit=False`.
- **Difficulty (1)**: compute the split once and reuse it for both fits.
- **Decision (1)**: `unclear-spec`: in #33540, a maintainer asked why the same split should be used; the existing test `test_tuned_threshold_classifier_cv_float` asserts that `estimator_` is fitted on the train part of the split, but only with `random_state=0`. `rng-order`: the fix consumes the random state once instead of twice.

## Upstream search

`gh issue list --search` (GraphQL): "TunedThresholdClassifierCV refit False".

- #33540 (open, "Needs Decision"): same bug, same root cause. #33560 (a PR from another contributor) was closed in favour of the reporter's fix, which is not opened yet. @betatim asked for the rationale; no decision so far.

## Describe the bug and its user-facing impact

With `refit=False`, `TunedThresholdClassifierCV` is meant to use a single train/validation split: `estimator_` is fitted on the train part, and the threshold is tuned on the validation part. The code calls `cv.split(X, y)` twice: once to fit `estimator_` and once to fit the clone used for tuning. When the splitter is random and not seeded (e.g. `cv=0.5` with the default `random_state=None`), the two calls give different splits. `estimator_` (used by `predict`) then differs from the model whose validation scores picked `best_threshold_`, and it was trained on ~half of the samples used to tune the threshold, which is the leakage `refit=False` with a held-out split is supposed to avoid.

## Steps/Code to Reproduce

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TunedThresholdClassifierCV

train_sets = []


class RecordingLR(LogisticRegression):
    def fit(self, X, y):
        train_sets.append(frozenset(X[:, 0].tolist()))  # column 0 is a sample id
        return super().fit(X, y)


X, y = make_classification(n_samples=200, random_state=0)
X[:, 0] = np.arange(len(X)) / len(X)
TunedThresholdClassifierCV(RecordingLR(), cv=0.5, refit=False).fit(X, y)
estimator_train, tuning_train = train_sets
print("same training set:", estimator_train == tuning_train)
print("validation samples of the tuning split used to fit estimator_:",
      len(estimator_train - tuning_train))
```

## Expected Results

`same training set: True`, and 0 validation samples used to fit `estimator_`, as with `random_state=0` (what `test_tuned_threshold_classifier_cv_float` checks).

## Actual Results

```
same training set: False
validation samples of the tuning split used to fit estimator_: 48
```

## Root cause and possible fix

`sklearn/model_selection/_classification_threshold.py:757` creates the `splits` generator and line 764 calls `next(cv.split(...))` again for `estimator_`. Fix: materialize `splits = list(cv.split(...))` once and, when `refit=False`, take `train_idx` from `splits[0]`. That also allows reusing the fitted model instead of fitting the same split twice.

## Versions

```
sklearn: 1.10.dev0, commit f570497251 (local branch; _classification_threshold.py is
identical on upstream/main 289c4d5cfa)
python: 3.13, numpy: 2.5.3, scipy: 1.18.1, platform: linux
```
