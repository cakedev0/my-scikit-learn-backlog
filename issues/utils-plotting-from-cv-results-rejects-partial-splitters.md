# `RocCurveDisplay.from_cv_results` rejects `ShuffleSplit` and `TimeSeriesSplit` results

```json
{
    "estimators": ["RocCurveDisplay", "PrecisionRecallDisplay"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 2,
    "difficulty": 1,
    "decision": 0,
    "decision_tags": []
}
```

## Classification

- **Priority (2)**: a loud but misleading error on valid input with common splitters (`TimeSeriesSplit`, `ShuffleSplit` with `train_size`). There is no workaround through the public API besides building the curves by hand with `from_predictions`.
- **Difficulty (1)**: replace one consistency check in a shared helper.
- **Decision (0)**: the new check still catches a wrong `X`, without rejecting valid input.

## Upstream search

Queries: "from_cv_results ShuffleSplit", "from_cv_results TimeSeriesSplit", "does not contain the correct number of samples". The check was added with `from_cv_results` in #30399 (merged, 2025, "FEA add `from_cv_results` in `RocCurveDisplay`"); the only test of the message passes `X[:10]` with a `KFold` result.

## Describe the bug and its user-facing impact

`from_cv_results` checks that the `X` passed matches the one used in `cross_validate` with `n_samples == len(train_0) + len(test_0)`, using the first fold. That only holds for splitters whose train and test sets partition the data (`KFold`, `StratifiedKFold`, ...). It fails for valid results of:

- `TimeSeriesSplit`: the first fold uses only the beginning of the data;
- `ShuffleSplit` / `StratifiedShuffleSplit` with `train_size + test_size < 1`;
- custom iterables of `(train, test)` pairs that leave samples out.

Users get `` ValueError: `X` does not contain the correct number of samples `` although they passed the very same `X`.

## Steps/Code to Reproduce

```python
import matplotlib
matplotlib.use("Agg")
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay
from sklearn.model_selection import ShuffleSplit, TimeSeriesSplit, cross_validate

X, y = make_classification(n_samples=100, random_state=0)
for cv in [ShuffleSplit(n_splits=3, train_size=0.5, test_size=0.2, random_state=0),
           TimeSeriesSplit(n_splits=3)]:
    cv_results = cross_validate(
        LogisticRegression(), X, y, cv=cv, return_estimator=True, return_indices=True
    )
    for display in [RocCurveDisplay, PrecisionRecallDisplay]:
        try:
            display.from_cv_results(cv_results, X, y)
            print(type(cv).__name__, display.__name__, "OK")
        except ValueError as e:
            print(type(cv).__name__, display.__name__, "ValueError:", e)
```

## Expected Results

Four "OK" lines: each fold's estimator is evaluated on its test indices, which are all valid rows of `X`.

## Actual Results

```
ShuffleSplit RocCurveDisplay ValueError: `X` does not contain the correct number of samples. Expected 70, got 100.
ShuffleSplit PrecisionRecallDisplay ValueError: `X` does not contain the correct number of samples. Expected 70, got 100.
TimeSeriesSplit RocCurveDisplay ValueError: `X` does not contain the correct number of samples. Expected 50, got 100.
TimeSeriesSplit PrecisionRecallDisplay ValueError: `X` does not contain the correct number of samples. Expected 50, got 100.
```

## Root cause and possible fix

`_BinaryClassifierCurveDisplayMixin._validate_from_cv_results_params` in `sklearn/utils/_plotting.py:92-101` (upstream/main 289c4d5cfa) assumes that train and test indices of the first fold cover all samples.

A check that holds for any splitter: the largest train or test index over all folds must be smaller than `_num_samples(X)`. It still catches the case tested today (`X[:10]` with a 3-fold result). It can't detect an `X` with extra trailing rows, which seems an acceptable loss.

## Versions

```
sklearn: 1.10.dev0, commit 289c4d5cfa (upstream/main; local build f570497251, no diff on sklearn/utils/_plotting.py)
python: 3.13, numpy: 2.5.3, scipy: 1.18.1, platform: linux
```
