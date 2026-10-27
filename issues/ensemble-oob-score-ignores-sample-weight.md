# `oob_score_` of forests and Bagging ignores `sample_weight`, so zero-weighted samples count in the score

```json
{
    "estimators": ["RandomForestClassifier", "RandomForestRegressor", "ExtraTreesClassifier", "ExtraTreesRegressor", "BaggingClassifier", "BaggingRegressor"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 2,
    "difficulty": 2,
    "decision": 1,
    "decision_tags": ["output-change", "unclear-spec"]
}
```

## Classification

- **Priority (2)**: silent and plausible-looking wrong estimate whenever `oob_score=True` is used with non-uniform `sample_weight`. Zero weights, the usual way to exclude rows, make it obvious: those rows are never drawn, so they are always out-of-bag and enter the score with full weight.
- **Difficulty (2)**: passing `sample_weight` to the default scorers is a few lines, but it touches four `_set_oob_score*` methods and the callable `oob_score` case needs a convention.
- **Decision (1)**: `output-change`: `oob_score_` changes for every weighted fit. `unclear-spec`: the documented callable signature is `metric(y_true, y_pred)`, so one has to decide whether to pass `sample_weight` to it (only if it accepts it?) or at least drop zero-weighted rows.

## Upstream search

GraphQL search on scikit-learn/scikit-learn: "oob_score sample_weight", "oob score ignores sample weight". Only unrelated results (#18603 OOB permutation importance, #24882 GB OOB scores, #27814 OOB target type).

- #16298 (open) lists estimators with incorrect `sample_weight` handling; it does not mention OOB scores.
- #31529 / #31414 (merged, 1.9 / 1.8): draw bootstrap indices with probabilities proportional to `sample_weight`. Since then zero-weighted samples are never in-bag, which makes this bug more visible. Before, zero-weighted rows were drawn uniformly (with a tree weight of 0) and still counted in `oob_score_` unweighted.

No match for this bug.

## Describe the bug and its user-facing impact

`fit(X, y, sample_weight)` with `oob_score=True` computes `oob_score_` as an unweighted `accuracy_score` / `r2_score` over all training rows that got at least one OOB prediction. `sample_weight` is not used. So:

- rows with `sample_weight=0` (meant to be ignored) are always out-of-bag and count like any other row;
- with positive weights, low-weight rows are out-of-bag more often and the score is not the weighted generalization estimate a user would compare with `score(X_test, y_test, sample_weight=w_test)`.

In the reproducer, half the rows have random labels and zero weight. The model is the same as when these rows are dropped, but `oob_score_` falls from ~0.95 to ~0.71.

## Steps/Code to Reproduce

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier

X, y = make_classification(n_samples=1000, random_state=0)
# Corrupt the labels of the second half, and exclude it with zero weights.
y[500:] = np.random.RandomState(0).randint(0, 2, 500)
sw = np.r_[np.ones(500), np.zeros(500)]

for est in [RandomForestClassifier(oob_score=True, random_state=0),
            BaggingClassifier(n_estimators=50, max_samples=1.0, oob_score=True, random_state=0)]:
    weighted = est.fit(X, y, sample_weight=sw).oob_score_
    dropped = est.fit(X[:500], y[:500]).oob_score_
    print(f"{type(est).__name__}: zero-weighted half {weighted:.3f}, half dropped {dropped:.3f}")
```

## Expected Results

Zero-weighted rows do not contribute, so both columns are close (~0.95). More generally, `oob_score_` is the `sample_weight`-weighted score of the OOB predictions, consistent with `score(..., sample_weight=...)`.

## Actual Results

```
RandomForestClassifier: zero-weighted half 0.710, half dropped 0.952
BaggingClassifier: zero-weighted half 0.709, half dropped 0.950
```

## Root cause and possible fix

The OOB scores are computed without weights:

- `sklearn/ensemble/_forest.py:894` (`ForestClassifier`) and `sklearn/ensemble/_forest.py:1220` (`ForestRegressor`) on upstream/main: `scoring_function(y, ...)`;
- `sklearn/ensemble/_bagging.py:921` (`BaggingClassifier`) and `sklearn/ensemble/_bagging.py:1473` (`BaggingRegressor`).

The user `sample_weight` is available at that point (forests store the combined `self._sample_weight`; for scoring, the user weights without `class_weight` are the natural choice). Fix: pass `sample_weight` to `accuracy_score` / `r2_score`, restricted to rows with at least one OOB prediction. For a callable `oob_score`, either pass `sample_weight` when the callable accepts it, or at least exclude zero-weighted rows.

## Versions

```
sklearn: 1.10.dev0, commit f570497251 (OOB code identical to upstream/main 289c4d5cfa, line numbers above are from upstream/main for _forest.py)
python: 3.13, numpy: 2.5.3, scipy: 1.18.1, platform: linux
```

See also: `issues/ensemble-oob-score-counts-never-oob-samples-as-zero.md` (rows without any OOB prediction counted as predicting 0 / the first class).
