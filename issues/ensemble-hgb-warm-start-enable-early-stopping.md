# HistGradientBoosting crashes when `early_stopping` is turned on between two `warm_start` fits with a scorer

```json
{
    "estimators": ["HistGradientBoostingClassifier", "HistGradientBoostingRegressor"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 1,
    "difficulty": 2,
    "decision": 1,
    "decision_tags": ["new-error", "unclear-spec"]
}
```

## Classification

- **Priority (1)**: changing `early_stopping` between warm-started fits is unusual, and the scorer case fails loudly. The `scoring="loss"` case is silent but only produces an inconsistent fit history and a validation set that was already used for training.
- **Difficulty (2)**: the warm-start branch of `fit` needs to set up the early stopping state (scorer, validation split) when it wasn't set up by the first fit, or `fit` must reject the change.
- **Decision (1)**: the simple fix is to raise a clear error when `do_early_stopping_` changes under `warm_start=True` (`new-error` for the `scoring="loss"` case, which runs today). What warm start should do with a newly enabled validation split (the first trees were trained on it) has no obvious answer (`unclear-spec`).

## Upstream search

Core API dump of all open issues/PRs and of issues labeled `module:ensemble`, grepped for warm start and early stopping. `gh search issues` for "warm_start early_stopping HistGradientBoosting scoring" (no result). Related but different:
- #13285 (open): general question "Should fitting with warm_start=True validate changes in parameters?".
- #14034 / #15032 (open): data leak with warm start in the old `GradientBoosting*` (validation split changing between fits).
- #16662 / #16663 (closed, merged fix): HGB warm start with early stopping on both fits and no validation data.

None covers enabling early stopping on a warm-started HGB.

## Describe the bug and its user-facing impact

With `warm_start=True`, if the first `fit` runs without early stopping and the second one with early stopping (`early_stopping=True`, or `"auto"` with more than 10k samples), `fit` uses state that the first fit never created:

- with a scorer (`scoring` other than `"loss"`), `self._scorer` is still `None` and `fit` crashes with `TypeError: 'NoneType' object is not callable`;
- with `scoring="loss"`, `fit` runs but the new trees are fitted on the training part of a validation split, while the first trees were fitted on all the data (validation samples included), and `train_score_`/`validation_score_` have 5 entries for 10 iterations, while the docstring says shape `(n_iter_ + 1,)`.

## Steps/Code to Reproduce

```python
from sklearn.datasets import make_classification
from sklearn.ensemble import HistGradientBoostingClassifier

X, y = make_classification(n_samples=500, random_state=0)
clf = HistGradientBoostingClassifier(
    max_iter=5, early_stopping=False, scoring="accuracy", warm_start=True
)
clf.fit(X, y)
clf.set_params(max_iter=10, early_stopping=True)
clf.fit(X, y)
```

and, for the silent case:

```python
from sklearn.datasets import make_classification
from sklearn.ensemble import HistGradientBoostingClassifier

X, y = make_classification(n_samples=500, random_state=0)
clf = HistGradientBoostingClassifier(max_iter=5, early_stopping=False, warm_start=True)
clf.fit(X, y)
clf.set_params(max_iter=10, early_stopping=True)  # scoring="loss"
clf.fit(X, y)
print(clf.n_iter_, clf.train_score_.shape, clf.validation_score_.shape)
```

## Expected Results

Either early stopping starts working from the 6th iteration on (scorer and validation split created on the second fit), or a clear error says that `early_stopping` can't be changed when warm starting.

## Actual Results

```
Traceback (most recent call last):
  File ".../repro_ws.py", line 10, in <module>
    clf.fit(X, y)
  File "/home/arthur/scikit-learn/sklearn/base.py", line 1403, in wrapper
    return fit_method(estimator, *args, **kwargs)
  File "/home/arthur/scikit-learn/sklearn/ensemble/_hist_gradient_boosting/gradient_boosting.py", line 903, in fit
    should_early_stop = self._check_early_stopping_scorer(
  ...
  File "/home/arthur/scikit-learn/sklearn/ensemble/_hist_gradient_boosting/gradient_boosting.py", line 1051, in _score_with_raw_predictions
    return self._scorer(self, X, y)
           ~~~~~~~~~~~~^^^^^^^^^^^^
TypeError: 'NoneType' object is not callable
```

Second snippet:

```
10 (5,) (5,)
```

## Root cause and possible fix

In `sklearn/ensemble/_hist_gradient_boosting/gradient_boosting.py` on the commit below, the scorer is only created in the "first fit" branch (`self._scorer = None` at line 657, `self._scorer = check_scoring(...)` at line 695, under `if self.do_early_stopping_:`). The warm-start branch (lines 738-775) recomputes `do_early_stopping_` from the current parameters and builds the small training subset when `self.do_early_stopping_ and self.scoring != "loss"` (line 761), but never creates the scorer, so `_score_with_raw_predictions` calls `None` (line 1051). The initial scores (before the first new tree) are not recorded either, hence the short `train_score_`.

Possible fixes: raise a `ValueError` in the warm-start branch when `do_early_stopping_` differs from the previous fit (simplest, consistent with the `max_iter < n_iter_` check there), or create `self._scorer` in the warm-start branch when it is missing. The second one still leaves the leakage of the new validation set into the first trees.

## Versions

```
sklearn: 1.10.dev0, commit 289c4d5cfa (upstream/main; run on a local editable build whose gradient_boosting.py differs from upstream/main only in docstrings)
python: 3.13, numpy: 2.5.3, scipy: 1.18.1, platform: linux
```
