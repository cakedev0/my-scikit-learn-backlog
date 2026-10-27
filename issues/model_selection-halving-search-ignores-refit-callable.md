# HalvingGridSearchCV and HalvingRandomSearchCV silently ignore a callable `refit`

```json
{
    "estimators": ["HalvingGridSearchCV", "HalvingRandomSearchCV"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 2,
    "difficulty": 2,
    "decision": 1,
    "decision_tags": ["unclear-spec", "new-error"]
}
```

## Classification

- **Priority (2)**: the docstring documents `refit=callable` and the parameter validation accepts it, but the callable is never called. The search silently returns the best-scoring candidate instead of the one the user asked for. Experimental estimators, and the callable use case is not the most common one.
- **Difficulty (2)**: either call the callable in `BaseSuccessiveHalving._select_best_index`, or reject it. Calling it needs to decide which results the callable sees (see below).
- **Decision (1)**: `unclear-spec`: since #34250, `cv_results_` only holds the last iteration while `all_cv_results_` holds every iteration. Someone has to decide whether the callable gets the last iteration (and returns an index in it) or the full history. `new-error`: if we go back to rejecting callables, code that passes one today (and silently gets the default selection) would raise.

## Upstream search

`gh issue list --search` / `gh pr list --search` (GraphQL, REST search was rate limited): "HalvingGridSearchCV refit callable", "successive halving refit callable ignored", "halving refit callable", "halving search refit callable best_index_", "HalvingRandomSearchCV refit". No match.

History from `git log -S`:

- #20203 (2021) added an explicit check: "refit is expected to be a boolean".
- #25151 (2023, parameter validation for SearchCV) removed that check, so `refit` falls back to the `BaseSearchCV` constraint `["boolean", str, callable]`.
- #30990 (2025, DOC) changed the Halving docstrings from `refit : bool` to `refit : bool or callable` and copied the `GridSearchCV` text about callables, but the implementation never supported it.

## Describe the bug and its user-facing impact

The docstring of both halving searches says:

> Where there are considerations other than maximum score in choosing a best estimator, `refit` can be set to a function which returns the selected `best_index_` given `cv_results_`. In that case, the `best_estimator_` and `best_params_` will be set according to the returned `best_index_` while the `best_score_` attribute will not be available.

With a halving search, the callable is never called. `best_index_`, `best_params_`, `best_estimator_` and `best_score_` are those of the best-scoring candidate of the last iteration. A user following the linked example (`plot_grid_search_refit_callable.py`) to trade score for model simplicity gets no error and no warning, just a different model than the one they selected.

## Steps/Code to Reproduce

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import GridSearchCV, HalvingGridSearchCV
from sklearn.tree import DecisionTreeClassifier

X, y = make_classification(n_samples=400, random_state=0)
param_grid = {"max_depth": [1, 2, 3, 4, 5, 6, 7, 8, 9]}


def refit(cv_results):
    print("  refit callable called")
    # pick the worst ranked candidate to make the effect obvious
    return int(np.argmax(cv_results["rank_test_score"]))


for search in [
    GridSearchCV(DecisionTreeClassifier(random_state=0), param_grid, refit=refit),
    HalvingGridSearchCV(
        DecisionTreeClassifier(random_state=0), param_grid, refit=refit, random_state=0
    ),
]:
    print(type(search).__name__)
    search.fit(X, y)
    rank = search.cv_results_["rank_test_score"][search.best_index_]
    print("  rank of selected candidate:", rank)
    print("  has best_score_:", hasattr(search, "best_score_"))
```

## Expected Results

As for `GridSearchCV`: the callable is called, the selected candidate is the worst ranked one, and `best_score_` is not set. Or, if callables are not supported, a clear error at `fit`.

## Actual Results

```
GridSearchCV
  refit callable called
  rank of selected candidate: 9
  has best_score_: False
HalvingGridSearchCV
  rank of selected candidate: 1
  has best_score_: True
```

## Root cause and possible fix

`BaseSuccessiveHalving._select_best_index` (`sklearn/model_selection/_search_successive_halving.py:220`) overrides the static method of `BaseSearchCV` (`sklearn/model_selection/_search.py:866`, which calls `refit(results)` when it is callable) and ignores its `refit` argument: it always returns the `nanargmax` of the last iteration's `mean_test_score`. `BaseSuccessiveHalving.fit` then also sets `best_score_` unconditionally.

Two options:

1. Support it: in `_select_best_index`, if `refit` is callable, call it on the last-iteration slice of the results (what `cv_results_` exposes after `fit`) and map the returned index back. Skip `best_score_` in that case, as `BaseSearchCV` does.
2. Reject it: restrict the `refit` constraint of `BaseSuccessiveHalving` to `["boolean"]` and revert the docstring to `refit : bool`.

I'd go for option 1, since #34250 made `cv_results_` match what a callable would naturally index into.

## Versions

```
sklearn: 1.10.dev0, commit f570497251 (local branch; _search_successive_halving.py
is identical on upstream/main 289c4d5cfa, _search.py only differs in a docstring)
python: 3.13, numpy: 2.5.3, scipy: 1.18.1, platform: linux
```
