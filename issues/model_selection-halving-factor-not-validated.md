# Halving searches accept `factor <= 1` and then crash with ZeroDivisionError or IndexError

```json
{
    "estimators": ["HalvingGridSearchCV", "HalvingRandomSearchCV"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 1,
    "difficulty": 1,
    "decision": 0,
    "decision_tags": []
}
```

## Classification

- **Priority (1)**: `factor <= 1` makes no sense for successive halving, so this only affects users who mistype the parameter. They get a cryptic error deep in `_run_search` instead of a parameter validation error.
- **Difficulty (1)**: change the constraint to `Interval(Real, 1, None, closed="neither")`.
- **Decision (0)**: these values already fail, just with a worse message.

## Upstream search

`gh issue list --search` (GraphQL): "HalvingGridSearchCV factor 1 ZeroDivisionError", "halving factor validation". No match (only #27422 and #27740, open, about other halving behaviors).

## Describe the bug and its user-facing impact

The parameter constraint for `factor` is `Interval(Real, 0, None, closed="neither")`, so `factor=1` or `factor=0.5` pass validation. Successive halving keeps `1 / factor` of the candidates at each iteration and computes `log(n_candidates, factor)`, which is only meaningful for `factor > 1`. With `factor=1`, `fit` raises `ZeroDivisionError: float division by zero`; with `factor=0.5`, it raises `IndexError: list index out of range`.

## Steps/Code to Reproduce

```python
from sklearn.datasets import make_classification
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.tree import DecisionTreeClassifier

X, y = make_classification(n_samples=200, random_state=0)
for factor in [1, 0.5]:
    search = HalvingGridSearchCV(
        DecisionTreeClassifier(), {"max_depth": [1, 2, 3, 4]}, factor=factor
    )
    try:
        search.fit(X, y)
    except Exception as e:
        print(f"factor={factor}: {type(e).__name__}: {e}")
```

## Expected Results

`InvalidParameterError: The 'factor' parameter of HalvingGridSearchCV must be a float in the range (1, inf)...`

## Actual Results

```
factor=1: ZeroDivisionError: float division by zero
factor=0.5: IndexError: list index out of range
```

with, for `factor=1`:

```
  File ".../sklearn/model_selection/_search_successive_halving.py", line 300, in _run_search
    n_required_iterations = 1 + floor(log(len(candidate_params), self.factor))
ZeroDivisionError: float division by zero
```

## Root cause and possible fix

`sklearn/model_selection/_search_successive_halving.py:113`: `"factor": [Interval(Real, 0, None, closed="neither")]`. Use `Interval(Real, 1, None, closed="neither")` and update the docstring (`factor : int or float, default=3`) to say it must be greater than 1.

## Versions

```
sklearn: 1.10.dev0, commit f570497251 (local branch; _search_successive_halving.py
is identical on upstream/main 289c4d5cfa)
python: 3.13, numpy: 2.5.3, scipy: 1.18.1, platform: linux
```
