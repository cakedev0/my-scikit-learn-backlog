# LinearSVC/LinearSVR docstring misdescribes `dual="auto"`: the default LinearSVR always uses the dual

```json
{
    "estimators": ["LinearSVR", "LinearSVC"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 1,
    "difficulty": 1,
    "decision": 0,
    "decision_tags": []
}
```

## Classification

- **Priority (1)**: documentation only; the code does the sensible thing, but users reading the docstring get a wrong idea of which solver runs (e.g. that the default `LinearSVR` solves the primal on tall data).
- **Difficulty (1)**: rewrite one sentence in two docstrings.
- **Decision (0)**: no behavior change.

## Upstream search

- `gh search issues ... LinearSVR dual auto` and `dual auto docstring LinearSVC`: only #24731 (closed) and #26093 (merged), the PRs that added `dual="auto"`. No report of the docstring mismatch.

## Describe the bug and its user-facing impact

The `dual` docstring of `LinearSVR` says:

> If `n_samples` < `n_features` and optimizer supports chosen `loss`, then dual will be set to True, otherwise it will be set to False.

(`LinearSVC` has the same sentence, also mentioning `multi_class` and `penalty`.) The code (`_validate_dual_parameter`) does more: when `n_samples >= n_features` but the primal is not supported for the chosen loss, it picks `dual=True`. This is the case for the default `LinearSVR(loss="epsilon_insensitive")` and for `LinearSVC(loss="hinge")`. So the default `LinearSVR` always solves the dual, whatever the data shape, while the docstring says it switches to the primal for `n_samples >= n_features`. Setting `dual=False` explicitly then raises, which is confusing if one trusts the docstring.

## Steps/Code to Reproduce

```python
import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.svm import LinearSVC, LinearSVR

# n_samples > n_features: the docstring says dual="auto" picks dual=False
X, y = make_regression(n_samples=200, n_features=5, random_state=0)
auto = LinearSVR(dual="auto", random_state=0).fit(X, y)
dual = LinearSVR(dual=True, random_state=0).fit(X, y)
print("LinearSVR auto == dual=True:", np.array_equal(auto.coef_, dual.coef_))
try:
    LinearSVR(dual=False).fit(X, y)
except ValueError as e:
    print("LinearSVR dual=False:", e)

X, y = make_classification(n_samples=200, n_features=5, random_state=0)
auto = LinearSVC(loss="hinge", dual="auto", random_state=0).fit(X, y)
dual = LinearSVC(loss="hinge", dual=True, random_state=0).fit(X, y)
print("LinearSVC(loss='hinge') auto == dual=True:", np.array_equal(auto.coef_, dual.coef_))
```

## Expected Results

The docstring describes the actual rule: prefer `dual=True` when `n_samples < n_features` and `dual=False` otherwise, falling back to the other formulation when the preferred one does not support the chosen `loss`/`penalty`/`multi_class`.

## Actual Results

```
LinearSVR auto == dual=True: True
LinearSVR dual=False: Unsupported set of arguments: The combination of penalty='l2' and loss='epsilon_insensitive' are not supported when dual=False, Parameters: penalty='l2', loss='epsilon_insensitive', dual=False
LinearSVC(loss='hinge') auto == dual=True: True
```

## Root cause and possible fix

The rule lives in `_validate_dual_parameter` (`sklearn/svm/_classes.py:25-41`). The docstrings are at `sklearn/svm/_classes.py:74-81` (`LinearSVC`) and `sklearn/svm/_classes.py:421-427` (`LinearSVR`). Update both to describe the fallback; for `LinearSVR`, it is worth saying explicitly that the default loss only supports the dual.

## Versions

```
sklearn: 1.10.dev0, commit 289c4d5cfa
python: 3.13, numpy: 2.5.3, scipy: 1.18.1, platform: linux
```
