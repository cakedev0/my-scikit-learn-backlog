# l1_min_c(loss="log", fit_intercept=True) is not the threshold for LogisticRegression, whose intercept is not penalized

```json
{
    "estimators": ["l1_min_c", "LogisticRegression"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 1,
    "difficulty": 2,
    "decision": 1,
    "decision_tags": ["output-change", "unclear-spec"]
}
```

## Classification

- **Priority (1)**: the returned value is too small (3x in the reproducer), so a regularization path started from it begins with a stretch of empty models. Not harmful for the fit itself, but the documented guarantee is wrong and the value is used to build paths (the docstring points to `plot_logistic_path.py`). Only with `fit_intercept=True` and non-centered features or imbalanced classes.
- **Difficulty (2)**: for `loss="log"` with `fit_intercept=True`, compute the bound with the optimal intercept, `max |X.T @ (y - y.mean())|` in the binary case, and the multinomial equivalent for 3+ classes; `intercept_scaling` then becomes meaningless for `loss="log"`.
- **Decision (1)**: `output-change`: the returned value changes for `loss="log", fit_intercept=True`. `unclear-spec`: decide whether `l1_min_c(loss="log")` targets the (deprecated) liblinear solver with a penalized intercept, or the other solvers.

## Upstream search

Queries: "l1_min_c", "l1_min_c intercept", "l1_min_c LogisticRegression" (issues and PRs).

No match. Related: #35067 (merged on main, "DEP liblinear, dual, intercept_scaling for LogisticRegression") switched `check_l1_min_c` to `solver="newton-cd"` and relaxed the `loss="log"` assertions, without changing `l1_min_c` or its docstring. #32042 (merged) introduced `alpha = 1 / C` and the docstring note. #18264 (closed) is a liblinear convergence issue.

## Describe the bug and its user-facing impact

The docstring of `l1_min_c` says: "The lower bound for C is computed such that for C in (l1_min_C, infinity) the model is guaranteed not to be empty. This applies to l1 penalized classifiers, such as LinearSVC with penalty='l1' and LogisticRegression with l1_ratio=1", and `fit_intercept` "must match the fit() method parameter".

The formula treats the intercept as an extra penalized feature of value `intercept_scaling`, which is what liblinear does. All other `LogisticRegression` solvers (and `liblinear` is being deprecated for it on main, #35067) do not penalize the intercept. With an unpenalized intercept, the optimum at `coef_ = 0` has `intercept_` equal to the log-odds of the class prior, and the coefficients stay at zero as long as `C * max_j |X[:, j] @ (y - y.mean())| <= 1`. `l1_min_c` uses `y - 1/2` instead of `y - y.mean()` (and adds the intercept column), so for imbalanced classes or non-centered features it returns a value below the real threshold, and `LogisticRegression` stays empty above it.

#35067 changed `check_l1_min_c` in `sklearn/svm/tests/test_bounds.py` to use `solver="newton-cd"` and only asserts `coef_ != 0 or intercept_ != 0` for `loss="log"`, which always holds since the intercept is free; the example `plot_logistic_path.py` uses `fit_intercept=False`, where the bound is exact.

## Steps/Code to Reproduce

```python
import warnings
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.svm import l1_min_c

warnings.simplefilter("ignore", FutureWarning)  # C is deprecated on main
X, y = make_classification(n_samples=300, n_features=10, weights=[0.8], random_state=0)
X += 3  # non-centered features

c_min = l1_min_c(X, y, loss="log", fit_intercept=True)
# Exact threshold with an unpenalized intercept: the gradient at coef=0 with
# the optimal intercept is X.T @ (y - y.mean()).
c_exact = 1 / np.abs(X.T @ (y - y.mean())).max()
print(f"l1_min_c: {c_min:.5f}, exact threshold: {c_exact:.5f}")

for factor in [1.01, 2, 5]:
    clf = LogisticRegression(l1_ratio=1, solver="saga", C=factor * c_min, tol=1e-10,
                             max_iter=100_000).fit(X, y)
    print(f"C = {factor:4} * l1_min_c -> n_nonzero_coef = {np.count_nonzero(clf.coef_)}")
```

(On main, `C` is deprecated in favor of `alpha = 1 / C`; the result is the same.)

## Expected Results

At least one non-zero coefficient for every `C > l1_min_c`, as documented.

## Actual Results

```
l1_min_c: 0.00357, exact threshold: 0.01116
C = 1.01 * l1_min_c -> n_nonzero_coef = 0
C =    2 * l1_min_c -> n_nonzero_coef = 0
C =    5 * l1_min_c -> n_nonzero_coef = 1
```

`2 * l1_min_c` is below the exact threshold, so the empty model there is the true optimum (KKT conditions), not a solver tolerance issue.

## Root cause and possible fix

`sklearn/svm/_bounds.py`, `l1_min_c` (upstream main, lines ~83-98): `Y = LabelBinarizer(neg_label=-1).fit_transform(y).T`, `den = max |Y @ X|`, and with `fit_intercept` the `intercept_scaling` column is added to `X`. This is the liblinear formulation.

Possible fix for `loss="log"` and `fit_intercept=True`: center the targets instead of adding the intercept column, i.e. `den = max |X.T @ (y01 - y01.mean())|` and return `1 / den` in the binary case; for 3+ classes, `LogisticRegression` is multinomial, and the gradient at `coef_ = 0` with optimal intercepts is `X.T @ (Y_onehot - class_priors)`. Keep the current formula for `loss="squared_hinge"` (`LinearSVC` does penalize the intercept). Alternatively, restrict the docstring guarantee to `fit_intercept=False` for `loss="log"`.

## Versions

```
sklearn: 1.10.dev0, commit 289c4d5cfa (upstream/main; reproducer ran on f570497251, whose l1_min_c only differs by a docstring line and whose LogisticRegression still uses C)
python: 3.13.14, numpy: 2.5.3, scipy: 1.18.1, platform: linux
```
