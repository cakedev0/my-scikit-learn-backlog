# HistGradientBoosting applies `interaction_cst` and recursion partial dependence to the wrong features when a categorical feature is not first

```json
{
    "estimators": ["HistGradientBoostingClassifier", "HistGradientBoostingRegressor", "partial_dependence", "PartialDependenceDisplay"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 3,
    "difficulty": 2,
    "decision": 0,
    "decision_tags": []
}
```

## Classification

- **Priority (3)**: silent and plausible-looking wrong results with default settings. A DataFrame with a categorical column that is not the first one, fitted with the default `categorical_features="from_dtype"`, is a very common setup, and `partial_dependence` / `PartialDependenceDisplay.from_estimator` use `method="recursion"` by default for HGB. The PD curves of numerical features come out flat or belong to another feature. `interaction_cst` is silently enforced on other features than the ones given. A low 3 (maintainer calibration, 2026-10-23): reaching it needs a somewhat niche combination (categorical column not first, plus `interaction_cst` or recursion partial dependence), but the results are silently wrong on an important estimator.
- **Difficulty (2)**: either map the feature indices into the internal column order in `fit` (interaction constraints) and in `_compute_partial_dependence_recursion`, or stop reordering the columns. The second option is what cakedev0/scikit-learn#30 does (see below).
- **Decision (0)**: no trade-off, the current results are wrong.

## Upstream search

- Core API dump of all open issues/PRs and of all issues labeled `module:ensemble`/`module:inspection`, grepped for interaction, recursion, partial dependence, categorical. `gh search issues` for "partial dependence recursion HistGradientBoosting categorical" (no result). The search for "interaction_cst categorical" failed (shared search API rate limit), so that part relies on the title dumps.
- #34554 (open PR): makes `partial_dependence` raise / fall back to brute for *categorical target features* (HGB recursion has no categorical branch). It doesn't cover this bug: here the target feature is numerical, and the wrong result comes from the column order.
- cakedev0/scikit-learn#30 (PR on a fork, stacked on upstream #35181, split from the closed #35162): "Use the categorical encoding of trees in HGB, keeping the column order". It lists and fixes exactly these bugs (`interaction_cst` on the wrong features, recursion PD on the wrong feature, and the cardinality error reporting the wrong feature index). Not opened upstream yet, and no upstream issue describes them.
- #28925 (merged) fixed the same column-order problem for `monotonic_cst`, which is why monotonic constraints are correct today.

## Describe the bug and its user-facing impact

When `categorical_features` is set (or detected from dtypes), `_preprocess_X` encodes X with a `ColumnTransformer` that outputs the categorical columns first. The trees are grown on this reordered X, so their `feature_idx` refer to the internal order. `monotonic_cst` is remapped to that order, but two other user inputs that are expressed in the original column order are not:

1. **`interaction_cst`**: the groups are applied to internal indices. With `X = [num0, cat1, num2]` and `interaction_cst=[[0, 2], [1]]`, the model forbids the 0-2 interaction the user asked for and lets the categorical feature interact with feature 2.
2. **`partial_dependence(method="recursion")`** (the default `"auto"` for HGB, also used by `PartialDependenceDisplay`): the target feature index and the grid are passed unchanged to the trees. The PD of feature 0 is computed for whatever feature sits at internal index 0, i.e. the categorical one, using a numerical threshold on it. The result is flat or meaningless while `method="brute"` gives the right curve.

A third, cosmetic consequence: the cardinality error in `_check_categories` reports the internal index ("Categorical feature at index 0") instead of the original one.

Nothing warns: everything works when all categorical features come first (or there are none), which is probably why tests didn't catch it.

## Steps/Code to Reproduce

```python
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import partial_dependence

rng = np.random.RandomState(0)
n = 2000
# feature 0: numerical, feature 1: categorical, feature 2: numerical
X = np.c_[rng.uniform(size=n), rng.randint(0, 3, size=n), rng.uniform(size=n)]
y = 10 * X[:, 0] * X[:, 2] + X[:, 1]

# 1. partial_dependence with the default method="auto" (i.e. "recursion")
est = HistGradientBoostingRegressor(categorical_features=[1], random_state=0)
est.fit(X, y)
for method in ["brute", "auto"]:
    pd = partial_dependence(est, X, features=[0], method=method, grid_resolution=5)
    print(f"PD of feature 0, method={method!r}: {pd['average'].round(2)}")

# 2. interaction_cst: features 0 and 2 may interact, feature 1 stays alone
est = HistGradientBoostingRegressor(
    categorical_features=[1], interaction_cst=[[0, 2], [1]], random_state=0
)
est.fit(X, y)
x0 = np.linspace(0, 1, 5)
def f(x0, cat, x2):
    return est.predict(np.c_[x0, np.full(5, cat), np.full(5, x2)])
print("f(x0, 0, 0.9) - f(x0, 0, 0.1):", (f(x0, 0, 0.9) - f(x0, 0, 0.1)).round(2))
print("f(x0, 2, 0.1) - f(x0, 0, 0.1):", (f(x0, 2, 0.1) - f(x0, 0, 0.1)).round(2))
print("f(x0, 2, 0.9) - f(x0, 0, 0.9):", (f(x0, 2, 0.9) - f(x0, 0, 0.9)).round(2))
```

## Expected Results

1. The recursion PD of feature 0 increases with feature 0, close to the brute one (`y` grows with `10 * x0 * x2`).
2. With `[[0, 2], [1]]`, the effect of `x2` may depend on `x0` (allowed interaction, present in `y`), and the effect of the category must not depend on `x2` (feature 1 is alone in its group), so the last two lines should be equal.

## Actual Results

```
PD of feature 0, method='brute': [[1.23 2.41 3.58 4.72 5.85]]
PD of feature 0, method='auto': [[-0.75 -0.75 -0.75 -0.75 -0.75]]
f(x0, 0, 0.9) - f(x0, 0, 0.1): [3.83 3.83 3.83 3.83 3.83]
f(x0, 2, 0.1) - f(x0, 0, 0.1): [1.67 1.67 1.67 1.67 1.67]
f(x0, 2, 0.9) - f(x0, 0, 0.9): [0.95 0.95 0.95 0.95 0.95]
```

The recursion PD is flat. The `x2` effect is the same for every `x0` (the allowed interaction is forbidden), and the category effect changes with `x2` (a forbidden interaction is used).

## Root cause and possible fix

`sklearn/ensemble/_hist_gradient_boosting/gradient_boosting.py` on the commit below:

- `_preprocess_X` (line 290) builds a `ColumnTransformer` with the encoder first, so the internal X is `[categorical columns, numerical columns]`. `fit` remaps `monotonic_cst` (line 501) but calls `self._check_interaction_cst(self._n_features)` (line 515) with the user's indices unchanged.
- `_compute_partial_dependence_recursion` (line 1295) passes `target_features` and `grid` to the predictors without mapping them to the internal order.
- `_check_categories` (line 332) builds `categorical_column_indices` from `output_indices_["encoder"]`, i.e. internal positions, for the error message.

Possible fixes:

- Keep the column order when encoding (what cakedev0/scikit-learn#30 does, by reusing the trees' encoding helpers that write each encoded column in place). This fixes all three at once.
- Or, minimal: compute the permutation once in `_preprocess_X` and apply it to the `interaction_cst` groups and to `target_features` in `_compute_partial_dependence_recursion` (plus #34554 for categorical target features).

Introduced with the `ColumnTransformer` based encoding in #26411 (1.4).

## Versions

```
sklearn: 1.10.dev0, commit 289c4d5cfa (upstream/main; run on a local editable build whose HGB code is identical to upstream/main except docstrings)
python: 3.13, numpy: 2.5.3, scipy: 1.18.1, platform: linux
```

See also: `inspection-hgb-pd-recursion-ignores-missing-values.md` (same recursion walker, NaN grid points routed without `missing_go_to_left`).
