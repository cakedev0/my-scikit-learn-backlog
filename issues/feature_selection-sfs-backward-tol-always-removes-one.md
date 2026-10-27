# SequentialFeatureSelector(direction="backward", tol=...) always removes at least one feature, even an informative one

```json
{
    "estimators": ["SequentialFeatureSelector"],
    "existing_issue": "https://github.com/scikit-learn/scikit-learn/issues/26369",
    "reproduced": true,
    "priority": 2,
    "difficulty": 1,
    "decision": 1,
    "decision_tags": ["output-change", "perf-low"]
}
```

## Classification

- **Priority (2)**: silent and can be costly (CV R2 1.00 -> 0.88 below), on the documented use of `tol` with backward selection. Only `n_features_to_select="auto"` with `tol` set and `direction="backward"` is affected.
- **Difficulty (1)**: initialize `old_score` with the CV score of the full feature set when going backward.
- **Decision (1)**: changes the selected features for affected users (`output-change`) and costs one more `cross_val_score` call per fit (`perf-low`, negligible compared to the n_features calls per iteration).

## Upstream search

- `gh issue list --search "SequentialFeatureSelector tol"`: #26369 (open, Bug) is exactly this bug, with a reproducer from 2023. No linked PR found.
- #25525 (closed) is about documenting negative `tol`, unrelated.
- Filed here because the issue is still open on main; the reproducer below is shorter and uses an informative feature.

## Describe the bug and its user-facing impact

The `tol` docstring says: "If the score is not incremented by at least `tol` between two consecutive feature additions or removals, stop adding or removing." In `fit`, `old_score` starts at `-np.inf` for both directions. For backward selection, the first removal is therefore always accepted (`new_score - (-inf) >= tol`), without comparing to the score with all features. If every feature matters, the selector still drops one, and the user gets a worse model with no warning.

## Steps/Code to Reproduce

```python
from sklearn.datasets import make_regression
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

X, y = make_regression(n_samples=200, n_features=5, n_informative=5, noise=1, random_state=0)
sfs = SequentialFeatureSelector(
    LinearRegression(), direction="backward", n_features_to_select="auto", tol=1e-3
).fit(X, y)
print("support:", sfs.get_support())
print("CV R2, all features:     ", cross_val_score(LinearRegression(), X, y).mean())
print("CV R2, selected features:", cross_val_score(LinearRegression(), sfs.transform(X), y).mean())
```

## Expected Results

All 5 informative features are kept, since removing any of them decreases the score by far more than `tol`.

## Actual Results

```
support: [ True False  True  True  True]
CV R2, all features:      0.9999659624937726
CV R2, selected features: 0.875517548073111
```

## Root cause and possible fix

`sklearn/feature_selection/_sequential.py:276`: `old_score = -np.inf`, used at line 288 for both directions. For `direction="backward"`, set `old_score` to `cross_val_score(cloned_estimator, X, y, cv=cv, scoring=self.scoring, params=params).mean()` before the loop (and keep `-inf` for forward, where the empty model has no score).

## Versions

```
sklearn: 1.10.dev0, commit 289c4d5cfa (upstream/main; reproduced on local checkout f570497251, no diff in sklearn/feature_selection/_sequential.py)
python: 3.13, numpy: 2.5.3, scipy: 1.18.1, platform: linux
```
