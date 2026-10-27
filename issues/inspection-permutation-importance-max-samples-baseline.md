# `permutation_importance` with `max_samples < 1` gives non-zero importance to features the model never uses

```json
{
    "estimators": ["permutation_importance"],
    "existing_issue": "https://github.com/scikit-learn/scikit-learn/issues/35027",
    "reproduced": true,
    "priority": 2,
    "difficulty": 1,
    "decision": 1,
    "decision_tags": ["output-change"]
}
```

## Classification

- **Priority (2)**: silent wrong importances on a widely used inspection tool, but only with the non-default `max_samples < 1`, and the error is a constant offset shared by all features and repeats, so the ranking and the gaps between features are unchanged; mostly the sign of near-zero importances is misleading. (priority review, 2026-10-23: was 3)
- **Difficulty (1)**: score the baseline on the same subsample as the permuted scores.
- **Decision (1)**: `output-change`: importances change for every user of `max_samples < 1`. They become correct, but results are not reproducible across versions.

## Upstream search

`gh issue list --search` (GraphQL): "permutation_importance max_samples", "permutation importance baseline score subsample".

- #35027 (open, "Needs Triage"): same bug, same root cause. The reporter proposes the fix described below; no PR yet.
- #28183 (closed): `max_samples` did not work with `sample_weight`, fixed. Different bug.

## Describe the bug and its user-facing impact

With `max_samples < n_samples`, the baseline score is computed on all of `X`, while the permuted scores are computed on a subsample of `max_samples` rows. The importance `baseline - permuted` therefore includes the score difference between the full data and the subsample. All features (and all repeats) share the same subsample, because each worker gets the same seed, so this offset is the same for every feature and every repeat: the std is 0 for an unused feature.

The docstring also says the samples are drawn "in each repeat", while the code draws them once per feature (the same rows for all features and repeats).

## Steps/Code to Reproduce

```python
from sklearn.datasets import make_regression
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression

X, y = make_regression(n_samples=1000, n_features=3, noise=50, random_state=0)
X[:, 2] = 0  # the model cannot use feature 2: its coefficient is exactly 0
reg = LinearRegression().fit(X, y)
print("coef_:", reg.coef_.round(2))
for max_samples in [1.0, 0.1]:
    r = permutation_importance(reg, X, y, max_samples=max_samples, random_state=0)
    print(f"max_samples={max_samples}: importances of feature 2 =",
          r.importances[2].round(4))
```

## Expected Results

Permuting a constant feature with a zero coefficient doesn't change the predictions, so its importance is exactly 0 for any `max_samples`.

## Actual Results

```
coef_: [71.1  93.91  0.  ]
max_samples=1.0: importances of feature 2 = [0. 0. 0. 0. 0.]
max_samples=0.1: importances of feature 2 = [-0.0669 -0.0669 -0.0669 -0.0669 -0.0669]
```

## Root cause and possible fix

`sklearn/inspection/_permutation_importance.py:284` computes `baseline_score` on the full `X`, while `_calculate_permutation_scores` (line 51) subsamples the rows before permuting.

Fix (as proposed in #35027): draw the subsample once in `permutation_importance` and compute the baseline on it, then permute within it. Alternatively, draw a new subsample per repeat with its own baseline, which matches the "in each repeat" docstring but changes the sampling more.

## Versions

```
sklearn: 1.10.dev0, commit f570497251 (local branch; _permutation_importance.py is
identical on upstream/main 289c4d5cfa)
python: 3.13, numpy: 2.5.3, scipy: 1.18.1, platform: linux
```
