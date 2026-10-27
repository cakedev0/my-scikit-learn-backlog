# LogisticRegressionCV(refit=False, l1_ratios=(1,)) reports l1_ratio_=0 for a pure L1 fit

```json
{
    "estimators": ["LogisticRegressionCV"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 1,
    "difficulty": 1,
    "decision": 0,
    "decision_tags": []
}
```

## Classification

- **Priority (1)**: only a fitted attribute is wrong, the coefficients are fine. But `l1_ratio_` is what a user reads (or passes to a `LogisticRegression` to refit by hand) to know which penalty was selected, and `0.0` means pure L2 while the model was fit with pure L1.
- **Difficulty (1)**: in the `refit=False` branch, use the selected `l1_ratios_` value for every penalty, as the `refit=True` branch does.
- **Decision (0)**: no trade-off.

## Upstream search

Queries: "LogisticRegressionCV l1_ratio_", "LogisticRegressionCV l1_ratio_ refit". Only #33885 (closed), the `TypeError` fixed by #33902, comes up.

#33902 (merged) replaced `self.l1_ratio_.append(None)` by `append(0.0)` in this branch to fix a `TypeError` with `use_legacy_attributes=False`. That was right for L2 but not for L1, which is the case reported here.

## Describe the bug and its user-facing impact

With `l1_ratios=(1.0,)` (pure L1, which `LogisticRegressionCV` maps to `penalty="l1"`), `refit=True` gives `l1_ratio_ = 1.0` but `refit=False` gives `l1_ratio_ = 0.0`, with both values of `use_legacy_attributes`. The docstring says that with `refit=False`, `l1_ratio_` is "the average of the l1_ratio's that correspond to the best score for each fold", which is 1.0 here.

## Steps/Code to Reproduce

```python
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegressionCV

X, y = make_classification(n_samples=200, random_state=0)
for refit in [True, False]:
    clf = LogisticRegressionCV(
        l1_ratios=(1.0,), solver="saga", max_iter=1000, refit=refit,
        use_legacy_attributes=False,
    ).fit(X, y)
    print(f"refit={refit}: l1_ratios_={clf.l1_ratios_}, l1_ratio_={clf.l1_ratio_}, "
          f"n zero coefs={(clf.coef_ == 0).sum()}")
```

## Expected Results

`l1_ratio_=1.0` in both cases.

## Actual Results

```
refit=True: l1_ratios_=[1.], l1_ratio_=1.0, n zero coefs=19
refit=False: l1_ratios_=[1.], l1_ratio_=0.0, n zero coefs=10
```

(With `use_legacy_attributes=True`, the `refit=False` case gives `l1_ratio_=array([0.])`.)

## Root cause and possible fix

`sklearn/linear_model/_logistic.py:2912-2916` (commit 289c4d5cfa):

```python
if penalty == "elasticnet":
    best_indices_l1 = best_indices[:, 1]
    self.l1_ratio_.append(xp.mean(l1_ratios_[best_indices_l1]))
else:
    self.l1_ratio_.append(0.0)
```

`penalty` is `"l1"` when all `l1_ratios` equal 1 (line 2532), and the `else` branch hard-codes 0. `l1_ratios_` is always a list of numbers here (`[None]` only with the deprecated `l1_ratios=None`), so the `else` branch can use `l1_ratios_[0]` (or the same mean as the elastic-net branch, with `np.asarray(l1_ratios_)`), falling back to 0.0 only for `None`.

## Versions

```
sklearn: 1.10.dev0, commit 289c4d5cfa (upstream/main, pure-Python files overlaid on the local build: no compiled file differs in sklearn/linear_model)
python: 3.13.14, numpy: 2.5.3, scipy: 1.18.1, platform: linux
```
