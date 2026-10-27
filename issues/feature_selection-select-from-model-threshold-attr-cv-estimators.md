# SelectFromModel.threshold_ reports the wrong threshold for ElasticNetCV / LogisticRegressionCV with an L1 penalty

```json
{
    "estimators": ["SelectFromModel"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 1,
    "difficulty": 1,
    "decision": 0,
    "decision_tags": []
}
```

## Classification

- **Priority (1)**: the selection itself is right; only the reported `threshold_` attribute is wrong, so users inspecting it (or reproducing the selection with it) get a different set of features. Needs `threshold=None` (default) with `ElasticNetCV(l1_ratio=1)` or `LogisticRegressionCV(l1_ratios=[1])`.
- **Difficulty (1)**: pass `self.estimator_` instead of `self.estimator` in the property.
- **Decision (0)**: no trade-off.

## Upstream search

- `gh issue list --search "SelectFromModel threshold_"`: no issue.
- `gh issue list --search "SelectFromModel ElasticNetCV threshold"`: #30936 (closed) is a crash with multiple `l1_ratio`, fixed by #31107, which added the `ElasticNetCV` branch based on the fitted `l1_ratio_`. It did not update `threshold_`.

## Describe the bug and its user-facing impact

With `threshold=None`, `_calculate_threshold` picks `1e-5` for L1-penalized models and `"mean"` otherwise. For `ElasticNetCV` and `LogisticRegressionCV`, the L1 check reads the **fitted** attribute `l1_ratio_`. `get_support()` calls `_calculate_threshold` with the fitted `estimator_`, so the selection uses `1e-5`. But the `threshold_` property calls it with the unfitted `self.estimator`, which has no `l1_ratio_`, so `threshold_` returns the mean importance. The documented attribute ("The threshold value used for feature selection") doesn't match the threshold that was used.

## Steps/Code to Reproduce

```python
import numpy as np
from sklearn.datasets import make_regression
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import ElasticNetCV

X, y = make_regression(n_samples=100, n_features=20, n_informative=10, random_state=0)
sel = SelectFromModel(ElasticNetCV(l1_ratio=1.0)).fit(X, y)
importances = np.abs(sel.estimator_.coef_)
print("threshold_:", sel.threshold_)
print("features selected by get_support():", sel.get_support().sum())
print("features with importance >= threshold_:", (importances >= sel.threshold_).sum())
```

## Expected Results

`threshold_` is `1e-5`, and both counts are equal.

## Actual Results

```
threshold_: 16.69671708075587
features selected by get_support(): 10
features with importance >= threshold_: 6
```

Same with `SelectFromModel(LogisticRegressionCV(l1_ratios=[1.0], solver="saga", Cs=3, max_iter=2000))` on `make_classification(n_samples=200, n_features=20, random_state=0)`: `threshold_` is 0.40, 16 features are selected, but only 6 have an importance >= `threshold_` (checked).

## Root cause and possible fix

`sklearn/feature_selection/_from_model.py:396`: `return _calculate_threshold(self.estimator, scores, self.threshold)`, while `_get_support_mask` (`sklearn/feature_selection/_from_model.py:300`) passes the fitted estimator. The fitted-attribute checks are at `sklearn/feature_selection/_from_model.py:40` (`ElasticNetCV.l1_ratio_`) and `:46` (`LogisticRegressionCV.l1_ratio_`). Use `self.estimator_` in `threshold_`.

## Versions

```
sklearn: 1.10.dev0, commit 289c4d5cfa (upstream/main; reproduced on local checkout f570497251, no diff in sklearn/feature_selection/_from_model.py)
python: 3.13, numpy: 2.5.3, scipy: 1.18.1, platform: linux
```
