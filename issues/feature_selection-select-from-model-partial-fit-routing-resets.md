# SelectFromModel.partial_fit re-clones the estimator at every call when metadata routing is enabled

```json
{
    "estimators": ["SelectFromModel"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 2,
    "difficulty": 1,
    "decision": 0,
    "decision_tags": []
}
```

## Classification

- **Priority (2)**: silent wrong result: with `enable_metadata_routing=True`, only the last mini-batch is learned and the feature selection is based on it. Metadata routing is opt-in and partial_fit on SelectFromModel is not very common, which keeps it from 3.
- **Difficulty (1)**: delete one line.
- **Decision (0)**: no trade-off, the fix makes routing-on match routing-off.

## Upstream search

- `gh issue list --search "SelectFromModel partial_fit"`: #23277 (closed, param validation), #30527 (closed, routing inside a Pipeline), #34880 (open, prefit with pandas output); none about state being reset.
- `gh issue list --search "SelectFromModel metadata routing partial_fit clone"`: nothing relevant.
- `gh pr list --search "SelectFromModel partial_fit"`: #27490 (merged) introduced routing for SelectFromModel, and this line (`git blame`: 2af3fb8f6c6).

## Describe the bug and its user-facing impact

`SelectFromModel.partial_fit` is meant to update the same sub-estimator across calls. When metadata routing is enabled, the routed branch does `self.estimator_ = clone(self.estimator)` before calling `partial_fit`, on every call. Each call therefore starts from a fresh estimator and all earlier batches are forgotten. `get_support()` and `transform` silently use importances learned on the last batch only.

For a classifier, the second call usually raises "classes must be passed on the first call to partial_fit" unless `classes` is passed and requested every time, which hides the bug a bit; for regressors and when `classes` is always passed, nothing warns.

## Steps/Code to Reproduce

```python
import sklearn
from sklearn.datasets import make_regression
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import SGDRegressor

X, y = make_regression(n_samples=150, n_features=5, random_state=0)
for routing in [False, True]:
    sklearn.set_config(enable_metadata_routing=routing)
    sel = SelectFromModel(SGDRegressor(random_state=0))
    for i in range(3):
        sel.partial_fit(X[i * 50 : (i + 1) * 50], y[i * 50 : (i + 1) * 50])
    print(f"routing={routing}: samples seen by estimator_: {sel.estimator_.t_ - 1:.0f}")
```

## Expected Results

The estimator sees all 150 samples in both cases.

## Actual Results

```
routing=False: samples seen by estimator_: 150
routing=True: samples seen by estimator_: 50
```

## Root cause and possible fix

`sklearn/feature_selection/_from_model.py:454-457`:

```python
if first_call:
    self.estimator_ = clone(self.estimator)
if _routing_enabled():
    routed_params = process_routing(self, "partial_fit", **partial_fit_params)
    self.estimator_ = clone(self.estimator)  # <- resets the state at every call
    self.estimator_.partial_fit(X, y, **routed_params.estimator.partial_fit)
```

Remove the second `clone`. The existing `test_partial_fit` only runs with routing disabled.

## Versions

```
sklearn: 1.10.dev0, commit 289c4d5cfa (upstream/main; reproduced on local checkout f570497251, no diff in sklearn/feature_selection/_from_model.py)
python: 3.13, numpy: 2.5.3, scipy: 1.18.1, platform: linux
```
