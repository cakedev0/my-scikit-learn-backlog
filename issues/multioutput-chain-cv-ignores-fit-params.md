# ClassifierChain and RegressorChain with `cv` drop `sample_weight` (and other fit params) in the cross-validated fits

```json
{
    "estimators": ["RegressorChain", "ClassifierChain"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 1,
    "difficulty": 1,
    "decision": 1,
    "decision_tags": ["output-change"]
}
```

## Classification

- **Priority (1)**: silent, but it needs a chain (minor estimators) with the non-default `cv` and fit params such as `sample_weight`. Only the out-of-fold chain features are affected, the final estimators do get the weights, so the effect on predictions is usually small. (priority review, 2026-10-23: was 2)
- **Difficulty (1)**: pass the routed fit params to `cross_val_predict` through its `params` argument.
- **Decision (1)**: results change for users who pass fit params together with `cv` (`output-change`), but towards what they asked for.

## Upstream search

- `gh issue list --search "RegressorChain cv sample_weight"`: no match (#15953 is about MultiOutputRegressor fit params support, closed).
- `gh issue list --search "ClassifierChain random_state"` / `"chain random_state base estimator"`: chain issues #13339, #29856, #26817, #21761, #32855, none about fit params in the CV step.
- `gh pr list --search "RegressorChain cv sample_weight"`: #35095 (open, auto-requesting `sample_weight` on routers), unrelated to this code path.

## Describe the bug and its user-facing impact

When `cv` is set, `_BaseChain.fit` replaces the true labels of the previous outputs by out-of-fold predictions from `cross_val_predict`. That call does not receive the fit params, so `sample_weight` (or any routed metadata, e.g. `eval_set`-like params) is silently ignored for those models, while the final estimators of the chain do get it. The chain features are therefore computed by models fitted on a different weighting than the one the user requested.

`RegressorChain` accepts fit params without metadata routing, so this happens with the plain `fit(X, Y, sample_weight=...)` call. `ClassifierChain` needs metadata routing enabled to accept `sample_weight`, and has the same problem then.

## Steps/Code to Reproduce

```python
import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import Ridge
from sklearn.multioutput import RegressorChain


class LoggingRidge(Ridge):
    def fit(self, X, y, sample_weight=None):
        print(f"fit on {X.shape[0]} samples, sample_weight passed: {sample_weight is not None}")
        return super().fit(X, y, sample_weight=sample_weight)


X, Y = make_regression(n_samples=100, n_targets=2, random_state=0)
sample_weight = np.random.RandomState(0).rand(100)
RegressorChain(LoggingRidge(), cv=2).fit(X, Y, sample_weight=sample_weight)
```

## Expected Results

All four fits receive the (subset of the) sample weights.

## Actual Results

```
fit on 100 samples, sample_weight passed: True
fit on 50 samples, sample_weight passed: False
fit on 50 samples, sample_weight passed: False
fit on 100 samples, sample_weight passed: True
```

The same happens with `ClassifierChain(LogisticRegression().set_fit_request(sample_weight=True), cv=2)` under `sklearn.set_config(enable_metadata_routing=True)` (checked).

## Root cause and possible fix

`sklearn/multioutput.py:781` (`_BaseChain.fit`):

```python
cv_result = cross_val_predict(
    self.estimator,
    X_aug[:, :col_idx],
    y=y,
    cv=self.cv,
    method=chain_method,
)
```

Fix: add `params=routed_params.estimator.fit`. `cross_val_predict` already slices sample-aligned params per fold. A related gap: there is no way to pass `groups` to the splitter, so `cv=GroupKFold()` can't be used; that would need a `splitter` entry in `get_metadata_routing`, which is a separate enhancement.

## Versions

```
sklearn: 1.10.dev0, commit 289c4d5cfa (upstream/main; reproduced on local checkout f570497251, no diff in the relevant code)
python: 3.13, numpy: 2.5.3, scipy: 1.18.1, platform: linux
```
