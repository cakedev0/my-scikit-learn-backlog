# SelfTrainingClassifier fails with routed sample_weight when y has unlabeled samples

```json
{
    "estimators": ["SelfTrainingClassifier"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 2,
    "difficulty": 1,
    "decision": 0,
    "decision_tags": []
}
```

## Classification

- **Priority (2)**: routing `sample_weight` (or any sample-aligned metadata) to the base estimator of `SelfTrainingClassifier` is the documented feature since 1.6, and it can't work in the normal case where `y` has unlabeled samples. The failure is a loud error, and metadata routing is opt-in.
- **Difficulty (1)**: subset the routed fit params with `_check_method_params(X, params, indices=...)` at the two `estimator_.fit` calls, as `OneVsOneClassifier` or `CalibratedClassifierCV` do.
- **Decision (0)**: no trade-off.

## Upstream search

- `SelfTrainingClassifier metadata routing`: #28494 (merged, added routing to `SelfTrainingClassifier`; review discussed the `fit`/`score` mapping but not subsetting the metadata), #26797 (closed, "Allow Self Training classifier fit to accept kwargs").
- `SelfTrainingClassifier sample_weight`: no match.

See also `impute-iterative-imputer-routed-params-not-subset.md`, the same pattern in another meta-estimator.

## Describe the bug and its user-facing impact

`SelfTrainingClassifier.fit(X, y, sample_weight=sw)` routes `sw` (shape `(n_samples,)`) unchanged to `estimator_.fit(X[has_label], y[has_label], sample_weight=sw)`, where only the labeled samples are passed. As soon as `y` contains an unlabeled sample (the whole point of the estimator), the base estimator raises a shape mismatch error. The metadata routing tests only use a fully labeled `y`, so the subset is the whole dataset there.

A second, smaller problem in the same router: `get_metadata_routing` declares `.add(callee="score", caller="fit")`, but `fit` never calls `estimator.score`. Users must therefore also call `set_score_request` on the base estimator before `fit`, otherwise `fit` raises `UnsetMetadataPassedError` about `LogisticRegression.score`.

## Steps/Code to Reproduce

```python
import numpy as np
import sklearn
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.semi_supervised import SelfTrainingClassifier

sklearn.set_config(enable_metadata_routing=True)
X, y = make_classification(n_samples=100, random_state=0)
y[50:] = -1  # half of the samples are unlabeled

est = LogisticRegression().set_fit_request(sample_weight=True)
est.set_score_request(sample_weight=False)  # needed, although fit never calls score
SelfTrainingClassifier(est).fit(X, y, sample_weight=np.ones(100))
```

## Expected Results

The base estimator is fitted with the weights of the samples it receives (labeled ones, then pseudo-labeled ones), and `fit` works.

## Actual Results

```
Traceback (most recent call last):
  ...
  File "/home/arthur/scikit-learn/sklearn/linear_model/_logistic.py", line 1508, in fit
    sample_weight = _check_sample_weight(sample_weight, X, dtype=X.dtype)
  File "/home/arthur/scikit-learn/sklearn/utils/validation.py", line 2186, in _check_sample_weight
    raise ValueError(
    ...<3 lines>...
    )
ValueError: sample_weight.shape == (100,), expected (50,)!
```

Without the `set_score_request` line:

```
sklearn.exceptions.UnsetMetadataPassedError: [sample_weight] are passed but are not explicitly set as requested or not requested for LogisticRegression.score, which is used within SelfTrainingClassifier.fit. Call `LogisticRegression.set_score_request({metadata}=True/False)` for each metadata you want to request/ignore. See the Metadata Routing User guide <https://scikit-learn.org/stable/metadata_routing.html> for more information.
```

## Root cause and possible fix

`sklearn/semi_supervised/_self_training.py`, in `fit`: both calls `self.estimator_.fit(X[safe_mask(X, has_label)], self.transduction_[has_label], **routed_params.estimator.fit)` pass the full-length metadata. Fix: `_check_method_params(X, params=routed_params.estimator.fit, indices=np.flatnonzero(has_label))` before each call. And remove `.add(callee="score", caller="fit")` from `get_metadata_routing`. A routing test with a partially unlabeled `y` would cover it.

## Versions

```
sklearn: 1.10.dev0, commit f570497251 (no diff in sklearn/semi_supervised vs upstream/main 289c4d5cfa)
python: 3.13, numpy: 2.5.3, scipy: 1.18.1, platform: linux
```
