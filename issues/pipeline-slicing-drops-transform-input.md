# Slicing a `Pipeline` resets `transform_input` to its default

```json
{
    "estimators": ["Pipeline"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 1,
    "difficulty": 1,
    "decision": 0,
    "decision_tags": []
}
```

## Classification

- **Priority (1)**: silent, but it needs metadata routing, a non-default `transform_input`, and fitting the sliced pipeline with that metadata. Slices are mostly used to call `transform` or `get_feature_names_out`, which don't use `transform_input`.
- **Difficulty (1)**: pass `transform_input=self.transform_input` in `__getitem__`, as already done for `memory` and `verbose`.
- **Decision (0)**: plain fix, same as #18429 did for the other parameters.

## Upstream search

Queries (`gh search issues --include-prs`): "Pipeline slicing transform_input", "Pipeline __getitem__ slice parameters memory verbose".

- #18429 (merged): "Preserve parameters when slicing a pipeline", added `memory` and `verbose` to the slice.
- #28901 (merged): added `transform_input`, but did not update `__getitem__`.
- #34263 (merged): changed the default of `transform_input` to `("X_val",)`.

No report of this.

## Describe the bug and its user-facing impact

`Pipeline.__getitem__` with a slice builds a new pipeline from `steps`, `memory` and `verbose`, so `transform_input` falls back to its default `("X_val",)`. The docstring says slicing "copies a slice of this Pipeline", and #18429 made it preserve the other constructor parameters.

With `transform_input=["X_test"]` (or `None`), fitting the slice passes the metadata to the steps untransformed (or transformed when the original did not), with no error or warning.

## Steps/Code to Reproduce

```python
import numpy as np
import sklearn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

sklearn.set_config(enable_metadata_routing=True)


class Recorder(TransformerMixin, BaseEstimator):
    """Records the `X_test` it receives in fit."""

    def fit(self, X, y=None, X_test=None):
        self.X_test_seen_ = X_test
        return self

    def transform(self, X):
        return X


steps = [
    ("add_100", FunctionTransformer(lambda X: X + 100)),
    ("rec", Recorder().set_fit_request(X_test=True)),
    ("last", FunctionTransformer()),
]
pipe = Pipeline(steps, transform_input=["X_test"])
sub = pipe[:2]
print("transform_input:", pipe.transform_input, "->", sub.transform_input)

X, X_test = np.zeros((3, 1)), np.zeros((2, 1))
pipe.fit(X, X_test=X_test)
print("full pipeline, X_test seen:", pipe["rec"].X_test_seen_.ravel())
sub.fit(X, X_test=X_test)
print("sliced pipeline, X_test seen:", sub["rec"].X_test_seen_.ravel())
```

## Expected Results

The slice keeps `transform_input=['X_test']`, and `Recorder` sees `[100. 100.]` in both cases.

## Actual Results

```
transform_input: ['X_test'] -> ('X_val',)
full pipeline, X_test seen: [100. 100.]
sliced pipeline, X_test seen: [0. 0.]
```

## Root cause and possible fix

`sklearn/pipeline.py:372-374` (commit below):

```python
return self.__class__(
    self.steps[ind], memory=self.memory, verbose=self.verbose
)
```

Add `transform_input=self.transform_input`. A test that compares `get_params(deep=False)` of the slice and the original (except `steps`) would catch future parameters.

## Versions

```
sklearn: 1.10.dev0, commit 289c4d5cfa
python: 3.13.14, numpy: 2.5.3, scipy: 1.18.1, platform: linux
```
