# `FeatureUnion` has no `n_features_in_` / `feature_names_in_` when its first transformer is `"drop"`

```json
{
    "estimators": ["FeatureUnion", "Pipeline"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 1,
    "difficulty": 1,
    "decision": 0,
    "decision_tags": []
}
```

## Classification

- **Priority (1)**: a loud `AttributeError` (or a silently missing attribute for `hasattr` checks), only when the first entry is dropped, which happens naturally when a grid search sets a step to `"drop"`. Fitting and transforming still work.
- **Difficulty (1)**: delegate to the first non-dropped transformer, as `_iter()` already skips them.
- **Decision (0)**: plain fix.

## Upstream search

Queries (`gh search issues --include-prs`): "FeatureUnion n_features_in_ drop", "FeatureUnion feature_names_in_". Found #24754 / #25220 (closed / merged, added `feature_names_in_` by delegating to the first transformer) and #24058 (merged, `get_feature_names_out` with `"passthrough"`). None mention `"drop"`.

## Describe the bug and its user-facing impact

`FeatureUnion.n_features_in_` and `feature_names_in_` are properties that return the attribute of `transformer_list[0][1]`. When that entry is the string `"drop"`, they raise `AttributeError: 'str' object has no attribute ...`, so `hasattr(union, "n_features_in_")` is `False` on a fitted union. This propagates to an enclosing `Pipeline`, whose `n_features_in_` delegates to its first step.

`"passthrough"` in first position is fine, since `fit` replaces it with a fitted `FunctionTransformer`.

## Steps/Code to Reproduce

```python
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.pipeline import FeatureUnion, make_pipeline
from sklearn.linear_model import LinearRegression

rng = np.random.RandomState(0)
X = pd.DataFrame(rng.rand(10, 3), columns=["a", "b", "c"])
y = rng.rand(10)

union = FeatureUnion([("pca1", PCA(1)), ("pca2", PCA(2))])
print(union.fit(X).n_features_in_, union.feature_names_in_)

union.set_params(pca1="drop")  # e.g. from a grid search over steps
union.fit(X)
print(hasattr(union, "n_features_in_"), hasattr(union, "feature_names_in_"))
pipe = make_pipeline(union, LinearRegression()).fit(X, y)
pipe.n_features_in_
```

## Expected Results

`3 ['a' 'b' 'c']`, then `True True`, then `pipe.n_features_in_` returns 3.

## Actual Results

```
3 ['a' 'b' 'c']
False False
Traceback (most recent call last):
  File "fu_drop.py", line 18, in <module>
    pipe.n_features_in_
  File "/home/arthur/scikit-learn/sklearn/pipeline.py", line 1294, in n_features_in_
    return self.steps[0][1].n_features_in_
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/arthur/scikit-learn/sklearn/pipeline.py", line 2140, in n_features_in_
    return self.transformer_list[0][1].n_features_in_
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AttributeError: 'str' object has no attribute 'n_features_in_'
```

## Root cause and possible fix

`sklearn/pipeline.py:2136-2146` (commit below): both properties use `self.transformer_list[0][1]`. Use the first transformer yielded by `self._iter()` instead (it skips `"drop"`), e.g. `next(trans for _, trans, _ in self._iter())`. If all transformers are dropped, raising `AttributeError` is fine.

## Versions

```
sklearn: 1.10.dev0, commit 289c4d5cfa
python: 3.13.14, numpy: 2.5.3, scipy: 1.18.1, platform: linux
```
