# Pipeline `memory` cache ignores the global `transform_output` config and returns stale numpy/pandas outputs

```json
{
    "estimators": ["Pipeline", "make_pipeline"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 1,
    "difficulty": 2,
    "decision": 1,
    "decision_tags": ["perf-low"]
}
```

## Classification

- **Priority (1)**: needs `memory` (a minority of pipelines) and a change of the global `transform_output` config between two fits sharing the same cache. The column-selection case is a loud error, the other one only loses the feature names (with a warning at `predict`), no wrong predictions. Clearing the cache is a workaround. (priority review, 2026-10-23: was 2)
- **Difficulty (2)**: the cache key must include the config that changes transformer outputs; a few lines, but one has to decide which config keys to hash.
- **Decision (1)**: adding config to the key invalidates caches when unrelated config changes if the whole config is hashed (`perf-low`); hashing only output-affecting keys (`transform_output`, `array_api_dispatch`, ...) avoids that but needs a list to maintain.

## Upstream search

- `gh issue list --search "Pipeline memory set_output"`, `"memory cache transform_output"`, `"Pipeline memory config_context"`: no match. #30310 (open) is about set_output with sparse OneHotEncoder output in ColumnTransformer, unrelated to caching.
- `gh pr list` with the same queries: nothing related.

## Describe the bug and its user-facing impact

`Pipeline._fit` caches `_fit_transform_one_with_callbacks(transformer, X, y, ...)` with joblib. The key is built from the arguments only. The output container of each transformer also depends on the global `transform_output` config (via the `set_output` wrapper), which is not part of the key. After a fit with the default config, a fit under `transform_output="pandas"` hits the cache and gets numpy arrays back for the cached steps (and vice versa).

Consequences seen:

- a later step that selects DataFrame columns by name (e.g. a `ColumnTransformer` with string columns) raises "Specifying the columns using strings is only supported for dataframes." even under pandas output, and keeps raising until the cache is cleared;
- the final estimator is fitted on a numpy array, so it has no `feature_names_in_` and `predict` warns "X has feature names, but LinearRegression was fitted without feature names".

The same pipeline without `memory` behaves correctly.

## Steps/Code to Reproduce

```python
import tempfile

import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X = pd.DataFrame({"a": [1.0, 2, 3, 4], "b": [2.0, 1, 4, 3]})
y = np.arange(4.0)

with tempfile.TemporaryDirectory() as cache_dir:
    for memory in [None, cache_dir]:
        pipe = make_pipeline(
            StandardScaler(),
            ColumnTransformer([("keep_a", "passthrough", ["a"])]),
            LinearRegression(),
            memory=memory,
        )
        try:
            pipe.fit(X, y)  # fails: StandardScaler outputs numpy, columns selected by name
        except ValueError as e:
            print(f"memory={memory is not None}, default output: {e}")
        with sklearn.config_context(transform_output="pandas"):
            try:
                pipe.fit(X, y)
                print(f"memory={memory is not None}, pandas output: ok")
            except ValueError as e:
                print(f"memory={memory is not None}, pandas output: {e}")
```

## Expected Results

With pandas output enabled, the second fit succeeds with and without `memory`.

## Actual Results

```
memory=False, default output: Specifying the columns using strings is only supported for dataframes.
memory=False, pandas output: ok
memory=True, default output: Specifying the columns using strings is only supported for dataframes.
memory=True, pandas output: Specifying the columns using strings is only supported for dataframes.
```

With `make_pipeline(StandardScaler(), LinearRegression(), memory=cache_dir)` fitted once with the default config and once under `transform_output="pandas"`, the final step has `feature_names_in_` = `None` (vs `['a' 'b']` without memory) and `predict` warns "X has feature names, but LinearRegression was fitted without feature names" (checked).

## Root cause and possible fix

`sklearn/pipeline.py:532-538`:

```python
fit_transform_one_cached = memory.cache(
    _fit_transform_one_with_callbacks,
    ignore=["caller", "callback_ctx"],
)
```

The cached function's output depends on `sklearn.get_config()["transform_output"]` (and possibly `array_api_dispatch`), which is not an argument. Fix: pass the relevant config values as an extra (unused) argument of the cached function so they are part of the hash, e.g. `output_config=get_config()["transform_output"]`.

## Versions

```
sklearn: 1.10.dev0, commit 289c4d5cfa (upstream/main; reproduced on local checkout f570497251, no diff in sklearn/pipeline.py)
python: 3.13, numpy: 2.5.3, scipy: 1.18.1, platform: linux
```
