# Random projections with the default `n_components="auto"` crash in `get_feature_names_out` and `set_output(transform="pandas")`

```json
{
    "estimators": ["GaussianRandomProjection", "SparseRandomProjection"],
    "existing_issue": "https://github.com/scikit-learn/scikit-learn/issues/34993",
    "reproduced": true,
    "priority": 2,
    "difficulty": 1,
    "decision": 0,
    "decision_tags": []
}
```

## Classification

- **Priority (2)**: a crash with the default parameters as soon as feature names are needed (pandas/polars output, `ColumnTransformer` with `verbose_feature_names_out`, pipelines inspecting names). Loud and easy to work around by passing an integer `n_components`, but the error message does not point to the cause.
- **Difficulty (1)**: one line, `self._n_features_out = self.n_components_`.
- **Decision (0)**: no trade-off.

## Upstream search

Queries: "random projection get_feature_names_out", "GaussianRandomProjection set_output", "GaussianRandomProjection get_feature_names_out".

- #34993 (open, needs triage): this exact bug, same root cause and one-line fix.
- #34994 (closed PR, not merged): the fix plus tests, closed automatically because #34993 still has the "Needs Triage" label, not rejected. The author offers to reopen it once the issue is triaged.

## Describe the bug and its user-facing impact

`BaseRandomProjection.fit` stores `self._n_features_out = self.n_components`, i.e. the hyper-parameter, not the fitted `n_components_`. With the default `n_components="auto"`, `_n_features_out` is the string `"auto"`, and `get_feature_names_out` fails with `TypeError: 'str' object cannot be interpreted as an integer`. Since `set_output(transform="pandas")` calls `get_feature_names_out`, pandas/polars output is broken too.

`test_random_projection_feature_names_out` only uses `n_components=2`, and the common tests also set an integer `n_components`, so the default is never exercised.

## Steps/Code to Reproduce

```python
import numpy as np
from sklearn.random_projection import GaussianRandomProjection

X = np.random.RandomState(0).randn(50, 2000)
rp = GaussianRandomProjection(eps=0.5).fit(X)  # n_components="auto" (default)
print("n_components_:", rp.n_components_)
rp.set_output(transform="pandas").fit_transform(X)
```

## Expected Results

A DataFrame with columns `gaussianrandomprojection0`, ..., `gaussianrandomprojection{n_components_ - 1}`.

## Actual Results

```
n_components_: 187
Traceback (most recent call last):
  File "rp_names.py", line 7, in <module>
    rp.set_output(transform="pandas").fit_transform(X)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^
  File "/home/arthur/scikit-learn/sklearn/utils/_set_output.py", line 321, in wrapped
    data_to_wrap = f(self, X, *args, **kwargs)
  File "/home/arthur/scikit-learn/sklearn/base.py", line 969, in fit_transform
    return self.fit(X, **fit_params).transform(X)
           ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^
  File "/home/arthur/scikit-learn/sklearn/utils/_set_output.py", line 334, in wrapped
    return _wrap_data_with_container(method, data_to_wrap, X, self)
  File "/home/arthur/scikit-learn/sklearn/utils/_set_output.py", line 303, in _wrap_data_with_container
    columns = estimator.get_feature_names_out()
  File "/home/arthur/scikit-learn/sklearn/base.py", line 1064, in get_feature_names_out
    return _generate_get_feature_names_out(
        self, self._n_features_out, input_features=input_features
    )
  File "/home/arthur/scikit-learn/sklearn/utils/validation.py", line 2463, in _generate_get_feature_names_out
    [f"{estimator_name}{i}" for i in range(n_features_out)], dtype=object
                                     ~~~~~^^^^^^^^^^^^^^^^
TypeError: 'str' object cannot be interpreted as an integer
```

## Root cause and possible fix

`sklearn/random_projection.py:431`:

```python
self._n_features_out = self.n_components
```

should be `self._n_features_out = self.n_components_`. Same for both subclasses since the line is in `BaseRandomProjection.fit`. A regression test with `n_components="auto"` in `test_random_projection_feature_names_out` would cover it.

## Versions

```
sklearn: 1.10.dev0, commit 289c4d5cfa (upstream/main; ran on f570497251, no diff in the files involved)
python: 3.13.14, numpy: 2.5.3, scipy: 1.18.1, platform: linux
```
