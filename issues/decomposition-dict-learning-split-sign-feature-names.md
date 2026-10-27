# `split_sign=True` is ignored by `get_feature_names_out` of `SparseCoder`, `DictionaryLearning` and `MiniBatchDictionaryLearning`

```json
{
    "estimators": ["SparseCoder", "DictionaryLearning", "MiniBatchDictionaryLearning"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 1,
    "difficulty": 1,
    "decision": 0,
    "decision_tags": []
}
```

## Classification

- **Priority (1)**: `split_sign=True` is a niche option, but with it `set_output(transform="pandas"/"polars")` always crashes and `get_feature_names_out` returns half of the names.
- **Difficulty (1)**: double `_n_features_out` when `split_sign=True`, in the shared base class.
- **Decision (0)**: names should match the output columns.

## Upstream search

Queries (issues and PRs): "split_sign", "SparseCoder get_feature_names_out", "DictionaryLearning fit_transform transform different".

- #34768 (closed as not planned): reported this bug together with two other `split_sign` symptoms (`DictionaryLearning.fit_transform` does not apply the split, and the split code is always float64). It was closed by @lesteve because the reporter was blocked under the automated contributions policy, not on the merits: "If you are a real user with a real use case, feel free to open a separate issue." Nobody disputed the bug.
- #34774 (closed): fix PR for #34768, auto-closed because the issue was not triaged.

I'm filing it again because the closure was about the account, and the crash is still there on main.

## Describe the bug and its user-facing impact

With `split_sign=True`, `transform` returns `2 * n_components` columns (positive parts, then negative parts), but `_n_features_out` returns `n_components`. So `get_feature_names_out()` returns `n_components` names and `set_output(transform="pandas")` (or polars, or any `Pipeline`/`ColumnTransformer` with pandas output) raises a length mismatch error.

Related symptoms of the same parameter, from #34768 (separate code paths, listed for whoever fixes this):

- `DictionaryLearning.fit_transform` returns the `(n_samples, n_components)` code from fitting and ignores `split_sign`, while `transform` on the same data returns `2 * n_components` columns;
- the split code is allocated with `np.empty((n_samples, 2 * n_features))`, so float32 input gives float64 output.

## Steps/Code to Reproduce

```python
import numpy as np
from sklearn.decomposition import MiniBatchDictionaryLearning

X = np.random.RandomState(0).randn(30, 6)
dl = MiniBatchDictionaryLearning(n_components=4, split_sign=True, random_state=0).fit(X)
print("transform(X).shape[1]:", dl.transform(X).shape[1])
print("get_feature_names_out():", dl.get_feature_names_out())

dl.set_output(transform="pandas").transform(X)
```

## Expected Results

8 feature names and a pandas DataFrame with 8 columns.

## Actual Results

```
transform(X).shape[1]: 8
get_feature_names_out(): ['minibatchdictionarylearning0' 'minibatchdictionarylearning1'
 'minibatchdictionarylearning2' 'minibatchdictionarylearning3']
Traceback (most recent call last):
  ...
ValueError: Length mismatch: Expected axis has 8 elements, new values have 4 elements
```

Same with `DictionaryLearning` and `SparseCoder(dictionary=..., split_sign=True)`.

## Root cause and possible fix

`_n_features_out` at `sklearn/decomposition/_dict_learning.py:1388` (`SparseCoder`), `:1712` (`DictionaryLearning`) and `:2288` (`MiniBatchDictionaryLearning`) return the number of atoms, while `_BaseSparseCoding._transform` (`:1095-1101`) doubles the width when `split_sign=True`. Multiplying by `2 if self.split_sign else 1` in the three properties (or once in `_BaseSparseCoding`) fixes the names; `_inverse_transform` (`:1131-1133`) already accounts for the doubling.

## Versions

```
sklearn: 1.10.dev0, commit f570497251 (_dict_learning.py identical to upstream/main 289c4d5cfa)
python: 3.13.14, numpy: 2.5.3, scipy: 1.18.1, pandas: 3.0.6, platform: linux
```
