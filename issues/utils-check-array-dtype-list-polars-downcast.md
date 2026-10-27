# `check_array` with a dtype list ignores the dtype of polars/pyarrow inputs, so `TSNE` downcasts float64 polars data to float32

```json
{
    "estimators": ["TSNE", "check_array", "StandardScaler", "PCA", "KMeans"],
    "existing_issue": "https://github.com/scikit-learn/scikit-learn/issues/33294",
    "reproduced": true,
    "priority": 2,
    "difficulty": 2,
    "decision": 1,
    "decision_tags": ["output-change"]
}
```

## Classification

- **Priority (2)**: polars is a supported input type and the effect is silent. For most estimators it "only" upcasts float32 polars data to float64 (2x memory, float32 output not preserved, ~50 estimators in my sweep). For `TSNE`, the only estimator listing float32 first, float64 data is downcast to float32 and the embedding differs from the numpy/pandas one.
- **Difficulty (2)**: compute `dtype_orig` for non-pandas dataframes from the narwhals schema; the open question in #33294 is how to map nullable/extension dtypes.
- **Decision (1)**: changes `TSNE` results for polars/pyarrow input (back to what numpy input gives) and output dtypes for float32 polars input.

## Upstream search

Queries (issues and PRs): "polars float32", "check_array polars dtype float32", "pyarrow float32", "TSNE polars".

- #33294 (open, "Bug", "Needs Investigation"): same root cause, reported as `check_X_y(X_polars_f32, dtype=[np.float64, np.float32])` returning float64. A comment there already traces it to `dtype_orig` being `None` for every non-pandas container, and lists polars Series and pyarrow tables. The discussion only covers the float32 -> float64 upcast (memory); it does not mention the lossy float64 -> float32 direction hit by `TSNE`, which is the new argument here.
- #33344 (closed): fix attempt for #33294, closed without merge.

## Describe the bug and its user-facing impact

When `check_array` gets a list of accepted dtypes, it keeps the input dtype if it is in the list and otherwise converts to the first entry. For pandas, the input dtype is computed from the column dtypes; for polars DataFrames, polars Series and pyarrow Tables it is always `None`, so the input is always converted to `dtype[0]`:

- `dtype=[np.float64, np.float32]` (most estimators): float32 polars data becomes float64. Found with a sweep comparing `transform`/`predict` output dtypes for float32 numpy, pandas and polars input: numpy and pandas give float32, polars gives float64 for `StandardScaler`, `PCA`, `KMeans`, `Ridge`, `Lasso`, `MinMaxScaler`, `PolynomialFeatures`, `SimpleImputer`, `Pipeline`, search CVs, and ~40 others.
- `dtype=[np.float32, np.float64]` (`TSNE`, `sklearn/manifold/_t_sne.py:853,860`): float64 polars data becomes float32. `TSNE` then computes the distances and affinities in float32, and the embedding for a polars frame is the embedding of the float32 numpy array, not of the float64 one.

## Steps/Code to Reproduce

```python
import numpy as np
import polars as pl
from sklearn.manifold import TSNE
from sklearn.utils import check_array

X = np.random.RandomState(0).rand(60, 5)

# float32 listed first: a float64 polars frame is downcast to float32
print(check_array(X, dtype=[np.float32, np.float64]).dtype)
print(check_array(pl.DataFrame(X), dtype=[np.float32, np.float64]).dtype)

# TSNE validates X with dtype=[np.float32, np.float64]
emb_np = TSNE(perplexity=5, random_state=0).fit_transform(X)
emb_pl = TSNE(perplexity=5, random_state=0).fit_transform(pl.DataFrame(X))
print("TSNE numpy vs polars, max abs diff:", np.abs(emb_np - emb_pl).max())
print("TSNE numpy vs float32 numpy, max abs diff:",
      np.abs(TSNE(perplexity=5, random_state=0).fit_transform(X.astype(np.float32)) - emb_pl).max())
```

## Expected Results

`float64` for both `check_array` calls (the input dtype is in the list), and the same `TSNE` embedding for the numpy array and the polars frame holding the same float64 values, as for pandas.

## Actual Results

```
float64
float32
TSNE numpy vs polars, max abs diff: 69.1767
TSNE numpy vs float32 numpy, max abs diff: 0.0
```

The polars embedding is exactly the one of `X.astype(np.float32)`.

## Root cause and possible fix

`sklearn/utils/validation.py:870-873` sets `dtype_orig = getattr(array, "dtype", None)`, which is `None` for polars/pyarrow frames (no `.dtype`) and polars Series (dtype without `.kind`). The block that computes `dtype_orig` from the column dtypes (`validation.py:890-925`) only runs when `array_df.implementation.is_pandas()`. Then `validation.py:953-960` falls back to `dtype = dtype[0]`.

Fix direction (as discussed in #33294): compute `dtype_orig` from the narwhals schema for every dataframe backend, e.g. via `np.result_type` of the numpy equivalents of the column dtypes, keeping the pandas special cases for nullable/extension dtypes. It changes `TSNE` results for polars/pyarrow input and output dtypes for float32 polars input (both towards the numpy behavior).

## Versions

```
sklearn: 1.10.dev0, commit f570497251 (check_array code identical to upstream/main 289c4d5cfa; the only diff in validation.py is a docstring)
python: 3.13.14, numpy: 2.5.3, scipy: 1.18.1, pandas: 3.0.6, polars: 1.44.2, platform: linux
```
