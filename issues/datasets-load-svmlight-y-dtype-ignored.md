# `load_svmlight_file(dtype=...)` does not apply `dtype` to `y`, contrary to its docstring

```json
{
    "estimators": ["load_svmlight_file", "load_svmlight_files"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 1,
    "difficulty": 1,
    "decision": 1,
    "decision_tags": ["unclear-spec"]
}
```

## Classification

- **Priority (1)**: docstring mismatch with a cheap workaround (`y.astype(...)`); it only costs memory or an extra cast for users who asked for float32.
- **Difficulty (1)**: either cast `labels` or reword the docstring.
- **Decision (1)**: `unclear-spec`: casting `y` to `dtype` would follow the docstring, but integer `dtype` would then truncate float labels (svmlight targets can be real-valued for regression), so fixing the docstring may be the better choice.

## Upstream search

Queries: "load_svmlight_file dtype labels". No match. The sentence was added in b3fbccca38 (2014, "FIX various mismatch between docstring and signature params"), not with a code change.

## Describe the bug and its user-facing impact

The `dtype` parameter of `load_svmlight_file` and `load_svmlight_files` is documented as "Data type of dataset to be loaded. This will be the data type of the output numpy arrays ``X`` and ``y``." Only `X` gets that dtype: `y` is always float64 (and multilabel `y` is a list of tuples of floats). Users who load with `dtype=np.float32` to save memory get a float64 target. The tests only check `X.dtype`.

## Steps/Code to Reproduce

```python
import io
import numpy as np
from sklearn.datasets import load_svmlight_file, load_svmlight_files

f = io.BytesIO(b"1 1:0.5 3:1.5\n2 2:2.5\n")
X, y = load_svmlight_file(f, dtype=np.float32)
print(X.dtype, y.dtype)

f.seek(0)
X, y = load_svmlight_files([f], dtype=np.int32)
print(X.dtype, y.dtype)
```

## Expected Results

Per the docstring:

```
float32 float32
int32 int32
```

## Actual Results

```
float32 float64
int32 float64
```

## Root cause and possible fix

`sklearn/datasets/_svmlight_format_io.py:229` (upstream/main 289c4d5cfa) always reads labels as float64 (`np.frombuffer(labels, np.float64)`) and `dtype` is only applied to the data array. The docstrings at lines 117-118 and 305-306 promise more. I'd reword the docstring to "This will be the data type of the output `X`; `y` is always float64", unless maintainers prefer casting `y` for float dtypes only.

## Versions

```
sklearn: 1.10.dev0, commit 289c4d5cfa (upstream/main; local build f570497251, no diff on the files involved)
python: 3.13, numpy: 2.5.3, scipy: 1.18.1, platform: linux
```
