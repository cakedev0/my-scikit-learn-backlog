# `img_to_graph` gives wrapped-around gradients for unsigned integer images (e.g. `uint8`)

```json
{
    "estimators": ["img_to_graph"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 2,
    "difficulty": 1,
    "decision": 0,
    "decision_tags": []
}
```

## Classification

- **Priority (2)**: silent wrong edge weights for the most common image dtype (`uint8`, e.g. from `load_sample_image`, PIL or imageio). Half of the edges with a non-zero gradient get `256 - g` instead of `g`, so a spectral clustering built on the graph is meaningless. Users who rescale to float first (as the gallery examples do) are not affected.
- **Difficulty (1)**: compute the difference in a signed or float dtype.
- **Decision (0)**: no trade-off; the output dtype can stay `img.dtype` since `|a - b|` fits in it.

## Upstream search

Queries: "img_to_graph uint8", "img_to_graph gradient", "img_to_graph overflow unsigned", "img_to_graph dtype".

No match.

## Describe the bug and its user-facing impact

`img_to_graph` weights each edge with the absolute intensity difference between the two neighboring pixels. The difference is computed in the image dtype, so for unsigned integers `a - b` wraps around when `b > a`, and `np.abs` of the wrapped value is still wrong: a gradient of 2 becomes 254 for `uint8`.

## Steps/Code to Reproduce

```python
import numpy as np
from sklearn.feature_extraction.image import img_to_graph

img = np.array([[10, 12], [10, 10]], dtype=np.uint8)
print(img_to_graph(img, return_as=np.ndarray))
print(img_to_graph(img.astype(np.int64), return_as=np.ndarray))
```

## Expected Results

The same graph for both dtypes, with weight 2 between pixels 0 and 1, and between pixels 1 and 3.

## Actual Results

```
[[ 10 254   0   0]
 [254  12   0   2]
 [  0   0  10   0]
 [  0   2   0  10]]
[[10  2  0  0]
 [ 2 12  0  2]
 [ 0  0 10  0]
 [ 0  2  0 10]]
```

## Root cause and possible fix

`_compute_gradient_3d` (`sklearn/feature_extraction/image.py:56-70`) computes `np.abs(img[...] - img[...])` in `img.dtype`. Fix: compute the difference in a signed type, e.g. `np.abs(a.astype(np.int64) - b)` for integer images (or simply `np.abs(np.subtract(a, b, dtype=...))`), or use `np.maximum(a, b) - np.minimum(a, b)`, which never wraps and keeps the dtype.

## Versions

```
sklearn: 1.10.dev0, commit f570497251 (code identical on upstream/main 289c4d5cfa)
python: 3.13.14, numpy: 2.5.3, scipy: 1.18.1, platform: linux
```
