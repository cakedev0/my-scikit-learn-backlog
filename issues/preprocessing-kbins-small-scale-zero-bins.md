# `KBinsDiscretizer` drops all bins of small-scale features (absolute 1e-8 width threshold)

```json
{
    "estimators": ["KBinsDiscretizer"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 1,
    "difficulty": 2,
    "decision": 1,
    "decision_tags": ["output-change"]
}
```

## Classification

- **Priority (1)**: a non-constant feature whose whole range is below ~1e-8 is an exotic input (features are rarely left at that scale, and p-values or probabilities span far more), and the symptoms are a loud `IndexError` with the default `encode="onehot"` or a warning with `"ordinal"`; rescaling first is a cheap workaround. (priority review, 2026-10-23: was 2)
- **Difficulty (2)**: the threshold has to become relative to the feature scale; choosing the relative tolerance needs a bit of care.
- **Decision (1)**: `output-change`: features with a range around 1e-8 to 1e-6 can keep bins that are removed today.

## Upstream search

Queries: "KBinsDiscretizer bins width too small", "KBinsDiscretizer 1e-8", "KBinsDiscretizer small values".

Related but different (same edge-removal code, other trigger):

- #19416 (open, "Needs Decision") and #19433 (closed as duplicate of it): bins collapse when many samples share the same value (ties), on data of normal scale.
- #25594 (open, "Needs Decision"): with only 2 distinct values and `n_bins=3`, a single bin is created because the first/last edges are compared before being replaced by `-inf`/`+inf`.
- #13165 (merged): introduced the `1e-8` threshold.

None of them covers the scale dependence of the absolute threshold, nor the `n_bins_ == 0` state and the `IndexError` with `encode="onehot"`.

## Describe the bug and its user-facing impact

For `strategy="quantile"` and `strategy="kmeans"`, `fit` removes bins whose width is `<= 1e-8`. The threshold is absolute, so on a feature whose values span less than ~1e-8, every bin is removed: `n_bins_` becomes `0` and `bin_edges_` keeps a single edge.

- with `encode="ordinal"`, every sample is mapped to bin 0 (with a warning about small bins);
- with the default `encode="onehot"`, `fit` raises `IndexError` from the internal `OneHotEncoder`.

`strategy="uniform"` does not have the threshold and works on the same data, so the result depends on the unit of the feature, which a discretizer should not.

## Steps/Code to Reproduce

```python
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer

rng = np.random.RandomState(0)
X = rng.rand(100, 1) * 1e-9  # e.g. a quantity measured in meters

for strategy in ["uniform", "quantile", "kmeans"]:
    kbd = KBinsDiscretizer(n_bins=4, encode="ordinal", strategy=strategy)
    Xt = kbd.fit_transform(X)
    print(f"{strategy:8}: n_bins_={kbd.n_bins_}, bins used: {np.unique(Xt)}")

KBinsDiscretizer(n_bins=4, strategy="quantile").fit(X)  # default encode="onehot"
```

## Expected Results

4 bins for all strategies (same result as `X * 1e9`), and no error.

## Actual Results

```
UserWarning: Bins whose width are too small (i.e., <= 1e-8) in feature 0 are removed. Consider decreasing the number of bins.
uniform : n_bins_=[4], bins used: [0. 1. 2. 3.]
quantile: n_bins_=[0], bins used: [0.]
kmeans  : n_bins_=[0], bins used: [0.]
Traceback (most recent call last):
  File ".../B.py", line 12, in <module>
    KBinsDiscretizer(n_bins=4, strategy="quantile").fit(X)  # default encode="onehot"
  File "/home/arthur/scikit-learn/sklearn/preprocessing/_discretization.py", line 397, in fit
    self._encoder.fit(np.zeros((1, len(self.n_bins_))))
  File "/home/arthur/scikit-learn/sklearn/preprocessing/_encoders.py", line 157, in _fit
    stop_idx = -1 if np.isnan(sorted_cats[-1]) else None
                              ~~~~~~~~~~~^^^^
IndexError: index -1 is out of bounds for axis 0 with size 0
```

## Root cause and possible fix

`sklearn/preprocessing/_discretization.py:376`:

```python
mask = np.ediff1d(bin_edges[jj], to_begin=np.inf) > 1e-8
```

The `to_begin=np.inf` keeps the first edge, but every following edge is dropped when the gaps are all `<= 1e-8`, which leaves one edge and `n_bins_[jj] = 0`. A constant feature is already handled earlier (`col_min == col_max`), so this only happens for non-constant features with a tiny range.

Fix: make the tolerance relative to the feature range, e.g. `> 1e-8 * (col_max - col_min)` (or a few ulps of `max(abs(col_min), abs(col_max))`), so that the behavior does not depend on the unit. The documented threshold in the `n_bins_` docstring would need updating. Small-scale features whose bins are narrower than 1e-8 (e.g. a range of 1e-6 with 100 bins) would get more bins than today (`output-change`), but those bins were removed for no good reason.

## Versions

```
sklearn: 1.10.dev0, commit f570497251 (code identical on upstream/main 289c4d5cfa)
python: 3.13.14, numpy: 2.5.3, scipy: 1.18.1, platform: linux
```
