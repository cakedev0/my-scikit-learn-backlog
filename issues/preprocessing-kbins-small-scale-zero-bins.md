# `KBinsDiscretizer` drops all bins of small-scale features (absolute 1e-8 width threshold)

```json
{
    "estimators": ["KBinsDiscretizer"],
    "existing_issue": "https://github.com/scikit-learn/scikit-learn/issues/19416",
    "reproduced": true,
    "priority": 1,
    "difficulty": 3,
    "decision": 2,
    "decision_tags": ["output-change", "unclear-spec"]
}
```

## Classification

- **Priority (1)**: a non-constant feature whose whole range is below ~1e-8 is an exotic input (features are rarely left at that scale, and p-values or probabilities span far more), and the symptoms are a loud `IndexError` with the default `encode="onehot"` or a warning with `"ordinal"`; rescaling first is a cheap workaround. (priority review, 2026-10-23: was 2)
- **Difficulty (3)**: the same edge-removal block also causes #19416 (ties) and #25594 (2 distinct values), so a fix should handle the three together. In #25594, a contributor who tried appending infs and reindexing bins reported that "almost any change will break a lot of stuff: discretization, encoding, calibration". (triage of #35300, 2026-11-09: was 2)
- **Decision (2)**: `output-change`: features with a range around 1e-8 to 1e-6 can keep bins that are removed today. `unclear-spec`: three incompatible fixes were proposed upstream and none was chosen, so both issues have been "Needs Decision" since 2021 and 2023. (triage of #35300, 2026-11-09: was 1)

## Upstream search

Queries: "KBinsDiscretizer bins width too small", "KBinsDiscretizer 1e-8", "KBinsDiscretizer small values".

Already reported in #19416 (open, "API", "Needs Decision"). The issue is about ties, but its author posted the small-scale trigger in [a comment](https://github.com/scikit-learn/scikit-learn/issues/19416#issuecomment-780141212) (`uniform(-1e-8, 1e-8)` plus zeros, all bins removed) and asked for the `1e-8` to be a parameter. That example was constructed to make a point; the use case in the thread is the ties one. #19433 was closed as a duplicate of #19416.

#35300 (2026-11-02, found by reading the code, no use case) reported the `n_bins_ == 0` state and the `IndexError`, and was closed as a duplicate of #19416. Its PR #35324 (treat the feature as constant when every bin is removed) was closed by the bot.

Fixes proposed upstream, not compatible with each other:

- a tolerance relative to the feature range (#19416, and the fix below);
- a tolerance from the dtype resolution, `np.finfo(X.dtype).resolution` (#19416);
- replacing the first and last edges with `-inf`/`+inf` before removing duplicates (#25594), which can give 3 bins for a binary feature.

Related but different (same edge-removal code, other trigger):

- #25594 (open, "Bug", "Needs Decision"): with only 2 distinct values and `n_bins=3`, a single bin is created because the first/last edges are compared before being replaced by `-inf`/`+inf`. Maintainers agreed it's a bug in 2023, but no PR followed.
- #13165 (merged): introduced the `1e-8` threshold.

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
