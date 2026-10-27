# `SplineTransformer(knots="quantile")`: `sample_weight=np.ones(n)` gives different knots than no weights

```json
{
    "estimators": ["SplineTransformer"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 2,
    "difficulty": 1,
    "decision": 1,
    "decision_tags": ["output-change", "unclear-spec"]
}
```

## Classification

- **Priority (2)**: silent inconsistency: passing unit weights (or integer weights) changes the knot positions. The effect is moderate (knots move within the gap between neighboring samples) and only with `knots="quantile"`, which is not the default.
- **Difficulty (1)**: pick one quantile definition and use it in both branches of `_get_base_knot_positions`.
- **Decision (1)**: `output-change` for one of the two paths whichever definition is chosen; `unclear-spec`: which quantile definition to keep (`KBinsDiscretizer` went through a deprecation cycle for the same choice and now defaults to `"averaged_inverted_cdf"`).

## Upstream search

Queries: "SplineTransformer quantile sample_weight", "SplineTransformer sample_weight knots", "SplineTransformer averaged_inverted_cdf".

No match. #20526 (merged) added `sample_weight` to `SplineTransformer.fit` with the current two code paths. The same inconsistency was fixed in `KBinsDiscretizer` (`quantile_method`, default `"averaged_inverted_cdf"`) and handled from the start in `QuantileTransformer`'s weighted support (#32761).

## Describe the bug and its user-facing impact

With `knots="quantile"`, the knots are computed with `np.nanpercentile` (linear interpolation) when `sample_weight is None`, and with `_weighted_percentile` (inverted CDF, no interpolation) otherwise. So:

- `fit(X, sample_weight=np.ones(n))` does not give the same model as `fit(X)`;
- integer weights are not equivalent to repeating samples.

Users who pass weights in a pipeline (e.g. a uniform default weight vector) silently get a different spline basis.

## Steps/Code to Reproduce

```python
import numpy as np
from sklearn.preprocessing import SplineTransformer

X = np.arange(12, dtype=float).reshape(-1, 1) ** 2
st = SplineTransformer(n_knots=4, knots="quantile")
knots_no_weights = st.fit(X).bsplines_[0].t[3:-3]
knots_unit_weights = st.fit(X, sample_weight=np.ones(12)).bsplines_[0].t[3:-3]
print("sample_weight=None    :", knots_no_weights)
print("sample_weight=ones    :", knots_unit_weights)

# integer weights vs repeated samples
w = np.array([1, 2] * 6)
knots_weighted = st.fit(X, sample_weight=w).bsplines_[0].t[3:-3]
knots_repeated = st.fit(np.repeat(X, w, axis=0)).bsplines_[0].t[3:-3]
print("sample_weight=[1, 2,.]:", knots_weighted)
print("repeated samples      :", knots_repeated)
```

## Expected Results

Identical knots for the first two lines, and for the last two lines.

## Actual Results

```
sample_weight=None    : [  0.          13.66666667  54.         121.        ]
sample_weight=ones    : [  0.   9.  49. 121.]
sample_weight=[1, 2,.]: [  0.   9.  49. 121.]
repeated samples      : [  0.          13.66666667  54.         121.        ]
```

## Root cause and possible fix

`sklearn/preprocessing/_polynomial.py:770-773`:

```python
if sample_weight is None:
    knots = np.nanpercentile(X, percentile_ranks, axis=0)
else:
    knots = _weighted_percentile(X, sample_weight, percentile_ranks).T
```

The two branches use different quantile definitions. Using `method="averaged_inverted_cdf"` in the unweighted branch and `average=True` in the weighted one would make them agree (as done in `KBinsDiscretizer` and `QuantileTransformer`), at the cost of moving the knots of unweighted fits (`output-change`). The common test `check_sample_weight_equivalence_on_dense_data` does not catch this because it runs with the default `knots="uniform"`; adding `knots="quantile"` to the checked parameter sets in `sklearn/utils/_test_common/instance_generator.py` would.

## Versions

```
sklearn: 1.10.dev0, commit f570497251 (code identical on upstream/main 289c4d5cfa)
python: 3.13.14, numpy: 2.5.3, scipy: 1.18.1, platform: linux
```
