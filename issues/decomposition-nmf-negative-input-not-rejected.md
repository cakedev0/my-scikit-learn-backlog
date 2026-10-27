# NMF(init="custom"), MiniBatchNMF.transform and MiniBatchNMF.partial_fit accept negative X and return negative W

```json
{
    "estimators": ["NMF", "MiniBatchNMF", "non_negative_factorization"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 1,
    "difficulty": 1,
    "decision": 1,
    "decision_tags": ["new-error"]
}
```

## Classification

- **Priority (1)**: negative input is a user error, but these code paths silently return a "non-negative" factorization with negative entries, while the sibling paths (`NMF.fit` with other inits, `NMF.transform`, `MiniBatchNMF.fit`) raise a clear error.
- **Difficulty (1)**: add `ensure_non_negative=True` (or `check_non_negative`) to three validation calls.
- **Decision (1)**: `new-error`: code that runs today (with meaningless results) would raise.

## Upstream search

- `"MiniBatchNMF negative"`, `"MiniBatchNMF negative partial_fit transform"`: no match.
- `"MiniBatchNMF transform"`: no match. `"MiniBatchNMF"`: #29563 (open, `_check_w_h` reads `self.solver`, which `MiniBatchNMF` lacks), unrelated.

## Describe the bug and its user-facing impact

The non-negativity check on `X` is done in different places depending on the code path, and three paths miss it:

- `NMF.fit`/`fit_transform` and `non_negative_factorization` with `init="custom"`: the only check is in `_initialize_nmf`, which is skipped for custom inits. With `solver="mu"` the returned `W` has negative entries; with `solver="cd"` the result is clipped but meaningless.
- `MiniBatchNMF.transform`: unlike `NMF.transform`, it does not pass `ensure_non_negative=True` to `validate_data`, and returns negative `W`.
- `MiniBatchNMF.partial_fit` after the first call: `X` is only validated with `validate_data`, so negative batches are silently used to update `components_` (and can produce `NaN` errors deep in the solver, e.g. `partial_fit(-10 * X)` fails with "array must not contain infs or NaNs").

## Steps/Code to Reproduce

```python
import numpy as np
from sklearn.decomposition import NMF, MiniBatchNMF

rng = np.random.RandomState(0)
X = rng.rand(20, 5)
X_neg = X - 0.3

# Raises as expected
try:
    NMF(n_components=2).fit(X_neg)
except ValueError as e:
    print("NMF.fit:", e)

# Accepted silently, W has negative entries
nmf = NMF(n_components=2, init="custom", solver="mu")
W = nmf.fit_transform(X_neg, W=rng.rand(20, 2), H=rng.rand(2, 5))
print("NMF(init='custom', solver='mu').fit_transform: min(W) =", W.min())

mbnmf = MiniBatchNMF(n_components=2, random_state=0).fit(X)
print("MiniBatchNMF.transform: min(W) =", mbnmf.transform(X_neg).min())
mbnmf.partial_fit(X_neg)  # no error
print("MiniBatchNMF.partial_fit: accepted")
```

## Expected Results

A `ValueError: Negative values in data passed to ...` in all four cases, as for `NMF.fit` and `NMF.transform`.

## Actual Results

```
NMF.fit: Negative values in data passed to NMF initialization.
NMF(init='custom', solver='mu').fit_transform: min(W) = -0.28480701607256226
MiniBatchNMF.transform: min(W) = -0.23041426814599913
MiniBatchNMF.partial_fit: accepted
```

## Root cause and possible fix

In `sklearn/decomposition/_nmf.py` (commit 289c4d5cfa):

- `NMF.fit_transform` (`validate_data`, line 1586) and `non_negative_factorization` (`check_array`, line 1110) rely on `_initialize_nmf` (line 276) for the check, which is not called when `init="custom"` (`_check_w_h`, line 1182, only checks `W` and `H`).
- `MiniBatchNMF.transform` (`validate_data`, line 2303) calls `validate_data` without `ensure_non_negative=True`, while `NMF.transform` does.
- `MiniBatchNMF.partial_fit` (`validate_data`, line 2349) never checks `X`; `MiniBatchNMF.fit` does it in `_fit_transform` (line 2232).

Passing `ensure_non_negative=True` to `validate_data` in `_BaseNMF`'s entry points (and in `non_negative_factorization`) would make all paths consistent. The common tests do not catch this because `positive_only` estimators are only checked with the default init and through `fit`.

## Versions

```
sklearn: 1.10.dev0, commit 289c4d5cfa (upstream/main)
python: 3.13, numpy: 2.5.3, scipy: 1.18.1, platform: linux
```
