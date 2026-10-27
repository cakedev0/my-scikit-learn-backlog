# NMF.transform with beta_loss="kullback-leibler" adds the L1 penalty on W once more at each iteration

```json
{
    "estimators": ["NMF", "non_negative_factorization"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 2,
    "difficulty": 1,
    "decision": 0,
    "decision_tags": []
}
```

## Classification

- **Priority (2)**: silent wrong output of `NMF.transform` (and `non_negative_factorization(update_H=False)`): `W` shrinks towards 0 as iterations go, so the result depends on `max_iter`/`tol` instead of converging. Needs a non-default combination (`solver="mu"`, `beta_loss="kullback-leibler"` or `1`, `alpha_W > 0`, `l1_ratio > 0`), but this is the usual setup for sparse topic models with NMF.
- **Difficulty (1)**: an in-place `+=` on a cached array; replace it by an out-of-place addition.
- **Decision (0)**: no trade-off.

## Upstream search

- `"NMF kullback-leibler transform alpha_W"`, `"NMF transform regularization kullback"`: no match.
- `git log upstream/main -- sklearn/decomposition/_nmf.py`: no recent change to `_multiplicative_update_w`.

## Describe the bug and its user-facing impact

With `solver="mu"` and `beta_loss="kullback-leibler"`, `NMF.transform` runs the multiplicative updates of `W` with `H` fixed. In that mode, `H_sum = H.sum(axis=1)` is computed once and reused across iterations. The L1 penalty is then added to the denominator in place, and the denominator is a view of `H_sum`, so `H_sum` grows by `l1_reg_W` at every iteration. After `k` iterations the effective L1 penalty is `k` times the requested one.

The visible effect: `transform(X)` on the training data does not match `fit_transform(X)`, and with `tol=0` the output keeps shrinking towards 0 as `max_iter` grows. `fit`/`fit_transform` are not affected because `H_sum` is reset after every `H` update. Frobenius and other beta losses are not affected because their denominators are fresh arrays.

## Steps/Code to Reproduce

```python
import numpy as np
from sklearn.decomposition import NMF

rng = np.random.RandomState(0)
X = rng.rand(30, 10)

for alpha_W in [0.0, 0.01]:
    nmf = NMF(n_components=3, solver="mu", beta_loss="kullback-leibler",
              alpha_W=alpha_W, l1_ratio=1.0, max_iter=1000, random_state=0)
    W_fit = nmf.fit_transform(X)
    W_tr = nmf.transform(X)
    print(f"alpha_W={alpha_W}: W.sum() fit_transform={W_fit.sum():.3f}  transform={W_tr.sum():.3f}")

for max_iter in [10, 100, 1000]:
    nmf.set_params(max_iter=max_iter, tol=0)
    print(f"transform with max_iter={max_iter}: W.sum()={nmf.transform(X).sum():.3f}")
```

## Expected Results

`transform(X)` close to `fit_transform(X)` (as it is for `alpha_W=0`), and a `W` that converges when `max_iter` grows.

## Actual Results

```
alpha_W=0.0: W.sum() fit_transform=22.519  transform=22.539
alpha_W=0.01: W.sum() fit_transform=35.840  transform=24.307
transform with max_iter=10: W.sum()=29.293
transform with max_iter=100: W.sum()=10.524
transform with max_iter=1000: W.sum()=1.447
```

I also checked against a plain NumPy implementation of the KL multiplicative update with a fixed `H` and the same scaled penalty (`l1 = alpha_W * n_features`): it converges to `W.sum() ~ 39.98`, while `non_negative_factorization(..., update_H=False, tol=0)` gives 31.6, 10.8 and 1.45 after 10, 100 and 1000 iterations.

## Root cause and possible fix

`sklearn/decomposition/_nmf.py` (commit 289c4d5cfa), in `_multiplicative_update_w`:

```python
if beta_loss == 1:
    if H_sum is None:
        H_sum = np.sum(H, axis=1)  # shape(n_components, )
    denominator = H_sum[np.newaxis, :]          # line 591: a view of H_sum
...
if l1_reg_W > 0:
    denominator += l1_reg_W                     # line 612: mutates H_sum
```

`H_sum` is returned and passed back by `_fit_multiplicative_update` on the next iteration when `update_H=False`. Using `denominator = denominator + l1_reg_W` (as already done for the L2 term), or copying `H_sum` when building the denominator, fixes it. The `denominator[denominator == 0] = EPSILON` line also writes into `H_sum` and would be fixed by the same change.

## Versions

```
sklearn: 1.10.dev0, commit 289c4d5cfa (upstream/main)
python: 3.13, numpy: 2.5.3, scipy: 1.18.1, platform: linux
```
