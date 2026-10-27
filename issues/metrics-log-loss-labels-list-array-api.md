# `log_loss` and `brier_score_loss` crash with array API `y_true` and `labels` passed as a list

```json
{
    "estimators": ["log_loss", "brier_score_loss", "d2_log_loss_score", "d2_brier_score"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 1,
    "difficulty": 1,
    "decision": 0,
    "decision_tags": []
}
```

## Classification

- **Priority (1)**: array API dispatch with non-NumPy inputs is still a small minority of uses, the error is loud, and passing `labels` as an array of the input namespace works around it. (priority review, 2026-10-23: was 2)
- **Difficulty (1)**: convert `labels` to the namespace of `lb.classes_` (or compare in NumPy) before the comparison in one helper.
- **Decision (0)**: plain fix.

## Upstream search

Downloaded all issues/PRs labeled "Array API" (319) and grepped for `log_loss`, `brier`, `labels`: only the PRs adding array API support (#30439, #31191, #32422) and type promotion fixes (#33022, #32923), none about `labels`. The GitHub search API was rate limited (secondary limit) for the free text queries; `gh pr list --search "log_loss labels torch"` (GraphQL) only returns #32422, #30439 and #32755 (mixed inputs test, default `labels=None`).

## Describe the bug and its user-facing impact

With `array_api_dispatch=True`, `log_loss`, `brier_score_loss`, `d2_log_loss_score` and `d2_brier_score` raise a `TypeError` when `y_true` is an array API array (e.g. a torch tensor) and `labels` is a Python list or a NumPy array. The same call works with NumPy inputs, and with `labels` passed as a tensor.

`labels` is documented as array-like, and the user guide says all array inputs follow `y_pred`, so a list should be accepted. Users hit this as soon as they pass `labels` explicitly, which is common in cross-validation where a fold may miss a class.

## Steps/Code to Reproduce

```python
import numpy as np
import torch
import sklearn
from sklearn.metrics import brier_score_loss, log_loss

y_true = np.array([0, 1, 2, 1])
y_prob = np.array([[0.7, 0.2, 0.1], [0.2, 0.6, 0.2], [0.1, 0.2, 0.7], [0.3, 0.4, 0.3]])

print("NumPy:", log_loss(y_true, y_prob, labels=[0, 1, 2]))
with sklearn.config_context(array_api_dispatch=True):
    y_true_t, y_prob_t = torch.asarray(y_true), torch.asarray(y_prob)
    print("torch, labels=torch:", log_loss(y_true_t, y_prob_t, labels=torch.asarray([0, 1, 2])))
    for f in [log_loss, brier_score_loss]:
        try:
            f(y_true_t, y_prob_t, labels=[0, 1, 2])
        except TypeError as e:
            print(f"torch, {f.__name__}, labels=list:", type(e).__name__, str(e).splitlines()[0])
```

## Expected Results

The same value as NumPy, `0.5351165608794026`, for `labels=[0, 1, 2]`.

## Actual Results

```
NumPy: 0.5351165608794026
torch, labels=torch: 0.5351165608794026
torch, log_loss, labels=list: TypeError all() received an invalid combination of arguments - got (numpy.ndarray), but expected one of:
torch, brier_score_loss, labels=list: TypeError all() received an invalid combination of arguments - got (numpy.ndarray), but expected one of:
```

## Root cause and possible fix

In `_one_hot_encoding_multiclass_target` (`sklearn/metrics/_classification.py:208`), `xp` is the namespace of `y_true` (torch), but `lb = LabelBinarizer().fit(labels)` fits on the list, so `lb.classes_` is a NumPy array and `lb.classes_ == labels` is a NumPy bool array, passed to `torch.all`. The next check, `xpx.isin(y_true, labels, xp=xp)`, would also get a list for `labels`.

Fix: use the namespace of `lb.classes_` for the first check (`xp_classes.all(...)`), and move `labels` to `y_true`'s namespace/device before `isin` (or do both checks in NumPy, since `labels` is small).

Side note in the same helper: the error message for undeclared labels uses `set(y_true) - set(labels)` (line 217), which for torch tensors lists every element of `y_true` (tensors hash by identity), e.g. `y_true contains values {tensor(0), tensor(3), tensor(1), tensor(1)} not belonging to the passed labels tensor([0, 1, 2])`.

## Versions

```
sklearn: 1.10.dev0, commit f570497251 (metrics/_classification.py identical to upstream/main 289c4d5cfa)
python: 3.13.14, numpy: 2.5.3, scipy: 1.18.1, torch: 2.14.1+xpu, platform: linux
```
