# `GaussianNB.partial_fit` subtracts the new batch's `epsilon_` instead of the previous one, corrupting `var_`

```json
{
    "estimators": ["GaussianNB"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 1,
    "difficulty": 1,
    "decision": 0,
    "decision_tags": []
}
```

## Classification

- **Priority (1)**: silent, but the shift is `var_smoothing` times the change of the largest variance between batches, so it is numerical noise with the default `1e-9`. It only matters for `partial_fit` (grid searches use `fit`) with a large tuned `var_smoothing` and batches of very different scales. (priority review, 2026-10-23: was 2)
- **Difficulty (1)**: swap two statements so that the old `epsilon_` is removed before the new one is computed.
- **Decision (0)**: pure fix; results only change for multi-call `partial_fit`.

## Upstream search

GraphQL searches "GaussianNB partial_fit epsilon", "GaussianNB partial_fit variance", "GaussianNB partial_fit var_ epsilon_ subtracted", "GaussianNB partial_fit fit different var_".

- #14054 (open, "var_smoothing parameter of GaussianNB isn't documented nor tested"): one comment notes that `epsilon_` is redefined at each `partial_fit` call. That is a design question (which data `epsilon_` should be based on), not this bug.
- #35022 (open DOC PR), #33526 (open), #33994 / #33575 (closed): documentation of `var_smoothing` and `epsilon_`. #35022 documents that `epsilon_` is recomputed from each batch, but none of them mention that the wrong value is subtracted from `var_`.

## Describe the bug and its user-facing impact

`GaussianNB` stores `var_ = class variance + epsilon_`. At each `partial_fit` call, it removes `epsilon_` from `var_`, updates the variances with the new batch, and adds `epsilon_` back. But `epsilon_` is overwritten with the value computed on the new batch *before* it is subtracted, so the stored variances are shifted by `(old_epsilon - new_epsilon)` before being merged with the new batch.

Even if we accept that `epsilon_` follows the last batch (as #35022 documents), `var_ - epsilon_` should be the per-class variance of all data seen so far. It is not, and the error grows with `var_smoothing` and with the scale difference between batches (drifting streams, batches sorted by time, etc.).

## Steps/Code to Reproduce

```python
import numpy as np
from sklearn.naive_bayes import GaussianNB

rng = np.random.RandomState(0)
y = rng.randint(2, size=200)
X = rng.normal(size=(200, 1))
X[100:] *= 10  # the second batch has a 10x larger spread

clf = GaussianNB(var_smoothing=0.1)
clf.partial_fit(X[:100], y[:100], classes=[0, 1])
clf.partial_fit(X[100:], y[100:])

true_var = np.array([X[y == c].var() for c in [0, 1]])
print("var_ - epsilon_ (partial_fit):", (clf.var_ - clf.epsilon_).ravel())
print("per-class variance of X      :", true_var)
```

## Expected Results

`var_ - epsilon_` equals the per-class variance of the data seen so far, as it does after `fit`.

## Actual Results

```
var_ - epsilon_ (partial_fit): [56.03815171 33.34042549]
per-class variance of X      : [60.27125562 38.6213274 ]
```

## Root cause and possible fix

`sklearn/naive_bayes.py:441` sets `self.epsilon_` from the current batch, then `sklearn/naive_bayes.py:481` does `self.var_[:, :] -= self.epsilon_` with that new value, although `var_` contains the epsilon of the previous call. `sklearn/naive_bayes.py:514` then adds the new value back.

Fix: subtract the previous `epsilon_` before overwriting it (e.g. keep `old_epsilon = self.epsilon_` in the non-first-call branch, or move the subtraction before line 441). Whether `epsilon_` should be based on all data seen so far rather than the last batch (#14054) is a separate design question.

## Versions

```
sklearn: 1.10.dev0, commit 289c4d5cfa (upstream/main; local HEAD f570497251 has no diff in sklearn/naive_bayes.py)
python: 3.13, numpy: 2.5.3, scipy: 1.18.1, platform: linux
```
