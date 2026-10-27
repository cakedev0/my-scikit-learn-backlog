# MLPClassifier.partial_fit with warm_start=True requires every batch to contain all classes

```json
{
    "estimators": ["MLPClassifier"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 1,
    "difficulty": 1,
    "decision": 0,
    "decision_tags": []
}
```

## Classification

- **Priority (1)**: loud error, and `warm_start` has no purpose with `partial_fit` (which always continues from the current state), so the workaround is to leave `warm_start=False`. But the error contradicts the `partial_fit` docstring, and users who set `warm_start=True` "to be safe" for incremental learning hit it on the first mini-batch that misses a class.
- **Difficulty (1)**: in `_validate_input`, only run the warm-start check when `incremental` is False.
- **Decision (0)**: no trade-off; it only removes an error on valid input.

## Upstream search

Queries: "MLPClassifier partial_fit warm_start", "MLPClassifier warm_start solver".

- #11758 (closed 2018, "warm_start can only be used where `y` has the same classes as in the previous call to fit."): same error, reported with `partial_fit` + `warm_start=True`. @jnothman wrote "this may be a subtle bug" and asked for a doc PR, the reporter switched `warm_start` off, and the issue was closed without any code or doc change. New argument: the `partial_fit` docstring explicitly says `y` does not need to contain all labels in `classes`, nothing documents that `warm_start` breaks this, and the fix is one condition.
- #8035 (merged, "Catch cases for different class size in MLPClassifier with warm start (#7976)"): introduced the check for `fit` with `warm_start`, the intended use.
- #12505 (open): multilabel + `partial_fit`, different issue.

## Describe the bug and its user-facing impact

`MLPClassifier.partial_fit` documents for `classes`: "Note that y doesn't need to contain all labels in `classes`." This holds with `warm_start=False`, but with `warm_start=True` any call (including the first one) whose `y` misses a class raises:

```
ValueError: warm_start can only be used where `y` has the same classes as in the previous call to fit.
```

The check is meant for `fit` with `warm_start=True` (re-fitting on a dataset with different classes). `partial_fit` already checks that `y` is a subset of `classes_`, and `warm_start` is irrelevant to it: the `partial_fit` code path never re-initializes the weights.

## Steps/Code to Reproduce

```python
import numpy as np
from sklearn.neural_network import MLPClassifier

rng = np.random.RandomState(0)
X, y = rng.randn(30, 3), np.repeat([0, 1, 2], 10)

for warm_start in [False, True]:
    clf = MLPClassifier(warm_start=warm_start, random_state=0)
    try:
        clf.partial_fit(X[:20], y[:20], classes=[0, 1, 2])  # batch with classes 0, 1
        print(f"warm_start={warm_start}: OK, classes_={clf.classes_}")
    except ValueError as e:
        print(f"warm_start={warm_start}: ValueError: {e}")
```

## Expected Results

Both calls succeed, as documented, since `y` only contains classes listed in `classes`.

## Actual Results

```
warm_start=False: OK, classes_=[0 1 2]
warm_start=True: ValueError: warm_start can only be used where `y` has the same classes as in the previous call to fit. Previously got [0 1 2], `y` has [0 1]
```

## Root cause and possible fix

`MLPClassifier._validate_input` (`sklearn/neural_network/_multilayer_perceptron.py:1238-1250` on the commit below) checks `if self.warm_start:` before `elif` (the `partial_fit` subset check), so the strict equality check wins whenever `warm_start=True`, including for `incremental=True`. The comment table above it even lists `incremental=1, warm_start=1 -> check compat warm_start`.

Fix: use `if self.warm_start and not incremental:` so `partial_fit` always uses the subset check.

## Versions

```
sklearn: 1.10.dev0, commit 289c4d5cfa (upstream/main; ran on f570497251, no diff near the code involved)
python: 3.13.14, numpy: 2.5.3, scipy: 1.18.1, platform: linux
```
