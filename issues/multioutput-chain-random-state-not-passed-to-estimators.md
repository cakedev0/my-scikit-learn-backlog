# ClassifierChain / RegressorChain `random_state` does not seed the base estimators, contrary to the docstring

```json
{
    "estimators": ["ClassifierChain", "RegressorChain"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 1,
    "difficulty": 1,
    "decision": 1,
    "decision_tags": ["output-change", "rng-order"]
}
```

## Classification

- **Priority (1)**: a docstring promise that the code does not keep. Users who rely on `random_state=0` for reproducibility get different models at each fit with a randomized base estimator. The workaround (set `random_state` on the base estimator) is cheap and common.
- **Difficulty (1)**: either fix the docstring, or call `_set_random_states(estimator, random_state)` on each clone (as the ensemble meta-estimators do).
- **Decision (1)**: implementing the documented behavior changes the fitted models for users who set `random_state` and a randomized base estimator (`output-change`, `rng-order`). Fixing the docstring instead has no trade-off.

## Upstream search

- `gh issue list --search "ClassifierChain random_state"` and `"chain random_state base estimator"`: no match (#13339, #29856, #32855, #26817, #21761 are about other chain problems).
- `git blame`: the "In addition, it controls the random seed given at each `base_estimator`" sentence comes from 98f0eb8137e (2020, random_state docstring sprint). `git log -S_set_random_states -- sklearn/multioutput.py` is empty: it was never implemented.

## Describe the bug and its user-facing impact

The docstring of `random_state` in both chains says:

> If ``order='random'``, determines random number generation for the chain order. In addition, it controls the random seed given at each `base_estimator` at each chaining iteration. Thus, it is only used when `base_estimator` exposes a `random_state`. Pass an int for reproducible output across multiple function calls.

But `fit` only uses `random_state` for `order="random"`. The base estimators are plain clones, so with a randomized estimator like `RandomForestClassifier()` two fits with `random_state=0` give different models.

## Steps/Code to Reproduce

```python
import numpy as np
from sklearn.datasets import make_multilabel_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import ClassifierChain

X, Y = make_multilabel_classification(n_samples=200, n_classes=3, random_state=0)
proba = [
    ClassifierChain(RandomForestClassifier(n_estimators=5), random_state=0)
    .fit(X, Y)
    .predict_proba(X[:2])
    for _ in range(2)
]
print(proba[0], proba[1], sep="\n")
print("identical:", np.array_equal(*proba))
```

## Expected Results

Identical predictions, as the docstring says `random_state` "controls the random seed given at each `base_estimator`" and "Pass an int for reproducible output across multiple function calls".

## Actual Results

```
[[0. 1. 0.]
 [0. 1. 0.]]
[[0.2 1.  0. ]
 [0.  1.  0. ]]
identical: False
```

(The exact values change from run to run, which is the point.)

## Root cause and possible fix

`sklearn/multioutput.py:718` (`_BaseChain.fit`): `self.estimators_ = [clone(self.estimator) for _ in range(Y.shape[1])]`, and `random_state` is only used at `sklearn/multioutput.py:714` for the permutation. Docstrings at `sklearn/multioutput.py:878` (ClassifierChain) and `:1136` (RegressorChain).

Two options: drop the second sentence of the docstring, or seed each clone with `sklearn.ensemble._base._set_random_states` like bagging does. The cross-validated fits (`cross_val_predict(self.estimator, ...)`) would need the same treatment. I'd go for the docstring fix: MultiOutputClassifier/Regressor don't do this either, and users already seed the base estimator.

## Versions

```
sklearn: 1.10.dev0, commit 289c4d5cfa (upstream/main; reproduced on local checkout f570497251, no diff in the relevant code)
python: 3.13, numpy: 2.5.3, scipy: 1.18.1, platform: linux
```
