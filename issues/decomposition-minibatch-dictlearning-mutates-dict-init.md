# MiniBatchDictionaryLearning modifies dict_init in place (and the components_ of the model it came from)

```json
{
    "estimators": ["MiniBatchDictionaryLearning", "dict_learning_online"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 2,
    "difficulty": 1,
    "decision": 0,
    "decision_tags": []
}
```

## Classification

- **Priority (2)**: silent mutation of a user array and of a constructor parameter. The common warm-start pattern `dict_init=other_model.components_` triggers it, because `components_` is Fortran-ordered, and it then corrupts `other_model`. Refitting the same estimator also starts from a different dictionary each time.
- **Difficulty (1)**: force a copy of `dict_init` in `_initialize_dict`.
- **Decision (0)**: no trade-off, the copy is `(n_components, n_features)`.

## Upstream search

- `"MiniBatchDictionaryLearning dict_init"`: no match. `git log -L` shows `_initialize_dict` comes from #18975 ("Refactor MiniBatchDictionaryLearning and add stopping criterion", merged) and has not changed since.

## Describe the bug and its user-facing impact

`MiniBatchDictionaryLearning._initialize_dict` uses `dict_init` directly and converts it with `check_array(..., order="F", dtype=X.dtype, copy=False)` and `np.require(..., requirements="W")`. When `dict_init` is already a writeable Fortran-ordered array of the right dtype, no copy is made, and `_update_dict` then updates it in place at every minibatch step. `components_` ends up being the very same array as `dict_init`.

`components_` of a fitted `MiniBatchDictionaryLearning` (or `DictionaryLearning`) is Fortran-ordered, so warm-starting from a previous model, as the docs of `dict_init` suggest, silently overwrites that previous model's dictionary. A C-ordered `dict_init` (or one with another dtype) is copied and not affected, which makes the behavior look random to users. `DictionaryLearning` (batch) does not have the problem.

## Steps/Code to Reproduce

```python
import numpy as np
from sklearn.decomposition import MiniBatchDictionaryLearning

rng = np.random.RandomState(0)
X = rng.randn(50, 8)

first = MiniBatchDictionaryLearning(n_components=3, batch_size=10, random_state=0).fit(X)
components_before = first.components_.copy()

# warm start a second model from the first one's dictionary
second = MiniBatchDictionaryLearning(
    n_components=3, batch_size=10, dict_init=first.components_, random_state=0
)
second.fit(X[:25])
print("first.components_ changed:", not np.array_equal(first.components_, components_before))
print("second.dict_init changed: ", not np.array_equal(second.dict_init, components_before))
print("shares memory:            ", np.shares_memory(first.components_, second.components_))
```

## Expected Results

```
first.components_ changed: False
second.dict_init changed:  False
shares memory:             False
```

## Actual Results

```
first.components_ changed: True
second.dict_init changed:  True
shares memory:             True
```

## Root cause and possible fix

`sklearn/decomposition/_dict_learning.py` (commit 289c4d5cfa):

```python
if self.dict_init is not None:
    dictionary = self.dict_init                                       # line 2015
...
dictionary = check_array(dictionary, order="F", dtype=X.dtype, copy=False)  # line 2036
dictionary = np.require(dictionary, requirements="W")
```

and `_update_dict` (line 467) writes into `dictionary` in place (`dictionary[k] = ...`, line 531). Using `copy=True` in the `check_array` call when `dict_init` is given (or always: the SVD init is a fresh array anyway, so the extra copy is negligible) fixes it. Estimators must not modify their constructor parameters, so a test with an F-ordered `dict_init` comparing it before and after `fit` and `partial_fit` would cover it.

## Versions

```
sklearn: 1.10.dev0, commit 289c4d5cfa (upstream/main)
python: 3.13, numpy: 2.5.3, scipy: 1.18.1, platform: linux
```
