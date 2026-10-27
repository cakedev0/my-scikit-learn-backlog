# GroupKFold docstring says folds are balanced when `shuffle=True`, but they are balanced only when `shuffle=False`

```json
{
    "estimators": ["GroupKFold"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 1,
    "difficulty": 1,
    "decision": 0,
    "decision_tags": []
}
```

## Classification

- **Priority (1)**: documentation only, but it states the opposite of what the code does, and users picking `shuffle=True` for "balanced" folds get unbalanced test folds.
- **Difficulty (1)**: one-word docstring fix, plus maybe a sentence about what `shuffle=True` balances.
- **Decision (0)**: no behavior change.

## Upstream search

`gh issue list --search` / `gh pr list --search` (GraphQL): "GroupKFold shuffle balanced", "GroupKFold shuffle docstring balanced". Only #28519 (merged, added `shuffle` to `GroupKFold`, which introduced this sentence). No issue about it.

## Describe the bug and its user-facing impact

The `GroupKFold` docstring says:

> The folds are approximately balanced in the sense that the number of samples is approximately the same in each test fold when `shuffle` is True.

The code does the opposite. With `shuffle=False`, groups are assigned greedily (largest group to the lightest fold) to balance the number of samples per fold. With `shuffle=True`, the shuffled groups are split into `n_splits` chunks with `np.array_split`, which balances the number of *groups* per fold, not the number of samples.

## Steps/Code to Reproduce

```python
import numpy as np
from sklearn.model_selection import GroupKFold

# one large group and many small ones
groups = np.repeat(np.arange(10), [30, 10, 10, 10, 10, 10, 10, 10, 10, 10])
X = np.zeros((len(groups), 1))
for shuffle in [False, True]:
    cv = GroupKFold(n_splits=3, shuffle=shuffle, random_state=0 if shuffle else None)
    print(f"shuffle={shuffle}: test fold sizes",
          [len(test) for _, test in cv.split(X, groups=groups)])
```

## Expected Results

Following the docstring, balanced test folds with `shuffle=True`.

## Actual Results

```
shuffle=False: test fold sizes [40, 40, 40]
shuffle=True: test fold sizes [40, 30, 50]
```

## Root cause and possible fix

`sklearn/model_selection/_split.py:537`: replace "when `shuffle` is True" by "when `shuffle` is False", and say that with `shuffle=True` each test fold gets approximately the same number of groups.

## Versions

```
sklearn: 1.10.dev0, commit f570497251 (local branch; _split.py is identical on
upstream/main 289c4d5cfa)
python: 3.13, numpy: 2.5.3, scipy: 1.18.1, platform: linux
```
