# `make_multilabel_classification(n_labels=0, allow_unlabeled=False)` never returns

```json
{
    "estimators": ["make_multilabel_classification"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 1,
    "difficulty": 1,
    "decision": 0,
    "decision_tags": []
}
```

## Classification

- **Priority (1)**: a hang on input accepted by parameter validation, but the parameter combination is contradictory and unlikely in practice.
- **Difficulty (1)**: raise a `ValueError` (or tighten the constraint) when `n_labels == 0` and `allow_unlabeled=False`.
- **Decision (0)**: the call can never succeed today, so raising breaks nothing.

## Upstream search

Queries: "make_multilabel_classification n_labels=0", "make_multilabel_classification infinite loop", "make_multilabel_classification n_labels allow_unlabeled".

- #5584 (closed unmerged in 2018, "Modified to avoid infinity loop if n_labels=0"): closed after @amueller noted that `make_multilabel_classification(n_labels=0)` works. It does, with the default `allow_unlabeled=True`; the hang only happens with `allow_unlabeled=False`, which the discussion did not consider.

## Describe the bug and its user-facing impact

The number of labels of each sample is drawn from `Poisson(n_labels)` and redrawn while it is 0 and `allow_unlabeled=False`. With `n_labels=0` the Poisson draw is always 0, so the rejection loop runs forever. `n_labels=0` passes parameter validation (`Interval(Integral, 0, None, closed="left")`), and the docstring says the number of labels "must be nonzero if `allow_unlabeled` is False" without saying `n_labels` must then be positive.

The user gets a frozen Python process with no error.

## Steps/Code to Reproduce

```python
from sklearn.datasets import make_multilabel_classification

X, Y = make_multilabel_classification(n_samples=5, n_labels=0, allow_unlabeled=True, random_state=0)
print("allow_unlabeled=True ok", Y.sum())
X, Y = make_multilabel_classification(n_samples=5, n_labels=0, allow_unlabeled=False, random_state=0)
print("never reached")
```

## Expected Results

A `ValueError` saying that `n_labels` must be at least 1 when `allow_unlabeled=False`.

## Actual Results

Run with `timeout 10 python -u repro.py; echo "exit=$?"`:

```
allow_unlabeled=True ok 0
exit=124
```

## Root cause and possible fix

`sklearn/datasets/_samples_generator.py:518` (upstream/main 289c4d5cfa): `while (not allow_unlabeled and y_size == 0) or y_size > n_classes: y_size = generator.poisson(n_labels)` cannot exit when `n_labels == 0` and `allow_unlabeled` is False. Add a check before sampling:

```python
if n_labels == 0 and not allow_unlabeled:
    raise ValueError("n_labels must be at least 1 when allow_unlabeled=False.")
```

## Versions

```
sklearn: 1.10.dev0, commit 289c4d5cfa (upstream/main; local build f570497251, no diff on sklearn/datasets/_samples_generator.py)
python: 3.13, numpy: 2.5.3, scipy: 1.18.1, platform: linux
```
