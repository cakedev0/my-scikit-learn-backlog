# LatentDirichletAllocation.perplexity resets n_features_in_ and feature_names_in_ of the fitted model

```json
{
    "estimators": ["LatentDirichletAllocation"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 1,
    "difficulty": 1,
    "decision": 1,
    "decision_tags": ["new-error"]
}
```

## Classification

- **Priority (1)**: with correctly shaped data, `perplexity` resets `n_features_in_` to the same value, so nothing visible happens. The bug only shows up on user error (wrong number of features, which should raise) or when switching from a DataFrame fit to ndarray input, rare for LDA which is usually fit on sparse count matrices (priority review, 2026-10-23: was 2)
- **Difficulty (1)**: change `reset_n_features=True` to `False`, as in `score` and `transform`.
- **Decision (1)**: `new-error`: `perplexity` on data with fewer features than at fit time raises instead of returning a number (the number was wrong anyway).

## Upstream search

- `"LatentDirichletAllocation perplexity"`: #8943 (open, perplexity too big on a Wiki dump), #9134 (open, perplexity with normalized vs unnormalized doc-topic matrix), #33363 (open PR, perplexity normalization for the component prior, does not touch the validation), #7954 (closed, 0.18.1 regression). None is about the validation resetting the fitted state.
- `git log -L` on the line: `reset_n_features=True` in `perplexity` predates #21389 (merged, 2021, "FIX delete feature_names_in_ when refitting on a ndarray"), which moved it from `_perplexity_precomp_distr` to `perplexity` unchanged. No later change.

## Describe the bug and its user-facing impact

`LatentDirichletAllocation.perplexity(X)` validates `X` with `reset=True`, as if it were `fit`. As a result:

- `n_features_in_` is overwritten with `X.shape[1]`, and `feature_names_in_` is reset (deleted when `X` is an ndarray), so a following `transform`/`score` on correctly shaped data raises `ValueError: X has 10 features, but LatentDirichletAllocation is expecting 5 features`.
- No consistency check is done, so `perplexity` on data with fewer features than `components_` returns a number computed on the wrong vocabulary columns instead of raising.

`score` and `transform` use `reset_n_features=False` and do not have the problem.

## Steps/Code to Reproduce

```python
import numpy as np
from sklearn.decomposition import LatentDirichletAllocation

rng = np.random.RandomState(0)
X = rng.poisson(1.0, size=(40, 10))
lda = LatentDirichletAllocation(n_components=3, random_state=0, max_iter=5).fit(X)

# Wrong number of features: no error, and the fitted model is modified
print("perplexity on 5 features:", lda.perplexity(X[:, :5]))
print("n_features_in_:", lda.n_features_in_)
lda.transform(X)
```

## Expected Results

`perplexity(X[:, :5])` raises `ValueError: X has 5 features, but LatentDirichletAllocation is expecting 10 features as input.`, and the fitted attributes are left unchanged.

## Actual Results

```
perplexity on 5 features: 15.854306295853235
n_features_in_: 5
Traceback (most recent call last):
  ...
  File "sklearn/decomposition/_lda.py", line 731, in transform
    X = self._check_non_neg_array(
        X, reset_n_features=False, whom="LatentDirichletAllocation.transform"
    )
  ...
ValueError: X has 10 features, but LatentDirichletAllocation is expecting 5 features as input.
```

## Root cause and possible fix

`sklearn/decomposition/_lda.py:932` (commit 289c4d5cfa):

```python
X = self._check_non_neg_array(
    X, reset_n_features=True, whom="LatentDirichletAllocation.perplexity"
)
```

Should be `reset_n_features=False`. With `reset_n_features=True`, `_check_non_neg_array` also accepts `float32` or `float64` independently of `components_.dtype`; with `False` it casts to `components_.dtype`, like `score`.

## Versions

```
sklearn: 1.10.dev0, commit 289c4d5cfa (upstream/main)
python: 3.13, numpy: 2.5.3, scipy: 1.18.1, platform: linux
```
