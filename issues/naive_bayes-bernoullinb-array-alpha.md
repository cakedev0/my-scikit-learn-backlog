# `BernoulliNB` with an array `alpha` gives wrong probabilities or crashes

```json
{
    "estimators": ["BernoulliNB"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 1,
    "difficulty": 1,
    "decision": 0,
    "decision_tags": []
}
```

## Classification

- **Priority (1)**: per-feature `alpha` is a niche option nobody seems to use, and in the usual case (`n_classes != n_features`) `fit` raises loudly; the silent wrong probabilities need `n_classes == n_features` on top of it. Workaround: scalar `alpha`. (priority review, 2026-10-23: was 2)
- **Difficulty (1)**: the class count normalizer must be broadcast per feature, a one-line change.
- **Decision (0)**: the current output for array `alpha` is wrong or an exception, nothing to preserve.

## Upstream search

GraphQL searches "BernoulliNB alpha array", "BernoulliNB alpha broadcast", "BernoulliNB alpha vector n_features", "naive bayes alpha array-like per feature", "is:open BernoulliNB". No match; the hits are unrelated (#13027, #15921, #22574).

## Describe the bug and its user-facing impact

`BernoulliNB` documents `alpha` as "float or array-like of shape (n_features,)", and `_check_alpha` accepts an array of length `n_features`. The smoothed estimate should be `P(x_j = 1 | c) = (N_cj + alpha_j) / (N_c + 2 alpha_j)`. But the denominator is computed as `class_count_ + alpha * 2`, i.e. a vector of shape `(n_classes,)` added to a vector of shape `(n_features,)`:

- when `n_classes != n_features` (the usual case) `fit` raises a broadcasting `ValueError`;
- when `n_classes == n_features` it silently uses `alpha_c` (indexed by class) instead of `alpha_j` (indexed by feature), so `feature_log_prob_` and predictions are wrong. With uneven `alpha`, `feature_log_prob_` can even exceed 0, giving NaN in `predict_proba`.

`MultinomialNB` and `ComplementNB` handle array `alpha` correctly.

## Steps/Code to Reproduce

```python
import numpy as np
from sklearn.naive_bayes import BernoulliNB

rng = np.random.RandomState(0)
X = rng.randint(2, size=(20, 2))
y = rng.randint(2, size=20)
alpha = np.array([1.0, 3.0])

clf = BernoulliNB(alpha=alpha).fit(X, y)
print("feature probabilities:\n", np.exp(clf.feature_log_prob_))

# Reference: P(x_j = 1 | c) = (N_cj + alpha_j) / (N_c + 2 * alpha_j)
N_cj = np.array([X[y == c].sum(axis=0) for c in [0, 1]])
N_c = np.bincount(y)[:, None]
print("expected:\n", (N_cj + alpha) / (N_c + 2 * alpha))

# With 3 features and 2 classes, fit crashes
X3 = rng.randint(2, size=(20, 3))
BernoulliNB(alpha=np.ones(3)).fit(X3, y)
```

## Expected Results

The feature probabilities match the reference, and `fit` works for any `n_features`.

## Actual Results

```
feature probabilities:
 [[0.44444444 0.88888889]
 [0.52631579 0.47368421]]
expected:
 [[0.44444444 0.61538462]
 [0.66666667 0.47368421]]
Traceback (most recent call last):
  File "bnb_repro.py", line 19, in <module>
    BernoulliNB(alpha=np.ones(3)).fit(X3, y)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^
  File "/home/arthur/scikit-learn/sklearn/base.py", line 1403, in wrapper
    return fit_method(estimator, *args, **kwargs)
  File "/home/arthur/scikit-learn/sklearn/naive_bayes.py", line 786, in fit
    self._update_feature_log_prob(alpha)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^
  File "/home/arthur/scikit-learn/sklearn/naive_bayes.py", line 1221, in _update_feature_log_prob
    smoothed_cc = self.class_count_ + alpha * 2
                  ~~~~~~~~~~~~~~~~~~^~~~~~~~~~~
ValueError: operands could not be broadcast together with shapes (2,) (3,) 
```

## Root cause and possible fix

`sklearn/naive_bayes.py:1221-1225`:

```python
smoothed_fc = self.feature_count_ + alpha
smoothed_cc = self.class_count_ + alpha * 2
self.feature_log_prob_ = np.log(smoothed_fc) - np.log(smoothed_cc.reshape(-1, 1))
```

Array `alpha` support was added to the discrete NB base class (`_check_alpha`) without updating this formula, which assumes a scalar. Fix:

```python
smoothed_cc = self.class_count_[:, np.newaxis] + 2 * np.asarray(alpha)
self.feature_log_prob_ = np.log(smoothed_fc) - np.log(smoothed_cc)
```

which works for both scalar and per-feature `alpha`. A test comparing against the closed form with an array `alpha` and `n_classes != n_features` would cover it (`test_alpha_vector` only tests `MultinomialNB`).

## Versions

```
sklearn: 1.10.dev0, commit 289c4d5cfa (upstream/main; local HEAD f570497251 has no diff in sklearn/naive_bayes.py)
python: 3.13, numpy: 2.5.3, scipy: 1.18.1, platform: linux
```
