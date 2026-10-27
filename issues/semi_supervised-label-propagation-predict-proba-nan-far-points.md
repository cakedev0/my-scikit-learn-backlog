# LabelPropagation/LabelSpreading predict_proba returns NaN for points far from the training data

```json
{
    "estimators": ["LabelPropagation", "LabelSpreading"],
    "existing_issue": "https://github.com/scikit-learn/scikit-learn/issues/34351",
    "reproduced": true,
    "priority": 2,
    "difficulty": 1,
    "decision": 1,
    "decision_tags": ["unclear-spec"]
}
```

## Classification

- **Priority (2)**: NaN probabilities and an arbitrary `classes_[0]` prediction, with only a numpy `RuntimeWarning`. With the default `gamma=20`, any query farther than ~6 (in data units) from every training point triggers it, which is common with unscaled features.
- **Difficulty (1)**: guard the zero normalizer in `predict_proba`, like `fit` already does.
- **Decision (1)**: `unclear-spec`: what to return for a point with no affinity to any training point (uniform probabilities as proposed in #34351, zeros as in `fit`, or an error/warning).

## Upstream search

- `LabelPropagation predict_proba nan`, `LabelSpreading nan probabilities`: #34351 (open issue, "Reconsider zero-affinity row fallback in LabelPropagation/LabelSpreading (fit + predict_proba)"), #34350 (open PR, guards the zero normalizer in `predict_proba`), #34835 (open PR, uniform probabilities for zero-affinity rows in `fit` and `predict_proba`, fixes #34351).

Filed for the backlog since the issue and both PRs are still open.

## Describe the bug and its user-facing impact

With `kernel="rbf"`, `predict_proba` weights the training label distributions by `exp(-gamma * ||x - x_i||^2)`. For a query far from all training points, all these weights underflow to 0, the row sum is 0, and `probabilities /= normalizer` gives NaN. `predict` then returns `classes_[np.argmax([nan, nan])]`, i.e. always the first class. `fit` guards the same division (`normalizer[normalizer == 0] = 1`), `predict_proba` doesn't.

## Steps/Code to Reproduce

```python
import numpy as np
from sklearn.semi_supervised import LabelPropagation

rng = np.random.RandomState(0)
X = rng.rand(30, 2)
y = rng.randint(2, size=30)
y[10:] = -1

model = LabelPropagation().fit(X, y)  # kernel="rbf", gamma=20
X_far = [[10.0, 10.0]]
print(model.predict_proba(X_far), model.predict(X_far))
```

## Expected Results

A valid probability distribution (e.g. uniform, or the class prior), or an explicit warning/error saying that the point has no affinity to the training data.

## Actual Results

```
/home/arthur/scikit-learn/sklearn/semi_supervised/_label_propagation.py:232: RuntimeWarning: invalid value encountered in divide
  probabilities /= normalizer
/home/arthur/scikit-learn/sklearn/semi_supervised/_label_propagation.py:232: RuntimeWarning: invalid value encountered in divide
  probabilities /= normalizer
[[nan nan]] [0]
```

## Root cause and possible fix

`sklearn/semi_supervised/_label_propagation.py:231-232` divides by the row sums without guarding zeros. The open PRs #34350 and #34835 fix it in two different ways; #34351 is where the expected output should be decided.

## Versions

```
sklearn: 1.10.dev0, commit f570497251 (no diff in sklearn/semi_supervised vs upstream/main 289c4d5cfa)
python: 3.13, numpy: 2.5.3, scipy: 1.18.1, platform: linux
```
