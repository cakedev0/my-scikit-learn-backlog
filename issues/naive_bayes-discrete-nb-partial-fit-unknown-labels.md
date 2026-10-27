# Discrete naive Bayes `partial_fit` silently accepts labels outside `classes` (counted as the first class in the binary case)

```json
{
    "estimators": ["MultinomialNB", "BernoulliNB", "ComplementNB", "CategoricalNB"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 2,
    "difficulty": 1,
    "decision": 1,
    "decision_tags": ["new-error"]
}
```

## Classification

- **Priority (2)**: silent wrong model on invalid but easy-to-produce input (a label typo or a class that only shows up in a later batch of a stream). In the binary case the samples are counted as the first class, which biases the model; with more classes they are dropped. `GaussianNB` and `SGDClassifier` raise a clear error in the same situation.
- **Difficulty (1)**: copy the check `GaussianNB._partial_fit` already does.
- **Decision (1)**: code that currently passes unknown labels would start raising, but that code was getting wrong counts.

## Upstream search

Queries (issues and PRs): "MultinomialNB partial_fit classes", "naive bayes partial_fit unseen label", "partial_fit label not in classes". Related but different:

- #11759 (open): about the error message of `SGDClassifier.partial_fit` when labels are outside `classes`; SGD does raise.
- #13232 (closed): priors when some declared classes are never observed, the opposite situation.

## Describe the bug and its user-facing impact

`partial_fit` of `MultinomialNB`, `BernoulliNB`, `ComplementNB` and `CategoricalNB` does not check that `y` only contains labels from `classes_`. It one-hot encodes `y` with `label_binarize(y, classes=self.classes_)`, which encodes unknown labels as all-zero rows:

- with 2 classes, `label_binarize` returns a single column and the code builds the other one as `1 - Y`, so samples with an unknown label are counted as `classes_[0]`;
- with 3 or more classes, they are silently ignored.

This happens on the first call too (when `y` has labels not listed in `classes`). `GaussianNB.partial_fit` raises `ValueError: The target label(s) [...] in y do not exist in the initial classes [...]`, and `SGDClassifier`, `Perceptron` and `MLPClassifier` raise as well.

## Steps/Code to Reproduce

```python
import numpy as np
from sklearn.naive_bayes import GaussianNB, MultinomialNB

X = np.array([[5, 0], [4, 1], [0, 5], [1, 4]])
y = np.array(["spam", "spam", "ham", "ham"])

clf = MultinomialNB().partial_fit(X, y, classes=["ham", "spam"])
print("class_count_ after 1st batch:", clf.class_count_)
# 2nd batch with a typo in the labels: "Spam" is not one of the declared classes
clf.partial_fit([[0, 9], [0, 9]], ["Spam", "Spam"])
print("class_count_ after 2nd batch:", clf.class_count_)

GaussianNB().partial_fit(X, y, classes=["ham", "spam"]).partial_fit(
    [[0, 9], [0, 9]], ["Spam", "Spam"]
)
```

## Expected Results

A `ValueError` for `MultinomialNB`, like for `GaussianNB`. The `classes` docstring says it is the "List of all the classes that can possibly appear in the y vector".

## Actual Results

```
class_count_ after 1st batch: [2. 2.]
class_count_ after 2nd batch: [4. 2.]
Traceback (most recent call last):
  ...
ValueError: The target label(s) ['Spam'] in y do not exist in the initial classes ['ham' 'spam']
```

The two "Spam" samples were added to the "ham" class of `MultinomialNB` (and to its `feature_count_`). Only `GaussianNB` raises.

## Root cause and possible fix

`sklearn/naive_bayes.py:702-707` (`_BaseDiscreteNB.partial_fit`):

```python
Y = label_binarize(y, classes=self.classes_)
if Y.shape[1] == 1:
    if len(self.classes_) == 2:
        Y = np.concatenate((1 - Y, Y), axis=1)
```

There is no check that `np.unique(y)` is a subset of `self.classes_`. Adding the same check as `GaussianNB._partial_fit` (`sklearn/naive_bayes.py:485-492`) before `label_binarize` fixes all four estimators, since `CategoricalNB` inherits this `partial_fit`.

## Versions

```
sklearn: 1.10.dev0, commit f570497251 (sklearn/naive_bayes.py identical to upstream/main 289c4d5cfa)
python: 3.13.14, numpy: 2.5.3, scipy: 1.18.1, platform: linux
```
