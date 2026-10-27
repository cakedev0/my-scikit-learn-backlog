# Setting `TfidfVectorizer.idf_` with a wrong length raises `TypeError` instead of the intended `ValueError`

```json
{
    "estimators": ["TfidfVectorizer"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 1,
    "difficulty": 1,
    "decision": 0,
    "decision_tags": []
}
```

## Classification

- **Priority (1)**: only the error path is broken, and only when the vocabulary was learned by `fit` (no `vocabulary` parameter). Users get a confusing `TypeError` about `NoneType` instead of the length mismatch message.
- **Difficulty (1)**: use `len(self.vocabulary_)` in the message.
- **Decision (0)**: plain fix.

## Upstream search

Queries: `gh search issues --include-prs` "TfidfVectorizer idf_ setter". Found #7102 (closed, "Setting idf_ is impossible", which added the setter) and #21832 (merged, moved the setter logic so `__init__` does not assign). Neither mentions this error path. The existing test `test_tfidfvectorizer_invalid_idf_attr` only covers a vectorizer built with `vocabulary=...`, where `self.vocabulary` is not `None`.

## Describe the bug and its user-facing impact

The `idf_` setter of `TfidfVectorizer` checks that the new value matches the vocabulary size, but its error message calls `len(self.vocabulary)` (the constructor parameter) instead of `len(self.vocabulary_)` (the fitted attribute). On a vectorizer fitted without an explicit `vocabulary`, the parameter is `None`, so building the message raises `TypeError: object of type 'NoneType' has no len()`.

## Steps/Code to Reproduce

```python
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

vec = TfidfVectorizer().fit(["the cat sat", "the dog ran"])
vec.idf_ = np.ones(3)  # wrong length: the vocabulary has 5 terms
```

## Expected Results

`ValueError: idf length = 3 must be equal to vocabulary size = 5`

## Actual Results

```
Traceback (most recent call last):
  File "idf.py", line 5, in <module>
    vec.idf_ = np.ones(3)  # wrong length: the vocabulary has 5 terms
    ^^^^^^^^
  File "/home/arthur/scikit-learn/sklearn/feature_extraction/text.py", line 2027, in idf_
    % (len(value), len(self.vocabulary))
                   ~~~^^^^^^^^^^^^^^^^^
TypeError: object of type 'NoneType' has no len()
```

## Root cause and possible fix

`sklearn/feature_extraction/text.py:2027` (commit below): replace `len(self.vocabulary)` with `len(self.vocabulary_)`, which is the value the condition on the line above already uses.

## Versions

```
sklearn: 1.10.dev0, commit 289c4d5cfa
python: 3.13.14, numpy: 2.5.3, scipy: 1.18.1, platform: linux
```
