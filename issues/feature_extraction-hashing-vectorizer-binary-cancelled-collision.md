# `HashingVectorizer(binary=True)` sets hashed features that cancel out to 1

```json
{
    "estimators": ["HashingVectorizer", "FeatureHasher"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 1,
    "difficulty": 1,
    "decision": 0,
    "decision_tags": []
}
```

## Classification

- **Priority (1)**: needs a hash collision between tokens of opposite signs in the same document, which is rare with the default `n_features=2**20` but common with small `n_features`. The effect on a model is small (one spurious feature per cancelled collision).
- **Difficulty (1)**: drop explicit zeros before `X.data.fill(1)`, or in `FeatureHasher.transform` after `sum_duplicates()`.
- **Decision (0)**: matches the docstring; the output for binary=False is unchanged except for the storage of zeros.

## Upstream search

Queries (`gh search issues --include-prs`): "HashingVectorizer binary alternate_sign", "HashingVectorizer binary sign", "HashingVectorizer binary negative values".

- #7513 / #7565 (closed / merged): hash collisions in `FeatureHasher`, introduced `alternate_sign`. The review mentions that binary counts don't preserve inner products, which covers the loss of signs with `binary=True` (expected), not the zero that becomes 1.
- #3637 (closed): stored zeros from `HashingVectorizer` break `TfidfTransformer(sublinear_tf=True)`. Same root (stored zeros from `FeatureHasher`), other symptom; tracked locally in `feature_extraction-tfidf-explicit-zeros.md`.

## Describe the bug and its user-facing impact

With `alternate_sign=True` (the default), two tokens that hash to the same column with opposite signs sum to 0. `FeatureHasher.transform` keeps that 0 as a stored entry after `sum_duplicates()`. `HashingVectorizer.transform` then applies `binary=True` with `X.data.fill(1)`, which turns the stored 0 into 1.

The docstring says "If True, all non zero counts are set to 1", so a column that is 0 with `binary=False` should stay 0 with `binary=True`. Currently `HashingVectorizer(binary=True)` and `HashingVectorizer() != 0` disagree.

## Steps/Code to Reproduce

```python
from sklearn.feature_extraction.text import HashingVectorizer

# "aa" and "ab" hash to the same column with opposite signs when n_features=4
doc = ["aa ab"]
counts = HashingVectorizer(n_features=4, norm=None).transform(doc)
binary = HashingVectorizer(n_features=4, norm=None, binary=True).transform(doc)
print("counts:", counts.toarray(), "stored entries:", counts.nnz)
print("binary:", binary.toarray())
```

## Expected Results

`binary: [[0. 0. 0. 0.]]`, since every count is zero.

## Actual Results

```
counts: [[0. 0. 0. 0.]] stored entries: 1
binary: [[0. 1. 0. 0.]]
```

## Root cause and possible fix

- `sklearn/feature_extraction/_hash.py:190`: `X.sum_duplicates()` keeps entries that sum to 0.
- `sklearn/feature_extraction/text.py:882-883`: `if self.binary: X.data.fill(1)` assumes all stored entries are non-zero.

Calling `X.eliminate_zeros()` after `sum_duplicates()` in `FeatureHasher.transform` fixes this and the stored zeros seen by downstream transformers (#3637). The cost is one pass over `X.data`, negligible next to hashing.

## Versions

```
sklearn: 1.10.dev0, commit 289c4d5cfa
python: 3.13.14, numpy: 2.5.3, scipy: 1.18.1, platform: linux
```
