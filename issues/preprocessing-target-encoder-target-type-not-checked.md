# `TargetEncoder` does not check `target_type` against `y`: silent nonsense for `"binary"`, `IndexError` for `"multiclass"`

```json
{
    "estimators": ["TargetEncoder"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 1,
    "difficulty": 1,
    "decision": 1,
    "decision_tags": ["new-error"]
}
```

## Classification

- **Priority (1)**: needs an explicit `target_type` that does not match `y`, which is a user mistake, but the `"binary"` case gives silently meaningless encodings (target means of label-encoded classes 0, 1, 2) and the `"multiclass"` case an unhelpful `IndexError`.
- **Difficulty (1)**: check the number of classes after label encoding and raise a `ValueError`.
- **Decision (1)**: `new-error`: `target_type="binary"` with more than 2 classes currently "works".

## Upstream search

Queries: "TargetEncoder target_type binary multiclass", "TargetEncoder IndexError", "TargetEncoder target_type validation".

No match. #32247 (closed) is about `type_of_target` inferring `"multiclass"` for integer regression targets, which is why users set `target_type` by hand.

## Describe the bug and its user-facing impact

When `target_type` is set explicitly, `TargetEncoder` uses it without checking `y`:

- `target_type="binary"` with 3 classes: `y` is label-encoded to `0, 1, 2` and the encodings are means of these integers (values above 1, not probabilities). No error and no warning; `classes_` shows the 3 classes.
- `target_type="multiclass"` with 2 classes: `LabelBinarizer` returns a single column, and `fit` fails with `IndexError: index 1 is out of bounds for axis 1 with size 1`.

A user who sets `target_type="multiclass"` to get one column per class on a binary problem, or who reuses a `"binary"` configuration on a new target, does not get a useful message.

## Steps/Code to Reproduce

```python
import numpy as np
from sklearn.preprocessing import TargetEncoder

rng = np.random.RandomState(0)
X = rng.randint(0, 3, size=(30, 1))
y_multiclass = rng.randint(0, 3, size=30)
y_binary = rng.randint(0, 2, size=30)

enc = TargetEncoder(target_type="binary").fit(X, y_multiclass)
print(enc.classes_, enc.encodings_)

TargetEncoder(target_type="multiclass").fit(X, y_binary)
```

## Expected Results

A `ValueError` saying that `y` has 3 classes but `target_type="binary"`, and for the second call either a `ValueError` or a valid one-column-per-class encoding.

## Actual Results

```
[0 1 2] [array([0.60598904, 1.25314716, 0.8630231 ])]
Traceback (most recent call last):
  File ".../I.py", line 12, in <module>
    TargetEncoder(target_type="multiclass").fit(X, y_binary)
  File "/home/arthur/scikit-learn/sklearn/preprocessing/_target_encoder.py", line 281, in fit
    self._fit_encodings_all(X, y)
  File "/home/arthur/scikit-learn/sklearn/preprocessing/_target_encoder.py", line 519, in _fit_encodings_all
    encodings = self._fit_encoding_multiclass(
        X_ordinal,
    ...<2 lines>...
        self.target_mean_,
    )
  File "/home/arthur/scikit-learn/sklearn/preprocessing/_target_encoder.py", line 578, in _fit_encoding_multiclass
    y_class = y[:, i]
              ~^^^^^^
IndexError: index 1 is out of bounds for axis 1 with size 1
```

## Root cause and possible fix

`sklearn/preprocessing/_target_encoder.py:493-505` (same lines on upstream/main): `self.target_type_ = self.target_type` is used as is. For `"binary"`, `LabelEncoder` accepts any number of classes; for `"multiclass"`, `LabelBinarizer` returns one column when there are 2 classes, while `_fit_encoding_multiclass` loops over `len(self.classes_)` columns.

Fix: after encoding `y`, raise a `ValueError` if `target_type_ == "binary"` and `len(classes_) != 2`. For `"multiclass"` with 2 classes, either raise too, or build the two-column indicator (e.g. `np.hstack([1 - y, y])`) so that the output has one column per class as documented.

## Versions

```
sklearn: 1.10.dev0, commit f570497251 (_target_encoder.py differs from upstream/main 289c4d5cfa only in the routing of `groups`, unrelated)
python: 3.13.14, numpy: 2.5.3, scipy: 1.18.1, platform: linux
```
