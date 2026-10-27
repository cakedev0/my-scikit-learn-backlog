# Boolean class labels break array API classifiers and classification metrics

```json
{
    "estimators": ["LogisticRegression", "LinearDiscriminantAnalysis", "RidgeClassifier", "RidgeClassifierCV", "GaussianNB", "LabelEncoder", "LabelBinarizer", "matthews_corrcoef", "log_loss"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 2,
    "difficulty": 2,
    "decision": 0,
    "decision_tags": []
}
```

## Classification

- **Priority (2)**: a boolean tensor is a natural binary target (`y = t > 0`), it works with NumPy everywhere, and with torch it fails in most documented array API classifiers with misleading messages (`y contains previously unseen labels: tensor([False,  True])`, a matmul shape error in LDA). Loud, and casting `y` to an integer dtype works around it.
- **Difficulty (2)**: two separate root causes in shared helpers (`_encode` and `xpx.isin` on bool), plus the array API standard does not allow `sort`/`searchsorted` on bool at all, so the clean fix is probably to encode bool labels as integers in one place while keeping `classes_` boolean.
- **Decision (0)**: plain fix.

## Upstream search

`gh issue list/pr list --search` (GraphQL): "boolean labels array API", "isin bool torch", "previously unseen labels torch"; grepped the 319 issues/PRs labeled "Array API" for `bool`. Nothing about boolean targets. #34142 (merged) added a common test for string `y` with array API `X`, not for boolean `y`.

## Describe the bug and its user-facing impact

With `array_api_dispatch=True` and a boolean torch target, `fit` of `LogisticRegression`, `LinearDiscriminantAnalysis`, `RidgeClassifier`, `RidgeClassifierCV` and `GaussianNB` raises, as do `LabelEncoder().fit(y).transform(y)`, `LabelBinarizer`, `matthews_corrcoef` and `log_loss`. The same boolean `y` as a NumPy array works. The error messages don't point to the dtype: LDA fails deep in `_solve_svd` with a shape mismatch, `LogisticRegression` says the labels it just saw are unknown.

## Steps/Code to Reproduce

```python
import numpy as np
import torch
import sklearn
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import log_loss, matthews_corrcoef
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder

rng = np.random.RandomState(0)
X = rng.randn(40, 3)
y = X[:, 0] > 0  # boolean binary target

with sklearn.config_context(array_api_dispatch=True):
    X_t, y_t = torch.asarray(X), torch.asarray(y)
    for clf in [LogisticRegression(), LinearDiscriminantAnalysis(), RidgeClassifier(solver="svd"), GaussianNB()]:
        clf.fit(X, y)  # NumPy bool y works
        try:
            clf.fit(X_t, y_t)
            print(f"{type(clf).__name__}: OK")
        except Exception as e:
            print(f"{type(clf).__name__}: {type(e).__name__}: {e}")
    for name, f in [
        ("LabelEncoder.transform", lambda: LabelEncoder().fit(y_t).transform(y_t)),
        ("matthews_corrcoef", lambda: matthews_corrcoef(y_t, ~y_t)),
        ("log_loss", lambda: log_loss(y_t, torch.rand(40, dtype=torch.float64))),
    ]:
        try:
            f()
            print(f"{name}: OK")
        except Exception as e:
            print(f"{name}: {type(e).__name__}: {e}")
```

## Expected Results

Everything works as with the NumPy boolean `y`.

## Actual Results

```
LogisticRegression: ValueError: y contains previously unseen labels: tensor([False,  True])
LinearDiscriminantAnalysis: RuntimeError: mat1 and mat2 shapes cannot be multiplied (1x1 and 2x3)
RidgeClassifier: RuntimeError: Unsupported input type encountered for isin(): Bool
GaussianNB: RuntimeError: Unsupported input type encountered for isin(): Bool
LabelEncoder.transform: ValueError: y contains previously unseen labels: tensor([False,  True])
matthews_corrcoef: ValueError: y contains previously unseen labels: tensor([False,  True])
log_loss: RuntimeError: Unsupported input type encountered for isin(): Bool
```

With `array_api_strict`, the same calls fail with `Only real numeric dtypes are allowed in sort/searchsorted`, `object of type 'Array' has no len()`, and also `f1_score`, `jaccard_score`, `roc_curve`, `average_precision_score`, `brier_score_loss` and `precision_recall_curve` fail on type promotion between bool and int.

## Root cause and possible fix

Two root causes for torch:

1. `_encode` (`sklearn/utils/_encode.py:285-287`) uses `xp.isdtype(values.dtype, "numeric")` to choose the vectorized path. Bool is not "numeric", so it goes through `_map_to_integer`, a Python dict lookup over the elements. That works for NumPy scalars but not for 0-d tensors, which hash by identity, so every value is "unknown" (`-1`). This breaks `LabelEncoder.transform` (used by `LogisticRegression` and `matthews_corrcoef`) and LDA's `_encode(y, uniques=self.classes_)`, where all samples land in one class (hence the 1x1 priors).
2. `xpx.isin` dispatches to `torch.isin`, which does not support bool inputs. It is called by `label_binarize` (`LabelBinarizer`, `RidgeClassifier*`, `log_loss`) and `GaussianNB._partial_fit`.

Since the array API standard also forbids `sort`/`searchsorted` on bool, I think the simplest fix is to convert a boolean `y` to an integer dtype in the shared label validation (and in `_encode`/`label_binarize`), while keeping `classes_` boolean so that `predict` still returns booleans.

## Versions

```
sklearn: 1.10.dev0, commit f570497251 (utils/_encode.py, preprocessing/_label.py, naive_bayes.py, discriminant_analysis.py identical to upstream/main 289c4d5cfa)
python: 3.13.14, numpy: 2.5.3, scipy: 1.18.1, torch: 2.14.1+xpu, array_api_strict: 2.6.1, platform: linux
```
