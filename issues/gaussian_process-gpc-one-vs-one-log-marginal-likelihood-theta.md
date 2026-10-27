# `GaussianProcessClassifier.log_marginal_likelihood` rejects `kernel_.theta` with `multi_class="one_vs_one"` and 4+ classes

```json
{
    "estimators": ["GaussianProcessClassifier"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 1,
    "difficulty": 1,
    "decision": 0,
    "decision_tags": []
}
```

## Classification

- **Priority (1)**: loud error on a documented input, in a rarely used combination (`one_vs_one`, 4+ classes, explicit `theta`). The default `one_vs_rest` and the `theta=None` call work.
- **Difficulty (1)**: use the number of fitted binary estimators instead of `n_classes` when splitting `theta`.
- **Decision (0)**: the affected inputs currently raise.

## Upstream search

GraphQL searches "GaussianProcessClassifier one_vs_one log_marginal_likelihood", "CompoundKernel theta", "CompoundKernel k1". No match (#21973, closed, is about using `CompoundKernel` in a regressor).

## Describe the bug and its user-facing impact

For multi-class problems, `GaussianProcessClassifier.log_marginal_likelihood(theta)` documents that "theta may be the hyperparameters of the compound kernel or of an individual kernel", and `kernel_` is a `CompoundKernel` with one sub-kernel per binary classifier. The method assumes there are `n_classes` binary classifiers, which is only true for `one_vs_rest`. With `multi_class="one_vs_one"` there are `n_classes * (n_classes - 1) / 2` of them, so for 4 or more classes:

- `log_marginal_likelihood(gpc.kernel_.theta)` raises "Shape of theta must be either 1 or 4. Obtained theta with shape 6.";
- passing `n_dims * n_classes` values, as the error message suggests, slices past the end of `theta` for the extra estimators and crashes with `IndexError: index 0 is out of bounds for axis 0 with size 0`.

(With 3 classes both strategies have 3 binary classifiers, so it happens to work.)

## Steps/Code to Reproduce

```python
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF

X, y = make_blobs(n_samples=40, centers=4, random_state=0)
gpc = GaussianProcessClassifier(RBF(1.0), multi_class="one_vs_one").fit(X, y)
print("n binary classifiers:", len(gpc.base_estimator_.estimators_))
print("kernel_.theta shape:", gpc.kernel_.theta.shape)
print("log_marginal_likelihood_value_:", gpc.log_marginal_likelihood_value_)
gpc.log_marginal_likelihood(gpc.kernel_.theta)
```

## Expected Results

`gpc.log_marginal_likelihood(gpc.kernel_.theta)` returns `gpc.log_marginal_likelihood_value_` (the mean over the 6 binary classifiers), as it does with `one_vs_rest`.

## Actual Results

```
n binary classifiers: 6
kernel_.theta shape: (6,)
log_marginal_likelihood_value_: -9.158604585056533
Traceback (most recent call last):
  File "gpc_ovo2.py", line 11, in <module>
    gpc.log_marginal_likelihood(gpc.kernel_.theta)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^
  File "/home/arthur/scikit-learn/sklearn/gaussian_process/_gpc.py", line 924, in log_marginal_likelihood
    raise ValueError(
    ...<3 lines>...
    )
ValueError: Shape of theta must be either 1 or 4. Obtained theta with shape 6.
```

## Root cause and possible fix

`sklearn/gaussian_process/_gpc.py:912-928` compares `theta.shape[0]` with `n_dims * self.classes_.shape[0]` and builds the error message from it. Replace `self.classes_.shape[0]` by `len(estimators)` (the number of binary classifiers) in both places. The docstring ("the mean log-marginal likelihood of the one-versus-rest classifiers") could also mention one-vs-one.

## Versions

```
sklearn: 1.10.dev0, commit 289c4d5cfa (upstream/main; local HEAD f570497251 has no diff in sklearn/gaussian_process/)
python: 3.13, numpy: 2.5.3, scipy: 1.18.1, platform: linux
```
