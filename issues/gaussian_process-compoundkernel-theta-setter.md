# `CompoundKernel.theta` setter always raises `AttributeError` (breaks `clone_with_theta`)

```json
{
    "estimators": ["CompoundKernel", "GaussianProcessClassifier"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 1,
    "difficulty": 1,
    "decision": 0,
    "decision_tags": []
}
```

## Classification

- **Priority (1)**: loud crash in a public kernel class, but `CompoundKernel` is mostly used internally (as `GaussianProcessClassifier.kernel_` in the multi-class case) and maintainers have said users should not build it themselves (#21973). Users only hit it when manipulating the fitted `gpc.kernel_`.
- **Difficulty (1)**: the setter references a non-existent attribute; split `theta` according to each sub-kernel's `n_dims`.
- **Decision (0)**: the code path can never have worked.

## Upstream search

GraphQL searches "CompoundKernel theta", "CompoundKernel k1", "GaussianProcessClassifier one_vs_one log_marginal_likelihood".

- #21973 (closed): "AttributeError of compound kernel of Gaussian processes", a user using `CompoundKernel` in `GaussianProcessRegressor`; closed as usage error ("only used in the internal of scikit-learn"). Different error (it was not about the `theta` setter).
- #16010 (open, WIP kernel cookbook): unrelated.

## Describe the bug and its user-facing impact

`CompoundKernel` is public (`sklearn.gaussian_process.kernels.CompoundKernel`) and is what `GaussianProcessClassifier.kernel_` returns for multi-class problems. Its `theta` setter reads `self.k1.n_dims`, but `CompoundKernel` stores its sub-kernels in `self.kernels` and has no `k1` attribute (that attribute belongs to `KernelOperator`). So every assignment to `theta` raises, and with it `clone_with_theta`, the standard way to evaluate a kernel at other hyperparameters.

The setter also assumes all sub-kernels have the same number of hyperparameters, which is not true in general (e.g. different `fixed` bounds per sub-kernel).

## Steps/Code to Reproduce

```python
from sklearn.datasets import load_iris
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF

X, y = load_iris(return_X_y=True)
gpc = GaussianProcessClassifier(RBF(1.0), optimizer=None).fit(X, y)
kernel = gpc.kernel_  # CompoundKernel, one RBF per one-vs-rest classifier
print(kernel, kernel.theta)
kernel.clone_with_theta(kernel.theta)
```

## Expected Results

`clone_with_theta` returns a copy of the compound kernel with the given hyperparameters, as for any other kernel.

## Actual Results

```
CompoundKernel(0, 0, 0) [0. 0. 0.]
Traceback (most recent call last):
  File "ck.py", line 9, in <module>
    kernel.clone_with_theta(kernel.theta)
    ~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^
  File "/home/arthur/scikit-learn/sklearn/gaussian_process/kernels.py", line 268, in clone_with_theta
    cloned.theta = theta
    ^^^^^^^^^^^^
  File "/home/arthur/scikit-learn/sklearn/gaussian_process/kernels.py", line 573, in theta
    k_dims = self.k1.n_dims
             ^^^^^^^
AttributeError: 'CompoundKernel' object has no attribute 'k1'
```

## Root cause and possible fix

`sklearn/gaussian_process/kernels.py:573-575`, unchanged since `CompoundKernel` was added (bf0e290a38):

```python
k_dims = self.k1.n_dims
for i, kernel in enumerate(self.kernels):
    kernel.theta = theta[i * k_dims : (i + 1) * k_dims]
```

Fix:

```python
start = 0
for kernel in self.kernels:
    kernel.theta = theta[start : start + kernel.n_dims]
    start += kernel.n_dims
```

which matches the getter (`np.hstack([kernel.theta for kernel in self.kernels])`). No test covers the setter.

## Versions

```
sklearn: 1.10.dev0, commit 289c4d5cfa (upstream/main; local HEAD f570497251 has no diff in sklearn/gaussian_process/)
python: 3.13, numpy: 2.5.3, scipy: 1.18.1, platform: linux
```
