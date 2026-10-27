# Nystroem(kernel="precomputed") fails unless n_components equals n_samples

```json
{
    "estimators": ["Nystroem"],
    "existing_issue": "https://github.com/scikit-learn/scikit-learn/issues/29353",
    "reproduced": true,
    "priority": 1,
    "difficulty": 2,
    "decision": 0,
    "decision_tags": []
}
```

## Classification

- **Priority (1)**: loud error, but on a documented option with default `n_components`: any precomputed kernel with more than 100 samples fails, i.e. the only case where Nystroem is useful (a low-rank approximation) does not work with `kernel="precomputed"`.
- **Difficulty (2)**: index the precomputed matrix on both axes in `fit`, and select the component columns in `transform`; a few lines, but the expected shape of `X` in `transform` must be decided and documented.
- **Decision (0)**: nothing that works today changes (the `n_components == n_samples` case gives the same Gram matrix).

## Upstream search

Queries: "Nystroem precomputed".

- #29353 (open, "kernel_approximation.Nystroem with precomputed kernel"): this exact bug. @jeremiedbb suggests `X[np.ix_(basis_inds, basis_inds)]` in `fit`, and notes that `transform` can then only be used on the training kernel unless it receives the kernel against the training samples.
- #34195 (open PR, "FIX Nystroem with precomputed kernel and n_components < n_samples"): attempts the fix; not merged at the commit below. #34196 (closed PR) was another attempt.
- #19178 (closed as duplicate of #29353), #33417 (closed docs-only PR documenting the `n_components == n_samples` constraint, not merged: the user guide still advertises precomputed kernels without restriction).

## Describe the bug and its user-facing impact

The user guide says: "By default Nystroem uses the rbf kernel, but it can use any kernel function or a precomputed kernel matrix. The number of samples used [...] is given by the parameter n_components." With `kernel="precomputed"`, `fit(K)` fails as soon as `n_components < n_samples`, with an error about the shape of a matrix the user never passed:

```
ValueError: Precomputed metric requires shape (n_queries, n_indexed). Got (100, 150) for 100 indexed.
```

`fit` selects the basis as rows of `X` (`basis = X[basis_inds]`) and calls `pairwise_kernels(basis, metric="precomputed")`, which just returns `basis`, a `(n_components, n_samples)` slice of `K`, instead of the `(n_components, n_components)` block `K[basis_inds][:, basis_inds]`. When `n_components == n_samples` (the only case tested, `test_nystroem_precomputed_kernel`), the slice is square and the result is a valid (row-permuted) `K^{-1/2}`, so the bug went unnoticed; but that case is an exact feature map, not an approximation.

## Steps/Code to Reproduce

```python
import numpy as np
from sklearn.kernel_approximation import Nystroem
from sklearn.metrics.pairwise import rbf_kernel

X = np.random.RandomState(0).randn(150, 3)
K = rbf_kernel(X)

Nystroem(kernel="rbf", random_state=0).fit_transform(X)  # works
Nystroem(kernel="precomputed", random_state=0).fit_transform(K)  # n_components=100 < 150
```

## Expected Results

The second call returns the same features as the first one (same `random_state`, so the same basis samples): `fit` should use the `(n_components, n_components)` block of `K` on the sampled indices, and `transform(K_test)` with `K_test` of shape `(n_test, n_train)` should use the columns `component_indices_`.

## Actual Results

```
Traceback (most recent call last):
  File "nys_final.py", line 9, in <module>
    Nystroem(kernel="precomputed", random_state=0).fit_transform(K)  # n_components=100 < 150
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^
  File "/home/arthur/scikit-learn/sklearn/utils/_set_output.py", line 321, in wrapped
    data_to_wrap = f(self, X, *args, **kwargs)
  File "/home/arthur/scikit-learn/sklearn/base.py", line 969, in fit_transform
    return self.fit(X, **fit_params).transform(X)
           ~~~~~~~~^^^^^^^^^^^^^^^^^
  File "/home/arthur/scikit-learn/sklearn/base.py", line 1403, in wrapper
    return fit_method(estimator, *args, **kwargs)
  File "/home/arthur/scikit-learn/sklearn/kernel_approximation.py", line 1041, in fit
    basis_kernel = pairwise_kernels(
        basis,
    ...<3 lines>...
        **self._get_kernel_params(),
    )
  File "/home/arthur/scikit-learn/sklearn/utils/_param_validation.py", line 191, in wrapper
    return func(*args, **kwargs)
  File "/home/arthur/scikit-learn/sklearn/metrics/pairwise.py", line 2638, in pairwise_kernels
    X, _ = check_pairwise_arrays(X, Y, precomputed=True)
           ~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/arthur/scikit-learn/sklearn/metrics/pairwise.py", line 202, in check_pairwise_arrays
    raise ValueError(
    ...<3 lines>...
    )
ValueError: Precomputed metric requires shape (n_queries, n_indexed). Got (100, 150) for 100 indexed.
```

## Root cause and possible fix

`Nystroem.fit`, `sklearn/kernel_approximation.py:1035-1046`, and `Nystroem.transform`, `:1076-1083`, on the commit below. Both compute the kernel with `pairwise_kernels(..., metric=self.kernel)`, which is a no-op for `"precomputed"`.

Fix: for `kernel="precomputed"`, use `basis_kernel = X[basis_inds][:, basis_inds]` in `fit` (and store `components_` as the rows `X[basis_inds]`, or nothing), and `embedded = X[:, self.component_indices_]` in `transform`, where `X` is the kernel between the new samples and the training samples, shape `(n_test, n_train)`, as for other estimators with `kernel="precomputed"`. Document that shape in the `transform` docstring.

## Versions

```
sklearn: 1.10.dev0, commit 289c4d5cfa (upstream/main; ran on f570497251, no diff in the files involved)
python: 3.13.14, numpy: 2.5.3, scipy: 1.18.1, platform: linux
```
