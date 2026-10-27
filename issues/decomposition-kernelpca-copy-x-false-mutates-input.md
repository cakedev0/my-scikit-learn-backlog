# KernelPCA(kernel="precomputed", copy_X=False).fit centers the user's kernel in place, which the docstring doesn't say

```json
{
    "estimators": ["KernelPCA"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 1,
    "difficulty": 1,
    "decision": 0,
    "decision_tags": []
}
```

## Classification

- **Priority (1)**: silent mutation of user data, but only with the non-default `copy_X=False` and `kernel="precomputed"`, and the in-place behavior is intended (#29100); what's wrong is the documentation, which describes `copy_X=False` as "store a reference" and `X_fit_` as "the data used to fit the model".
- **Difficulty (1)**: docstring change for `copy_X` and `X_fit_`.
- **Decision (0)**: documenting the current behavior has no trade-off.

## Upstream search

Queries: "KernelPCA copy_X", "KernelPCA precomputed inplace". The in-place centering was introduced on purpose in #29100 (merged, "Fix Make centering inplace in KernelPCA", 1.6): @jeremiedbb's point there is that `copy_X=False` means the estimator may modify `X`. The PR did not update the `copy_X` or `X_fit_` docstrings, and no issue reports the mismatch.

## Describe the bug and its user-facing impact

The `copy_X` docstring says:

> If True, input X is copied and stored by the model in the `X_fit_` attribute. If no further changes will be done to X, setting `copy_X=False` saves memory by storing a reference.

and `X_fit_`:

> The data used to fit the model. If `copy_X=False`, then `X_fit_` is a reference.

This reads as "the model keeps a reference to `X`, so don't modify `X` afterwards". But with `kernel="precomputed"`, `fit` itself double-centers the user's kernel matrix in place. A user who passes `copy_X=False` to avoid copying a large Gram matrix, as the docstring suggests, and then reuses that matrix (e.g. for `SVC(kernel="precomputed")`, or another `KernelPCA` fit) silently works with a different, centered kernel. `X_fit_` also holds the centered kernel, not "the data used to fit".

With other kernels, `X` is not modified (`K` is a new array), so the behavior also differs between `kernel="precomputed"` and the rest.

## Steps/Code to Reproduce

```python
import numpy as np
from sklearn.decomposition import KernelPCA
from sklearn.metrics.pairwise import rbf_kernel

rng = np.random.RandomState(0)
X = rng.rand(50, 3)
K = rbf_kernel(X)
K_orig = K.copy()

kpca = KernelPCA(n_components=2, kernel="precomputed", copy_X=False).fit(K)
print("K changed by fit:", not np.allclose(K, K_orig))
print("max |K - K_orig|:", np.abs(K - K_orig).max())
print("K_orig row means:", K_orig.mean(axis=1)[:3])
print("K row means after fit:", K.mean(axis=1)[:3])
```

## Expected Results

Either `K` is left untouched, or the docstring says that `copy_X=False` lets `fit` overwrite `X` (as for `copy_X` in the linear models: "If False, X may be overwritten").

## Actual Results

```
K changed by fit: True
max |K - K_orig|: 0.9815390906679771
K_orig row means: [0.9102183  0.92170479 0.83452292]
K row means after fit: [-2.66453526e-17 -1.08801856e-16 -1.95399252e-16]
```

## Root cause and possible fix

`sklearn/decomposition/_kernel_pca.py:430-438` (commit below): `validate_data(..., copy=self.copy_X)` returns the user's array when `copy_X=False`, `_get_kernel` returns it unchanged for `kernel="precomputed"`, and `_fit_transform_in_place` centers it with `KernelCenterer().transform(K, copy=False)`. Line 444 then stores it as `X_fit_`.

Since the in-place centering was a deliberate memory saving, I'd update the docstrings:

- `copy_X`: "If False, `X` is not copied: it is stored by reference in `X_fit_` and, when `kernel="precomputed"`, centered in place during `fit`, so it is modified."
- `X_fit_`: say that for `kernel="precomputed"` it holds the centered training kernel.

See also `issues/manifold-isomap-redundant-dense-copies.md`: `Isomap` could use this in-place path for its private kernel.

## Versions

```
sklearn: 1.10.dev0, commit 289c4d5cfa (upstream/main; local checkout f570497251, no diff on this file)
python: 3.13.14, numpy: 2.5.3, scipy: 1.18.1, platform: linux
```
