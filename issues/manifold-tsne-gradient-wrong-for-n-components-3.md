# `TSNE` gradient is wrong when `n_components >= 3` (degrees of freedom != 1)

```json
{
    "estimators": ["TSNE"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 2,
    "difficulty": 2,
    "decision": 1,
    "decision_tags": ["output-change"]
}
```

## Classification

- **Priority (2)**: silent and plausible-looking: with `n_components=3` (a common choice for 3D plots) both the exact and the Barnes-Hut solvers follow a vector field that is not the gradient of the KL divergence they report, so the embedding is not a stationary point of the documented objective (KL 0.1965 vs 0.1875 reachable from the same embedding in the reproducer). The embedding still looks like a t-SNE embedding, which is why it went unnoticed; `n_components=2` (the default) is not affected.
- **Difficulty (2)**: the fix is a change of the attraction/repulsion kernel in `_kl_divergence` (one line) and in two Cython kernels of `_barnes_hut_tsne.pyx`, plus a `check_grad` test for `dof != 1`. The math is short but needs to be derived and checked carefully.
- **Decision (1)**: `output-change`: embeddings and `kl_divergence_` change for every `TSNE(n_components>=3)` fit.

## Upstream search

- `gh search issues --include-prs`: "TSNE gradient degrees_of_freedom", "tsne gradient wrong n_components 3", "t-SNE degrees of freedom".
- #9711 / #9712 (closed / merged, 2017): fixed the Student-t kernel itself (`(1 + d^2/dof)` instead of `(1 + d^2)/dof`) in the exact and Barnes-Hut code, but left the gradient kernel as is.
- #7885 (closed, not merged): "bug with running using n dfs" for the exact method; it was about argument passing and exposing `degrees_of_freedom`, not the gradient.
- #10610 (merged): skips the power when `dof == 1`; it shows the `dof != 1` path was known, but not checked against the gradient.

No match for this bug.

## Describe the bug and its user-facing impact

t-SNE uses a Student-t kernel with `dof = max(n_components - 1, 1)` degrees of freedom (van der Maaten 2009):
`w_ij = (1 + d_ij^2 / dof)^(-(dof + 1) / 2)`, `q_ij = w_ij / Z`.
Differentiating `KL(P || Q)` gives

`dC/dy_i = (2 dof + 2) / dof * sum_j (p_ij - q_ij) (1 + d_ij^2 / dof)^(-1) (y_i - y_j)`

The factor in the sum is `(1 + d^2/dof)^(-1)`, whatever `dof`. scikit-learn uses `w_ij = (1 + d^2/dof)^(-(dof + 1)/2)` instead. Both coincide only for `dof = 1`, i.e. `n_components <= 2`. For `n_components >= 3` (exact) and `n_components = 3` (Barnes-Hut, the default method, which caps `n_components` at 3), the optimizer follows a vector field that is not the gradient of the objective:

- the embedding returned is not a (local) minimum of the KL divergence reported in `kl_divergence_`: a few L-BFGS steps with the right gradient from `embedding_` lower it by ~5% in the reproducer;
- `min_grad_norm` and the progress checks are applied to the wrong vector.

Anyone running `TSNE(n_components=3)` is affected, silently.

## Steps/Code to Reproduce

Part 1 checks the private objective function, since the gradient is not exposed publicly; part 2 shows the public consequence on `TSNE.fit_transform`.

```python
import numpy as np
from scipy.optimize import check_grad, minimize
from scipy.spatial.distance import pdist, squareform
from sklearn.datasets import load_digits
from sklearn.manifold import TSNE
from sklearn.manifold._t_sne import _joint_probabilities, _kl_divergence
from sklearn.metrics import pairwise_distances

X = load_digits().data[:300]
n = X.shape[0]
P = _joint_probabilities(pairwise_distances(X, squared=True), 30, verbose=0)


def true_grad(params, dof, n_components):
    # dC/dy_i = (2 dof + 2) / dof * sum_j (p_ij - q_ij) (1 + d_ij^2 / dof)^-1 (y_i - y_j)
    Y = params.reshape(n, n_components)
    d2 = pdist(Y, "sqeuclidean")
    w = (1 + d2 / dof) ** (-(dof + 1) / 2)
    Q = w / (2 * w.sum())
    PQd = squareform((P - Q) / (1 + d2 / dof))
    g = np.array([PQd[i] @ (Y[i] - Y) for i in range(n)])
    return (2 * (dof + 1) / dof * g).ravel()


for n_components in [2, 3]:
    dof = max(n_components - 1, 1)  # what TSNE uses
    kl = lambda p: _kl_divergence(p, P, dof, n, n_components)[0]
    sk_grad = lambda p: _kl_divergence(p, P, dof, n, n_components)[1]
    my_grad = lambda p: true_grad(p, dof, n_components)

    # 1. Gradients vs finite differences at a random point
    p0 = np.random.RandomState(0).randn(n * n_components)
    print(f"n_components={n_components}: finite-difference error of sklearn grad "
          f"{check_grad(kl, sk_grad, p0):.1e}, of closed-form grad "
          f"{check_grad(kl, my_grad, p0):.1e} (|grad|={np.linalg.norm(my_grad(p0)):.1e})")

    # 2. The TSNE result is not a stationary point of its own objective
    tsne = TSNE(n_components=n_components, method="exact", random_state=0)
    Y = tsne.fit_transform(X).ravel().astype(np.float64)
    res = minimize(lambda p: (kl(p), my_grad(p)), Y, jac=True, method="L-BFGS-B")
    print(f"    kl_divergence_={tsne.kl_divergence_:.4f}, "
          f"|true grad| at embedding_={np.linalg.norm(my_grad(Y)):.1e}, "
          f"KL after L-BFGS from embedding_={res.fun:.4f}")
```

## Expected Results

The gradient returned with the objective should match finite differences for any `n_components`, as it does for `n_components=2`, and the `TSNE` result should be (close to) a stationary point of the KL divergence it reports.

## Actual Results

```
n_components=2: finite-difference error of sklearn grad 1.1e-06, of closed-form grad 1.1e-06 (|grad|=2.4e-02)
    kl_divergence_=0.2738, |true grad| at embedding_=6.9e-06, KL after L-BFGS from embedding_=0.2738
n_components=3: finite-difference error of sklearn grad 1.7e-02, of closed-form grad 1.5e-06 (|grad|=3.0e-02)
    kl_divergence_=0.1965, |true grad| at embedding_=3.3e-03, KL after L-BFGS from embedding_=0.1875
```

The sklearn gradient error (1.7e-02) is half the gradient norm (3.0e-02). The Barnes-Hut gradient (`_kl_divergence_bh` with `angle=0`) has the same error: relative error 0.55 against finite differences for `n_components=3`, 6e-07 for `n_components=2`.

## Root cause and possible fix

- `sklearn/manifold/_t_sne.py:193` (commit below): `PQd = squareform((P - Q) * dist)` where `dist` holds `w_ij = (1 + d^2/dof)^(-(dof+1)/2)` (lines 176-178). It should be `(P - Q) / (1 + d^2/dof)`. The constant `2 (dof + 1) / dof` (line 197) is right.
- `sklearn/manifold/_barnes_hut_tsne.pyx`: in `compute_gradient_positive` (lines 142-145) the attractive force uses `pij * qij` with `qij = (dof / (dof + d))^((dof+1)/2)`; it should use `pij * dof / (dof + d)`. In `compute_gradient_negative` (lines 217-221) the repulsive force uses `size * qijZ * qijZ`; it should be `size * qijZ * dof / (dof + d)`. `sum_Q` and the error term should keep `qijZ`.
- `test_gradient` in `sklearn/manifold/tests/test_t_sne.py` only checks `alpha = 1.0`; `test_barnes_hut_angle` runs `n_components=3` but only compares the KL values, not the gradients. Adding `dof=2` to `test_gradient` and a BH-vs-exact gradient comparison would have caught it.

For `dof = 1` all changes are no-ops, so the default 2D case is unchanged.

## Versions

```
sklearn: 1.10.dev0, commit 289c4d5cfa
python: 3.13.14, numpy: 2.5.3, scipy: 1.18.1, platform: linux
```
