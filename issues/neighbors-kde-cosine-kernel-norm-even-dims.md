# KernelDensity with kernel="cosine" is wrongly normalized in even dimensions (NaN in 4D)

```json
{
    "estimators": ["KernelDensity", "KDTree.kernel_density", "BallTree.kernel_density"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 1,
    "difficulty": 1,
    "decision": 0,
    "decision_tags": []
}
```

## Classification

- **Priority (1)**: the cosine kernel is not the default and the error is a constant factor per dimension, so the shape of the density, bandwidth selection within the kernel and density ratios at fixed `d` are unaffected; only absolute densities and cross-kernel log-likelihood comparisons are wrong, and the 4D case is a loud NaN. (priority review, 2026-10-23: was 2)
- **Difficulty (1)**: the closed-form normalization in `_log_kernel_norm` misses one term for even `d`; a fixed recursion of a few lines fixes it, and quadrature gives an easy test.
- **Decision (0)**: correctness fix for a normalization constant, no trade-off.

## Upstream search

- `KernelDensity cosine`: #21224 (open, `sample()` for more kernels), #3117 and #1732 (closed, original implementations). Not related.
- `kernel_norm cosine`: no match.
- `KernelDensity normalization dimension`: #8327 (closed, about documenting that log densities are small in high dimension). Different topic.

## Describe the bug and its user-facing impact

`KernelDensity(kernel="cosine").score_samples` returns densities that don't integrate to 1 when the number of features is even. With 2 features the density is ~2.75x too small, with 4 features `score_samples` returns NaN, with 6 features it is ~220x too small. Odd dimensions (1, 3, 5, 7) are correct.

Users comparing kernels by log-likelihood (e.g. `GridSearchCV` over `kernel`, which uses `KernelDensity.score`) get a cosine kernel unfairly penalized in 2D, and NaN scores in 4D. `KDTree.kernel_density` and `BallTree.kernel_density` share the bug.

## Steps/Code to Reproduce

```python
import numpy as np
from sklearn.neighbors import KernelDensity

# The density of a single point should integrate to 1 over the plane.
grid = np.linspace(-1.5, 1.5, 601)
step = grid[1] - grid[0]
XX = np.stack(np.meshgrid(grid, grid), axis=-1).reshape(-1, 2)
for kernel in ["epanechnikov", "cosine"]:
    kde = KernelDensity(kernel=kernel, bandwidth=1.0).fit(np.zeros((1, 2)))
    print(kernel, np.exp(kde.score_samples(XX)).sum() * step**2)

# In 4 dimensions, the cosine kernel gives NaN
kde = KernelDensity(kernel="cosine").fit(np.zeros((1, 4)))
print(kde.score_samples(np.zeros((1, 4))))
```

## Expected Results

Both integrals close to 1, and a finite log-density in 4D.

## Actual Results

```
epanechnikov 0.9999995086600661
cosine 0.3633800762028652
[nan]
```

Comparing the private `kernel_norm(h=1, d, "cosine")` with a quadrature of `1 / (S_{d-1} * int_0^1 r^(d-1) cos(pi r / 2) dr)`:

```
d  sklearn                 quadrature           ratio
1  0.7853981633974484      0.7853981633974484   1.0
2  0.25                    0.6879845984710273   0.36338022763241856
3  0.6598725106858611      0.6598725106858604   1.000000000000001
4  nan                     0.6838594400571402   nan
5  0.7581659350259452      0.7581659350259536   0.9999999999999889
6  0.004019089808648445    0.8922440900816407   0.00450447344322639
7  1.1077373269643944      1.1077373269642237   1.000000000000154
```

## Root cause and possible fix

`sklearn/neighbors/_binary_tree.pxi.tp:489-496` computes `I_n = int_0^1 r^n cos(a r) dr` (with `a = pi/2`, `n = d - 1`) by repeated integration by parts:

```python
factor = 0
tmp = 2. / PI
for k in range(1, d + 1, 2):
    factor += tmp
    tmp *= -(d - k) * (d - k - 1) * (2. / PI) ** 2
```

This is the recursion `I_n = 1/a - n (n - 1) / a^2 * I_{n-2}`, which ends at `I_0 = 1/a` for odd `d`. For even `d` it ends at `I_1 = 1/a - 1/a^2`, but the loop uses `1/a` (the `-1/a^2` term is dropped because the multiplier `(d - k - 1)` is 0 on the last step). For `d = 2`: sklearn uses `2/pi` instead of `2/pi - 4/pi^2`, hence the factor `0.2313 / 0.6366 = 0.363`. In 4D the truncated sum is negative and its log is NaN.

Fix: run the recursion from the right base case (checked against `quad` for `d` in 1..8):

```python
# I_n = int_0^1 r^n cos(a r) dr with a = pi / 2, n = d - 1
a = PI / 2
if (d - 1) % 2 == 0:
    I, start = 1 / a, 2               # I_0
else:
    I, start = 1 / a - 1 / a**2, 3    # I_1
for n in range(start, d, 2):
    I = 1 / a - n * (n - 1) / a**2 * I
factor = log(I) + logSn(d - 1)
```

The existing tests (`test_kde.py`, `test_neighbors_tree.py`) compare against a slow implementation that calls the same `kernel_norm`, with `n_features=3`, so they can't catch it. A test integrating the density numerically in 2D, or comparing `kernel_norm` to `scipy.integrate.quad` for `d` in 1..6, would.

## Versions

```
sklearn: 1.10.dev0, commit f570497251 (no diff in sklearn/neighbors vs upstream/main 289c4d5cfa)
python: 3.13, numpy: 2.5.3, scipy: 1.18.1, platform: linux
```
