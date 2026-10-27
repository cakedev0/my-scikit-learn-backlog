# `GaussianMixture` docstring says `n_init` is ignored with `warm_start=True`, but the first fit runs `n_init` initializations

```json
{
    "estimators": ["GaussianMixture", "BayesianGaussianMixture"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 1,
    "difficulty": 1,
    "decision": 0,
    "decision_tags": []
}
```

## Classification

- **Priority (1)**: documentation does not match the code; the code's behavior is the sensible one, the only cost is a first fit `n_init` times slower than the docstring announces.
- **Difficulty (1)**: rewrite two docstring sentences.
- **Decision (0)**: doc-only fix, no behavior change.

## Upstream search

Queries: "GaussianMixture warm_start n_init". Hits: #10870 (merged, `lower_bound_` with `n_init > 1`), unrelated. The code and docstring both date from the 2016 integration (`0625e0384e`).

## Describe the bug and its user-facing impact

The `warm_start` docstring of `GaussianMixture` says: "In that case, 'n_init' is ignored and only a single initialization occurs upon the first call." `BaseMixture.fit` says the same ("If `warm_start` is `True`, then `n_init` is ignored and a single initialization is performed upon the first call").

The code only ignores `n_init` on subsequent calls: `do_init = not (self.warm_start and hasattr(self, "converged_"))` is `True` on the first call, so `n_init` initializations are run and the best one is kept. Then later calls continue from it with a single run.

## Steps/Code to Reproduce

```python
import numpy as np
from sklearn.mixture import GaussianMixture

X = np.random.RandomState(0).randn(100, 2)
gm = GaussianMixture(n_components=2, n_init=3, warm_start=True, verbose=1, random_state=0)
gm.fit(X)
```

## Expected Results

Per the docstring, a single initialization on the first call.

## Actual Results

```
Initialization 0
Initialization converged.
Initialization 1
Initialization converged.
Initialization 2
Initialization converged.
```

## Root cause and possible fix

`sklearn/mixture/_base.py:239-240`:

```python
do_init = not (self.warm_start and hasattr(self, "converged_"))
n_init = self.n_init if do_init else 1
```

I think the code is right (picking the best of `n_init` starts before warm-starting is useful) and the docstrings should change: "`n_init` is only used on the first call; subsequent calls start from the previous solution with a single run." in `sklearn/mixture/_gaussian_mixture.py:648` and `sklearn/mixture/_base.py:180` (`fit` docstring, also inherited by `BayesianGaussianMixture`).

## Versions

```
sklearn: 1.10.dev0, commit 289c4d5cfa
python: 3.13.14, numpy: 2.5.3, scipy: 1.18.1, platform: linux
```
