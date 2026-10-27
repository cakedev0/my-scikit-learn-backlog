# `LatentDirichletAllocation` results depend on `n_jobs`, and are not reproducible with the threading backend

```json
{
    "estimators": ["LatentDirichletAllocation"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 1,
    "difficulty": 2,
    "decision": 1,
    "decision_tags": ["rng-order", "output-change"]
}
```

## Classification

- **Priority (1)**: only with `n_jobs > 1`; the fits stay valid LDA solutions and are reproducible for a fixed `n_jobs` under the default loky backend. Only the dependence on `n_jobs` and non-reproducibility under an explicit threading backend remain, a reproducibility nuisance rather than wrong results (priority review, 2026-10-23: was 2)
- **Difficulty (2)**: generate the random initialization in the parent process (for all documents at once, or one seed per chunk) and pass it to the workers. Small change, but it needs care to keep the online/batch paths consistent.
- **Decision (1)**: changes the fitted model for a given `random_state` (rng-order, output-change). If the init is drawn for all documents in the parent, results become independent of `n_jobs`, which is what users expect.

## Upstream search

Queries (issues and PRs): "LatentDirichletAllocation n_jobs", "LatentDirichletAllocation random_state reproducible". No match. #33499 (closed) only touched the `n_jobs` docstring.

## Describe the bug and its user-facing impact

During `fit`/`partial_fit`, the E-step initializes the document-topic distribution randomly with `self.random_state_`, and runs one job per chunk of documents with joblib. The same `RandomState` object is passed to every job:

- with the default loky backend, each worker gets a pickled copy of the same state. All chunks draw the same random numbers (document `k` of every chunk gets the same init), and the parent's `random_state_` never advances, so every EM iteration reuses the same draws. The result depends on how many chunks there are, i.e. on `n_jobs`;
- with the threading backend, the jobs share one `RandomState` and draw from it in whatever order the threads run, so two fits with the same `random_state` give different topics.

With `n_jobs=1` everything is drawn sequentially from one stream. scikit-learn usually makes results independent of `n_jobs` (e.g. `KMeans`, forests), and the `random_state` docstring says "Pass an int for reproducible results across multiple function calls".

## Steps/Code to Reproduce

```python
import numpy as np
from joblib import parallel_backend
from sklearn.decomposition import LatentDirichletAllocation

X = np.random.RandomState(0).poisson(1.0, size=(2000, 30))


def fit(n_jobs):
    lda = LatentDirichletAllocation(
        n_components=5, learning_method="batch", max_iter=3, n_jobs=n_jobs, random_state=0
    )
    return lda.fit(X).components_


ref = fit(n_jobs=1)
print("max components_ value:", ref.max())
print("n_jobs=2 vs n_jobs=1:", np.abs(fit(n_jobs=2) - ref).max())
print("n_jobs=4 vs n_jobs=2:", np.abs(fit(n_jobs=4) - fit(n_jobs=2)).max())

with parallel_backend("threading"):
    a, b = fit(n_jobs=4), fit(n_jobs=4)
print("threading, n_jobs=4, two identical fits:", np.abs(a - b).max())
```

## Expected Results

The same `components_` (up to float rounding) for any `n_jobs`, and identical results for two identical fits with `random_state=0`.

## Actual Results

```
max components_ value: 944.0314467109544
n_jobs=2 vs n_jobs=1: 37.00291226733003
n_jobs=4 vs n_jobs=2: 38.71958684880042
threading, n_jobs=4, two identical fits: 34.40324375185617
```

The last number changes from run to run (I got 68.0 and 45.3 on other runs).

## Root cause and possible fix

`sklearn/decomposition/_lda.py:447-463` (`_e_step`) passes `random_state = self.random_state_` unchanged to every `_update_doc_distribution` job, and `_update_doc_distribution` draws `random_state.gamma(100.0, 0.01, (n_samples, n_topics))` for its chunk (`_lda.py:93-96`).

Possible fix: when `random_init=True`, draw the initial `doc_topic_distr` for the whole batch in the parent (`self.random_state_.gamma(100.0, 0.01, (X.shape[0], n_topics))`) and pass each job its slice. This makes the init independent of `n_jobs` and of the backend, and keeps the parent's RNG advancing across iterations. It changes the results for a given `random_state` with `n_jobs > 1` (and possibly with `n_jobs=1`, depending on how the draws are ordered).

## Versions

```
sklearn: 1.10.dev0, commit f570497251 (_lda.py identical to upstream/main 289c4d5cfa)
python: 3.13.14, numpy: 2.5.3, scipy: 1.18.1, joblib: 1.6.0, platform: linux
```
