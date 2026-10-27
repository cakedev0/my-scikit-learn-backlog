# BernoulliRBM under-weights the negative phase when a batch has more samples than `batch_size`

```json
{
    "estimators": ["BernoulliRBM"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 1,
    "difficulty": 1,
    "decision": 1,
    "decision_tags": ["output-change"]
}
```

## Classification

- **Priority (1)**: silent wrong gradient, strong for `partial_fit` on chunks larger than `batch_size` (the negative phase becomes negligible), mild for `fit` (only the last, smaller batch is affected). `BernoulliRBM` is a niche estimator.
- **Difficulty (1)**: divide the positive and negative statistics by their own number of rows in `_fit`.
- **Decision (1)**: `output-change`: `fit` results change when `n_samples` is not a multiple of `batch_size`, and `partial_fit` results change whenever `X` does not have `batch_size` rows.

## Upstream search

Queries: "BernoulliRBM partial_fit", "BernoulliRBM batch_size", "BernoulliRBM gradient", "RBM gradient", PRs "BernoulliRBM".

No match. #2400 (closed, "Add public partial_fit to BernoulliRBM") added `partial_fit` but did not discuss the number of persistent chains vs the batch size. #31099 (closed PR, pseudo-likelihood normalization in `score_samples`) is about a different method.

## Describe the bug and its user-facing impact

`BernoulliRBM` trains with persistent contrastive divergence (PCD): the gradient of the mean log-likelihood is `mean over the data of v h^T` (positive phase) minus `mean over the persistent chains of v h^T` (negative phase). The number of chains is `batch_size` (`h_samples_` has `batch_size` rows), but `_fit` divides both sums by the number of data rows:

```python
lr = float(self.learning_rate) / v_pos.shape[0]
update = safe_sparse_dot(v_pos.T, h_pos, dense_output=True).T
update -= np.dot(h_neg.T, v_neg)   # sum over batch_size chains
```

So the negative phase gets a weight of `batch_size / n_rows` instead of 1:

- `partial_fit(X)` with `X` larger than `batch_size` (a natural way to call it on chunks of a stream) mostly ignores the negative phase: with 1000 rows and the default `batch_size=10`, it is 100x too small, and the update just pushes the visible biases and weights towards the data statistics.
- `fit` uses `gen_even_slices`, so when `n_samples` is not a multiple of `batch_size` the last batch is smaller (e.g. 10, 10, 5 for 25 samples) and its negative phase is over-weighted (2x here).

Nothing in the docstrings says `partial_fit` must be called with exactly `batch_size` rows; `test_partial_fit` happens to use batches of `batch_size`.

## Steps/Code to Reproduce

```python
import numpy as np
from sklearn.neural_network import BernoulliRBM

rng = np.random.RandomState(0)
x = (rng.rand(1, 6) > 0.5).astype(float)

# The same sample repeated 10 or 1000 times: the mean gradient of the
# log-likelihood is the same, so one update step should give the same model.
for n in [10, 1000]:
    rbm = BernoulliRBM(n_components=3, batch_size=10, random_state=0)
    rbm.partial_fit(np.repeat(x, n, axis=0))
    print(f"n={n:4d} intercept_visible_ =", np.round(rbm.intercept_visible_, 4))
```

## Expected Results

The same update for both calls: the data statistics are identical, and with the same `random_state` the negative samples are the same. In particular, the bias of the visible unit that is 0 in `x` (5th entry) should decrease by the same amount.

## Actual Results

```
n=  10 intercept_visible_ = [ 0.04  0.04  0.07  0.04 -0.07  0.02]
n=1000 intercept_visible_ = [ 0.0994  0.0994  0.0997  0.0994 -0.0007  0.0992]
```

With 1000 rows the negative phase is 100x smaller: the 5th bias barely moves (-0.0007 instead of -0.07) and the others are set to almost exactly `learning_rate * mean(x)`.

## Root cause and possible fix

`BernoulliRBM._fit`, `sklearn/neural_network/_rbm.py:330-337`. Use separate normalizations:

```python
lr = float(self.learning_rate)
n_pos, n_neg = v_pos.shape[0], v_neg.shape[0]
update = safe_sparse_dot(v_pos.T, h_pos, dense_output=True).T / n_pos
update -= np.dot(h_neg.T, v_neg) / n_neg
```

and the same for both intercepts. When every batch has exactly `batch_size` rows the results are unchanged, so only the cases above change.

## Versions

```
sklearn: 1.10.dev0, commit 289c4d5cfa (upstream/main; ran on f570497251, no diff in the files involved)
python: 3.13.14, numpy: 2.5.3, scipy: 1.18.1, platform: linux
```
