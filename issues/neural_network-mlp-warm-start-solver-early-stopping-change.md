# MLP warm start crashes when `solver` or `early_stopping` changed since the previous fit

```json
{
    "estimators": ["MLPClassifier", "MLPRegressor"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 1,
    "difficulty": 1,
    "decision": 0,
    "decision_tags": []
}
```

## Classification

- **Priority (1)**: a crash with an unhelpful message, on a combination that is plausible (L-BFGS pre-training then adam fine-tuning, or turning on early stopping for a continued fit) but uncommon. Workaround: fit from scratch.
- **Difficulty (1)**: move the solver/early-stopping dependent bookkeeping of `_initialize` to the start of `_fit_stochastic` when it is missing or inconsistent, a few lines.
- **Decision (0)**: no trade-off: this only turns crashes into working fits.

## Upstream search

Queries: "MLP warm_start solver", "MLP warm_start early_stopping", "MLP warm_start loss_curve_", "MLPClassifier warm_start solver".

No match. Related but different:
- #24683 (merged, "FIX always expose best_loss_, validation_scores_, and best_validation_score"): made these attributes always exist (set to `None` when unused), which is the `None` the warm-started fit then trips on; it did not consider a change of `early_stopping` between fits.
- #18269 (merged), #25443 (merged): `n_iter_`/`max_iter` with `warm_start`, different symptom.
- #6091 (closed, "using warm_start breaks adam algorithm?"), #16812 (closed, classifier vs regressor warm-start consistency): different symptoms.

## Describe the bug and its user-facing impact

With `warm_start=True`, `MLPClassifier`/`MLPRegressor` reuse the previous weights. `_initialize` also sets up the solver-specific state (`loss_curve_`, `_no_improvement_count`, `best_loss_`, `validation_scores_`, `best_validation_score_`), but it only runs on the first fit. If `solver` or `early_stopping` changed in between, the second `fit` crashes:

- `lbfgs` -> `sgd`/`adam`: `AttributeError: ... has no attribute 'loss_curve_'`. Same for `set_params(solver="sgd").partial_fit(...)` after an L-BFGS fit.
- `early_stopping=False` -> `True`: `AttributeError: 'NoneType' object has no attribute 'append'` (`validation_scores_` is `None`).
- `early_stopping=True` -> `False`: `TypeError: unsupported operand type(s) for -: 'NoneType' and 'float'` (`best_loss_` is `None`).

The docstring and glossary describe `warm_start` as reusing the previous solution as initialization; nothing says these parameters must stay fixed, and the errors do not point to them.

## Steps/Code to Reproduce

```python
import warnings
from sklearn.datasets import make_classification
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier

warnings.simplefilter("ignore", ConvergenceWarning)
X, y = make_classification(random_state=0)

for first, second in [
    ({"solver": "lbfgs"}, {"solver": "adam"}),
    ({"early_stopping": False}, {"early_stopping": True}),
    ({"early_stopping": True}, {"early_stopping": False}),
]:
    clf = MLPClassifier(warm_start=True, max_iter=20, random_state=0, **first)
    clf.fit(X, y).set_params(**second)
    try:
        clf.fit(X, y)
        print(first, "->", second, ": OK")
    except Exception as e:
        print(first, "->", second, f": {type(e).__name__}: {e}")
```

## Expected Results

The second fit starts from the previous weights and trains with the new solver / early-stopping setting (or, at least, raises an error saying these parameters cannot change under `warm_start`).

## Actual Results

```
{'solver': 'lbfgs'} -> {'solver': 'adam'} : AttributeError: 'MLPClassifier' object has no attribute 'loss_curve_'
{'early_stopping': False} -> {'early_stopping': True} : AttributeError: 'NoneType' object has no attribute 'append'
{'early_stopping': True} -> {'early_stopping': False} : TypeError: unsupported operand type(s) for -: 'NoneType' and 'float'
```

## Root cause and possible fix

`BaseMultilayerPerceptron._initialize` (`sklearn/neural_network/_multilayer_perceptron.py:431-441`) creates the stochastic-solver state depending on the current `solver` and `early_stopping`, and `_fit` only calls it when `first_pass` is true (`:471`). `_fit_stochastic` and `_update_no_improvement_count` then assume that state matches the current parameters.

Possible fix: split that block out of `_initialize` into a helper called at the start of `_fit_stochastic` whenever the state is missing or was built for the other `early_stopping` mode (e.g. `validation_scores_ is None` while `early_stopping=True`). Weights stay warm-started; only the bookkeeping is reset.

## Versions

```
sklearn: 1.10.dev0, commit 289c4d5cfa (upstream/main; ran on f570497251, no diff near the code involved)
python: 3.13.14, numpy: 2.5.3, scipy: 1.18.1, platform: linux
```
