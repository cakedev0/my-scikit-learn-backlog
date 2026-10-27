# GradientBoosting early stopping uses stale validation predictions after a warm start

```json
{
    "estimators": ["GradientBoostingClassifier", "GradientBoostingRegressor"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 1,
    "difficulty": 1,
    "decision": 1,
    "decision_tags": ["output-change"]
}
```

## Classification

- **Priority (1)**: silent, but limited to the legacy `GradientBoosting*` with both `warm_start` and `n_iter_no_change`, and the effect is mild: the continuation runs up to `begin_at_stage` extra stages past the validation plateau, so the model overfits slightly rather than being wrong. (priority review, 2026-10-23: was 2)
- **Difficulty (1)**: skip the first `begin_at_stage` values of the staged generator (or start it at `begin_at_stage`).
- **Decision (1)**: `output-change`: warm-start fits with early stopping will stop at a different iteration than today.

## Upstream search

GraphQL search on scikit-learn/scikit-learn: "GradientBoosting warm_start n_iter_no_change", "gradient boosting warm_start early stopping validation", "_staged_raw_predict warm_start".

- #14034 (open) "data leak in GBDT due to warm start": with `random_state=None`, each warm-start `fit` draws a different train/validation split. Different cause; the bug below happens with a fixed `random_state`.
- #16661 (closed): HistGradientBoosting early stopping with warm start, different estimator and cause.

No match for this bug.

## Describe the bug and its user-facing impact

In `_fit_stages`, the validation predictions come from `self._staged_raw_predict(X_val)`, a generator that starts at stage 0. The boosting loop starts at `begin_at_stage`, which is the number of already fitted stages on a warm start. Each `next(y_val_pred_iter)` therefore returns the prediction of stage `i - begin_at_stage` instead of stage `i`. The early-stopping test compares losses that lag behind the model by `begin_at_stage` stages.

In the reproducer below, a single fit stops after 204 stages. Fitting 50 stages and then continuing with warm start (same `random_state`, so same split and same trees) gives 254 stages: the second fit runs 204 more stages because it replays the validation curve from stage 0. Forcing the generator to skip the first 50 values gives 204, as expected.

## Steps/Code to Reproduce

```python
from sklearn.datasets import make_regression
from sklearn.ensemble import GradientBoostingRegressor

X, y = make_regression(n_samples=1000, n_features=10, noise=30, random_state=0)
params = dict(n_iter_no_change=5, validation_fraction=0.2, random_state=0)

ref = GradientBoostingRegressor(n_estimators=1000, **params).fit(X, y)
print("single fit:             n_estimators_ =", ref.n_estimators_)

ws = GradientBoostingRegressor(n_estimators=50, warm_start=True, **params).fit(X, y)
ws.set_params(n_estimators=1000).fit(X, y)
print("warm start (50 + more): n_estimators_ =", ws.n_estimators_)
```

## Expected Results

Both print 204: the warm-start run fits the same trees on the same split, so it should stop at the same stage.

## Actual Results

```
single fit:             n_estimators_ = 204
warm start (50 + more): n_estimators_ = 254
```

## Root cause and possible fix

`sklearn/ensemble/_gb.py:845` creates the generator:

```python
y_val_pred_iter = self._staged_raw_predict(X_val, check_input=False)
```

and `sklearn/ensemble/_gb.py:929` consumes one value per stage of the loop `for i in range(begin_at_stage, self.n_estimators)`, while `_staged_raw_predict` (`sklearn/ensemble/_gb.py:995`) yields from stage 0. Fix: advance the generator by `begin_at_stage` before the loop (e.g. `itertools.islice(..., begin_at_stage, None)`), or compute the initial validation raw predictions with `_raw_predict` and then update them stage by stage. A test comparing a single fit and a warm-start continuation with a fixed `random_state` would cover it.

## Versions

```
sklearn: 1.10.dev0, commit f570497251 (sklearn/ensemble/_gb.py identical to upstream/main 289c4d5cfa)
python: 3.13, numpy: 2.5.3, scipy: 1.18.1, platform: linux
```
