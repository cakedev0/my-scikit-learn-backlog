# Multi-metric scoring reuses cached predict_proba column across scorers with different pos_label

```json
{
    "estimators": ["cross_validate", "GridSearchCV", "RandomizedSearchCV", "make_scorer", "check_scoring"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 3,
    "difficulty": 1,
    "decision": 0,
    "decision_tags": []
}
```

## Classification

- **Priority (3)**: silent wrong scores in `cross_validate` / `*SearchCV` with
  only built-in scorer names (e.g. `["neg_brier_score", "average_precision"]`)
  on a binary target coded `{1, 2}`, for any classifier without
  `decision_function` (random forests, HGB, naive Bayes, KNN). The wrong
  values look plausible and can drive model selection.
- **Difficulty (1)**: include `pos_label` in the cache key of `_cached_call`
  (a few lines).
- **Decision (0)**: no trade-off, the cache is only an optimization.

## Upstream search

Queries: "MultimetricScorer cache pos_label", "multimetric scorer pos_label cache",
"scoring list pos_label wrong predict_proba", "_cached_call pos_label",
"multiple scorers pos_label different results".

- #35156 (closed, 2026-10-18): "Multimetric scoring mixes up scorers that
  differ only in pos_label". Same root cause, reported on 1.9.1 with two
  custom `average_precision` scorers (`pos_label=0` and `pos_label=1`). It was
  closed by a maintainer the day after it was opened, with no comment, no
  linked PR and no commit: `sklearn/metrics/_scorer.py` is unchanged on
  `upstream/main` (last change #34444) and the bug still reproduces there. I
  read the closing as triage of the report (new account, "Needs Triage"), not
  as a decision that the behavior is intended. The new argument here: built-in
  scorer strings alone (`["neg_brier_score", "average_precision"]` on a `{1, 2}`
  target with any classifier lacking `decision_function`) trigger it, so it
  is not limited to users writing custom `pos_label` scorers. The report also
  points to #26037 (1.3) as the change that moved the `pos_label` selection
  inside the cached call.
- #26840 (merged): `response_method` in `make_scorer`, unrelated.

## Describe the bug and its user-facing impact

When several scorers are evaluated together (`scoring` as a list or dict in
`cross_validate`, `GridSearchCV`, etc.), `_MultimetricScorer` caches the
estimator response per response method name. But for a binary classifier the
response also depends on `pos_label`: `predict_proba` is reduced to the
column of `pos_label`, and `decision_function` is negated when `pos_label` is
`classes_[0]`. The cache key ignores `pos_label`, so the second scorer gets
the response computed for the first scorer's positive class.

This happens as soon as two scorers using the same response method have a
different `pos_label`:

- custom scorers, e.g. `average_precision_score` for each of the two classes
  (`pos_label=0` and `pos_label=1`);
- built-in scorers only: `neg_brier_score` uses `pos_label=None`
  (`classes_[-1]`) while `average_precision` uses `pos_label=1`. With classes
  `{1, 2}` these are different columns.

Each scorer gives the right value when used alone, so the bug only shows when
comparing single- vs multi-metric runs. The results depend on the order of the
scorers.

## Steps/Code to Reproduce

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_validate
from sklearn.naive_bayes import GaussianNB

X, y = make_classification(n_samples=200, weights=[0.7], random_state=0)
y = y + 1  # classes are {1, 2}
clf = GaussianNB()  # no decision_function: both scorers use predict_proba

for scoring in (["average_precision"], ["neg_brier_score"],
                ["neg_brier_score", "average_precision"],
                ["average_precision", "neg_brier_score"]):
    res = cross_validate(clf, X, y, scoring=scoring)
    print(scoring, {s: round(res[f"test_{s}"].mean(), 4) for s in scoring})
```

## Expected Results

Each metric has the same value whether it is computed alone or together with
other metrics: `average_precision` ~0.9726 and `neg_brier_score` ~-0.0656.

## Actual Results

```
['average_precision'] {'average_precision': np.float64(0.9726)}
['neg_brier_score'] {'neg_brier_score': np.float64(-0.0656)}
['neg_brier_score', 'average_precision'] {'neg_brier_score': np.float64(-0.0656), 'average_precision': np.float64(0.5094)}
['average_precision', 'neg_brier_score'] {'average_precision': np.float64(0.9726), 'neg_brier_score': np.float64(-0.8738)}
```

The same happens with `{0, 1}` targets and two custom scorers, e.g.
`make_scorer(average_precision_score, response_method="predict_proba",
pos_label=0)` next to the same scorer with `pos_label=1` (LogisticRegression,
5-fold: 0.9775 alone vs 0.5053 together for `pos_label=0`).

## Root cause and possible fix

`sklearn/metrics/_scorer.py:92-104` (`_cached_call`) stores and looks up
`cache[response_method]`, but the value returned by `_get_response_values`
also depends on the `pos_label` keyword passed by `_Scorer._score`
(`sklearn/metrics/_scorer.py:459-465`). `_MultimetricScorer._use_cache`
(`sklearn/metrics/_scorer.py:195-214`) enables the cache whenever two scorers
resolve to the same method name.

Fix: key the cache on `(response_method, pos_label)`, or cache the raw
estimator output and apply the `pos_label` column selection / sign flip after
the lookup (keeps a single `predict_proba` call). A non-regression test in
`test_score_objects.py` comparing single- and multi-metric scores with two
`pos_label` values would cover it.

## Versions

```
sklearn: 1.10.dev0, commit 289c4d5cfa (upstream/main; sklearn/metrics/ sources identical to the tested checkout)
python: 3.13, numpy: 2.5.3, scipy: 1.18.1, platform: linux
```
