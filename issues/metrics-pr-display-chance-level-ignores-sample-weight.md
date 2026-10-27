# `PrecisionRecallDisplay` chance level ignores `sample_weight`

```json
{
    "estimators": ["PrecisionRecallDisplay"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 1,
    "difficulty": 1,
    "decision": 0,
    "decision_tags": []
}
```

## Classification

- **Priority (1)**: misleading baseline in a plot only, and it needs both the non-default `plot_chance_level=True` and `sample_weight`; the curve and AP themselves are right. (priority review, 2026-10-23: was 2)
- **Difficulty (1)**: replace the unweighted mean by a weighted one in
  `from_predictions` and `from_cv_results`.
- **Decision (0)**: no trade-off.

## Upstream search

Queries: "PrecisionRecallDisplay chance level sample_weight",
"PrecisionRecallDisplay plot_chance_level sample_weight", "prevalence
sample_weight display", "precision recall chance level weighted prevalence".

- #30717 (closed) and #29151 (merged) touch the display parameters and
  `drop_intermediate`, not the chance level.

No match.

## Describe the bug and its user-facing impact

The chance level of a precision-recall curve is the AP of a non-informative
classifier, i.e. the prevalence of the positive class. With sample weights,
the curve and AP use weighted counts, so the matching chance level is the
weighted prevalence `sum(w[y == pos]) / sum(w)`. `from_predictions` (hence
`from_estimator`) and `from_cv_results` compute the unweighted prevalence
instead, so the chance line is drawn at the wrong height and its legend shows
the wrong AP.

## Steps/Code to Reproduce

```python
import matplotlib
matplotlib.use("Agg")
import numpy as np
from sklearn.metrics import PrecisionRecallDisplay, average_precision_score

rng = np.random.RandomState(0)
y_true = rng.randint(0, 2, 100)
y_score = rng.rand(100)
# Positive samples get 5 times more weight
sample_weight = np.where(y_true == 1, 5.0, 1.0)

disp = PrecisionRecallDisplay.from_predictions(
    y_true, y_score, sample_weight=sample_weight, plot_chance_level=True
)
# AP of a non-informative (constant) classifier under the same weights
chance_ap = average_precision_score(
    y_true, np.zeros_like(y_score), sample_weight=sample_weight
)
print("display prevalence_pos_label:", disp.prevalence_pos_label)
print("AP of constant scores:       ", chance_ap)
print("chance line label:", disp.chance_level_.get_label())
```

## Expected Results

`prevalence_pos_label` and the chance line at the weighted prevalence,
`0.864` (= AP of constant scores with the same weights).

## Actual Results

```
display prevalence_pos_label: 0.56
AP of constant scores:        0.8641975308641975
chance line label: Chance level (AP = 0.56)
```

## Root cause and possible fix

`sklearn/metrics/_plot/precision_recall_curve.py:714` computes
`(y_true == pos_label).sum() / len(y_true)`, and
`sklearn/metrics/_plot/precision_recall_curve.py:900-902` does the same per
fold in `from_cv_results`, both ignoring `sample_weight` (resp.
`sample_weight_fold`) that they pass to `precision_recall_curve` and
`average_precision_score` a few lines above.

Fix: `np.average(y_true == pos_label, weights=sample_weight)` in both places,
and a test comparing `prevalence_pos_label` with
`average_precision_score` of constant scores.

## Versions

```
sklearn: 1.10.dev0, commit 289c4d5cfa (upstream/main; sklearn/metrics/ sources identical to the tested checkout)
python: 3.13, numpy: 2.5.3, scipy: 1.18.1, platform: linux
```
