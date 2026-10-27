# `metric_at_thresholds` compares `y_true` labels to 0/1 predictions, wrong for labels other than {0, 1}

```json
{
    "estimators": ["metric_at_thresholds"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 2,
    "difficulty": 2,
    "decision": 1,
    "decision_tags": ["api-change", "unclear-spec"]
}
```

## Classification

- **Priority (2)**: silent wrong values for binary targets coded `{-1, 1}` or
  `{1, 2}`, which the sibling curve functions (`roc_curve`,
  `precision_recall_curve`, `confusion_matrix_at_thresholds`) all accept.
  String labels raise a confusing "Mix of label input types" error. The
  function is new (1.9), so few users are affected yet.
- **Difficulty (2)**: map thresholded predictions back to the original
  labels (`_threshold_scores_to_class_labels` in `_scorer.py` does this), which
  needs a way to know the positive and negative labels.
- **Decision (1)**: the function has no `pos_label` parameter; fixing it
  properly means adding one (or inferring it like `_check_pos_label_consistency`
  and erroring on strings), and deciding how it interacts with a `pos_label`
  passed through `metric_params`.

## Upstream search

Queries: "metric_at_thresholds", "metric_at_thresholds pos_label",
"metric_at_thresholds labels".

- #32732 (merged): the PR adding `metric_at_thresholds` in 1.9. Its
  description says mapping thresholded labels back to the original classes
  "isn't really needed", but `y_true` is neither validated nor documented as
  `{0, 1}`, and all tests use `{0, 1}` targets.
- #21391 (open): request for a display plotting precision/recall/F1 against
  thresholds, a feature request, not this bug.
- #25639 (closed): earlier "metric per threshold" implementation.

No report of the label mismatch.

## Describe the bug and its user-facing impact

`metric_at_thresholds` builds `y_pred = (y_score >= threshold).astype(np.int32)`
and calls `metric_func(y_true, y_pred)` with the original `y_true`. Predictions
are always `{0, 1}`, so unless `y_true` is also coded `{0, 1}`, the labels of
`y_true` and `y_pred` don't match:

- `{-1, 1}`: the negative class `-1` never matches the prediction `0`, so
  every true negative counts as an error;
- `{1, 2}`: the positive class `2` never matches the prediction `1`, and the
  negative class `1` matches the *positive* prediction;
- strings: `ValueError: Mix of label input types (string and number)`.

`y_true` is not validated either (no binary check, unlike the other
threshold curve functions), and the docstring describes it as "Ground truth
(correct) target labels" with no restriction to `{0, 1}`.

## Steps/Code to Reproduce

```python
import numpy as np
from sklearn.metrics import accuracy_score, metric_at_thresholds

y_score = np.array([0.1, 0.4, 0.35, 0.8])
for y_true in ([0, 0, 1, 1], [-1, -1, 1, 1], [1, 1, 2, 2], ["neg", "neg", "pos", "pos"]):
    try:
        print(y_true, metric_at_thresholds(np.array(y_true), y_score, accuracy_score)[0])
    except ValueError as e:
        print(y_true, "ValueError:", e)
```

## Expected Results

The same accuracies for all four encodings of the same binary problem (the
docstring example gives `[0.75, 0.5, 0.75, 0.5]` for `{0, 1}`), or an
informative error asking for `pos_label`.

## Actual Results

```
[0, 0, 1, 1] [0.75 0.5  0.75 0.5 ]
[-1, -1, 1, 1] [0.25 0.25 0.5  0.5 ]
[1, 1, 2, 2] [0.   0.25 0.25 0.5 ]
['neg', 'neg', 'pos', 'pos'] ValueError: Mix of label input types (string and number); Got ['neg' 'pos'] and [0 1].
```

## Root cause and possible fix

`sklearn/metrics/_ranking.py:2343-2345` thresholds the scores into `0/1`
without mapping back to the labels of `y_true`. It also skips the target
type check and `pos_label` handling that `confusion_matrix_at_thresholds`
does (`sklearn/metrics/_ranking.py:1003-1014`).

Possible fix: add a `pos_label` parameter, resolve it with
`_check_pos_label_consistency` (errors for strings without `pos_label`, like
`roc_curve`), check that `y_true` is binary, and map `y_pred` to
`[neg_label, pos_label]` before calling `metric_func`, as
`_threshold_scores_to_class_labels` does for `TunedThresholdClassifierCV`
(`sklearn/metrics/_scorer.py:1054-1063`). A cheaper alternative is to
binarize `y_true` to `y_true == pos_label`, but then a `pos_label` passed in
`metric_params` (as in `test_metric_at_thresholds_pos_label`) would refer to
`{0, 1}` instead of the user's labels. Choosing between the two is the
decision point.

## Versions

```
sklearn: 1.10.dev0, commit 289c4d5cfa (upstream/main; sklearn/metrics/ sources identical to the tested checkout)
python: 3.13, numpy: 2.5.3, scipy: 1.18.1, platform: linux
```
