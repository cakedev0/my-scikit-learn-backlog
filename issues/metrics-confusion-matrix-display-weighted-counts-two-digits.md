# `ConfusionMatrixDisplay` shows weighted counts with 2 significant digits (`252` becomes `2.5e+02`)

```json
{
    "estimators": ["ConfusionMatrixDisplay"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 1,
    "difficulty": 1,
    "decision": 1,
    "decision_tags": ["output-change"]
}
```

## Classification

- **Priority (1)**: display only, and `values_format` is a direct workaround. But passing `sample_weight`, even all ones, turns exact counts into 2-significant-digit scientific notation, which loses information and looks like a different matrix.
- **Difficulty (1)**: a few lines in `plot` to pick a better default format for float matrices.
- **Decision (1)**: changes the default text of existing plots with float matrices; the exact rule (integers detection, number of digits) is a small judgment call.

## Upstream search

Queries (`gh search issues --include-prs`): "ConfusionMatrixDisplay sample_weight scientific notation", "ConfusionMatrixDisplay values_format float", "confusion matrix display weighted counts rounded"; `gh issue list --search` "ConfusionMatrixDisplay format", "plot_confusion_matrix scientific".

- #16127 (closed, fixed by #16159): large integer counts were shown in scientific notation; the fix added "'d' or '.2g', whichever is shorter", which only applies to integer dtypes.
- #23478 (closed): unrelated normalize issue.

No report about float (weighted) counts.

## Describe the bug and its user-facing impact

With `values_format=None`, `ConfusionMatrixDisplay.plot` formats each cell with `'.2g'`, and only tries `'d'` when the matrix has an integer dtype. `confusion_matrix(..., sample_weight=...)` always returns a float matrix, so any weighted confusion matrix with counts above 99 is displayed with 2 significant digits in scientific notation. This happens even with unit weights, where the matrix holds exact integers, and with typical class weights where cells are in the hundreds or thousands.

The normalized case (values in `[0, 1]`) is fine with `'.2g'`.

## Steps/Code to Reproduce

```python
import matplotlib
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay

matplotlib.use("Agg")
rng = np.random.RandomState(0)
y_true, y_pred = rng.randint(0, 2, 1000), rng.randint(0, 2, 1000)

disp = ConfusionMatrixDisplay.from_predictions(y_true, y_pred)
print("no weights:  ", [t.get_text() for t in disp.text_.ravel()])
disp = ConfusionMatrixDisplay.from_predictions(
    y_true, y_pred, sample_weight=np.ones(1000)
)
print("unit weights:", [t.get_text() for t in disp.text_.ravel()])
```

## Expected Results

The same text in both cases (`252`, `244`, ...), and in general a readable default for weighted counts.

## Actual Results

```
no weights:   ['252', '244', '233', '271']
unit weights: ['2.5e+02', '2.4e+02', '2.3e+02', '2.7e+02']
```

## Root cause and possible fix

`sklearn/metrics/_plot/confusion_matrix.py:169-173` (commit below):

```python
text_cm = format(cm[i, j], ".2g")
if cm.dtype.kind != "f":
    text_d = format(cm[i, j], "d")
    if len(text_d) < len(text_cm):
        text_cm = text_d
```

Possible fix: when the float matrix only holds integral values, format with `'.0f'` (or `'d'` after casting); otherwise compare `'.2g'` with a fixed-point format such as `'.1f'` or `'.2f'` and keep the shorter one that does not lose the integer part. `'.2g'` should stay the default for normalized matrices.

## Versions

```
sklearn: 1.10.dev0, commit 289c4d5cfa
python: 3.13.14, numpy: 2.5.3, scipy: 1.18.1, platform: linux
```
