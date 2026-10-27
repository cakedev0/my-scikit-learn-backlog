# `export_text` prints a meaningless `class:` and `weights:` for multi-output classification trees

```json
{
    "estimators": ["export_text", "DecisionTreeClassifier", "ExtraTreeClassifier"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 1,
    "difficulty": 1,
    "decision": 1,
    "decision_tags": ["output-change"]
}
```

## Classification

- **Priority (1)**: silently wrong text, but only for multi-output classification trees, which are rarely exported. `export_graphviz`/`plot_tree` show the per-output values correctly and can be used instead.
- **Difficulty (1)**: in the multi-output case, print the predicted class of each output (or skip the class like `export_graphviz` does) and the full weights. A few lines in `print_tree_recurse` / `_add_leaf`.
- **Decision (1)**: `output-change`: the text of `export_text` changes for multi-output classifiers; anyone parsing it would see a different format (unlikely to matter since the current output is not usable).

## Upstream search

Queries: `export_text multi-output`, `export_text multioutput classifier class`. Related: #25387 (merged, added `class_names` to `export_text`, documented as "not supported for multi-output"), #20576 (closed, feature request for that). Neither mentions the multi-output leaf output. No match.

## Describe the bug and its user-facing impact

For a `DecisionTreeClassifier` fitted on a 2D `y`, `tree_.value[node]` has shape `(n_outputs, n_classes)`. `export_text` takes `tree_.value[node].T[0]`, i.e. the proportion of class 0 for each output, and then prints `class: argmax(...)`, which is the index of the output with the highest proportion of class 0. Neither the "class" nor the "weights" have anything to do with the prediction of the leaf. In the reproducer, the two leaves predict `[0, 0]` and `[1, 1]` but both are printed as `class: 0`.

`export_graphviz` and `plot_tree` only print a class for single-output trees and show the full value array otherwise, so `export_text` is inconsistent with them.

## Steps/Code to Reproduce

```python
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text

X = np.array([[0.0], [1.0], [2.0], [3.0]])
Y = np.array([[0, 0], [0, 0], [1, 1], [1, 1]])
clf = DecisionTreeClassifier(random_state=0).fit(X, Y)
print(clf.predict([[0.0], [3.0]]))
print(export_text(clf))
print(export_text(clf, show_weights=True))
```

## Expected Results

Each leaf shows the class predicted for each output (`class: [0, 0]` on the left, `class: [1, 1]` on the right) and, with `show_weights=True`, the per-output class weights. Or, like `export_graphviz`, no class line for multi-output trees.

## Actual Results

```
[[0 0]
 [1 1]]
|--- feature_0 <= 1.50
|   |--- class: 0
|--- feature_0 >  1.50
|   |--- class: 0

|--- feature_0 <= 1.50
|   |--- weights: [2.00, 2.00] class: 0
|--- feature_0 >  1.50
|   |--- weights: [0.00, 0.00] class: 0
```

## Root cause and possible fix

`sklearn/tree/_export.py:1189-1193`:

```python
if tree_.n_outputs == 1:
    value = tree_.value[node][0]
else:
    value = tree_.value[node].T[0]
class_name = np.argmax(value)
```

The `else` branch is right for multi-output regression (`value` has shape `(n_outputs, 1)`, so `.T[0]` gives one value per output) but not for classification. For a multi-output classifier, compute `np.argmax(tree_.value[node], axis=1)` and map each index through `decision_tree.classes_[k]`, and print the weights per output. This has been the case since `export_text` was added in #9424.

## Versions

```
sklearn: 1.10.dev0, commit 289c4d5cfa
python: 3.13, numpy: 2.5.3, scipy: 1.18.1, platform: linux
```
