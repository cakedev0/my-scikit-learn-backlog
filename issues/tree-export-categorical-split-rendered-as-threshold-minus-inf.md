# `export_text`, `export_graphviz` and `plot_tree` render categorical splits as `feature <= -inf`

```json
{
    "estimators": ["export_text", "export_graphviz", "plot_tree", "DecisionTreeClassifier", "DecisionTreeRegressor", "ExtraTreeClassifier", "ExtraTreeRegressor"],
    "existing_issue": "https://github.com/scikit-learn/scikit-learn/issues/33965",
    "reproduced": true,
    "priority": 2,
    "difficulty": 2,
    "decision": 1,
    "decision_tags": ["complexity"]
}
```

## Classification

- **Priority (2)**: the exporters accept trees with categorical splits and silently print a rule that is wrong ("every non-missing value goes right"), so the printed tree contradicts `predict`. Exporting a tree is a frequent reason to fit a single tree in the first place. `-inf` thresholds look odd enough that some users will notice. Categorical support is not released yet (1.10.dev0), and the export part is a known open item.
- **Difficulty (2)**: read `tree_._left_cat_bitset` / `split_kind` / `missing_go_to_left` and map the encoded categories back through the fitted encoder in three exporters; not hard but touches several functions and needs tests for best (bitset) and random (hash) splits.
- **Decision (1)**: `complexity`: rendering category sets (possibly long, with hash splits up to 2**24 categories) adds code. The cheap alternative is to raise `NotImplementedError` for trees with categorical splits until proper rendering lands.

## Upstream search

Queries: `export_text categorical`, `plot_tree categorical split`.

- #33965 (open): roadmap for native categorical splits in trees. Item 8, "[Feature] Add support for categorical in exporting", is unchecked. It does not mention that the exporters currently print a wrong rule instead of failing.
- #34809 (open, draft, not reviewed): adds `export_dict` and categorical rendering in `export_text`, `export_graphviz`, `plot_tree`.

## Describe the bug and its user-facing impact

For a categorical split, `Tree._add_node` stores `threshold = -inf` and puts the routing in `left_cat_bitset` (or a hash seed for random splits). The exporters only read `threshold`, so a split that sends `{"green", "yellow"}` left and `{"red", "blue"}` right is printed as `color <= -inf` / `color > -inf`. Read literally, this says that all known colors go right, which is false. Nothing in the output tells the user that this is a categorical split, nor which categories go where, nor where missing or unknown categories go.

## Steps/Code to Reproduce

```python
import numpy as np
import matplotlib
matplotlib.use("Agg")
from sklearn.tree import DecisionTreeClassifier, export_text, export_graphviz, plot_tree

X = np.array([["red"], ["green"], ["blue"], ["yellow"]] * 5, dtype=object)
y = np.isin(X[:, 0], ["red", "blue"]).astype(int)
tree = DecisionTreeClassifier(categorical_features=[0], max_depth=1).fit(X, y)
print(tree.predict([["red"], ["green"], ["blue"], ["yellow"]]))
print(export_text(tree, feature_names=["color"]))
print(export_graphviz(tree, feature_names=["color"]).splitlines()[3])
print(plot_tree(tree, feature_names=["color"])[0].get_text())
```

## Expected Results

The root split is shown as a category membership test, e.g. `color in {green, yellow}` / `color not in {green, yellow}` (with the direction of missing and unknown categories), matching `predict`. Or, until that is implemented, a clear `NotImplementedError` instead of a wrong rule.

## Actual Results

```
[1 0 1 0]
|--- color <= -inf
|   |--- class: 0
|--- color >  -inf
|   |--- class: 1

0 [label="color <= -inf\ngini = 0.5\nsamples = 20\nvalue = [10, 10]"] ;
color <= -inf
gini = 0.5
samples = 20
value = [10, 10]
```

## Root cause and possible fix

`sklearn/tree/_tree.pyx:1001-1009` stores `-INFINITY` as the threshold of categorical nodes. `export_text` (`sklearn/tree/_export.py:1207-1213`) and `_BaseTreeExporter.node_to_str` (`sklearn/tree/_export.py:366-371`, used by `export_graphviz` and `plot_tree`) format `tree_.threshold` without checking `tree_.split_kind`. The fix is to branch on `split_kind`, decode the left categories from `left_cat_bitset` (or the hash for random splits) and map them to labels with the estimator's fitted encoder. That needs the estimator, not only `tree_`, which `export_text` and `export_graphviz` already receive. #34809 is a draft along these lines. A minimal stopgap is to raise in the exporters when `tree_.split_kind` contains a categorical kind.

## Versions

```
sklearn: 1.10.dev0, commit 289c4d5cfa
python: 3.13, numpy: 2.5.3, scipy: 1.18.1, platform: linux
```
