# sort_graph_by_row_values misses an unsorted last row when the first row is empty

```json
{
    "estimators": ["sort_graph_by_row_values", "NearestNeighbors", "RadiusNeighborsTransformer"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 1,
    "difficulty": 1,
    "decision": 0,
    "decision_tags": []
}
```

## Classification

- **Priority (1)**: silent unsorted output, but it needs a precomputed sparse graph whose first row(s) store nothing and whose only out-of-order pair is the last two stored values. Graphs built by scikit-learn are already sorted.
- **Difficulty (1)**: one-line fix in the index arithmetic of `_is_sorted_by_data`.
- **Decision (0)**: no trade-off.

## Upstream search

- `sort_graph_by_row_values`: #31030 and #34647 (closed, spurious `EfficiencyWarning` in DBSCAN/OPTICS, i.e. the opposite direction: sorted graphs reported as unsorted).
- `precomputed sparse graph sorted`: #34399, #34692 (merged, fixes for those warnings), #23139 (merged, added `sort_graph_by_row_values`), #10482 (merged, precomputed sparse graphs). None about a leading empty row.

## Describe the bug and its user-facing impact

`sort_graph_by_row_values` (also called on every precomputed sparse input by the neighbors estimators) first checks whether the graph is already sorted with `_is_sorted_by_data`. That check allows a decrease between the last value of a row and the first value of the next row. When the first row is empty, `indptr[1] - 1 == -1`, and index `-1` points to the last comparison of the whole array. If the last two stored values (in the same row) are out of order, the graph is reported as sorted and returned unchanged, without the `EfficiencyWarning`.

Consequence: `radius_neighbors(..., sort_results=True)` and `radius_neighbors_graph(..., sort_results=True)` on such a precomputed graph return unsorted neighbors, and `sort_graph_by_row_values` itself returns an unsorted graph.

## Steps/Code to Reproduce

```python
import numpy as np
from scipy.sparse import csr_array
from sklearn.neighbors import NearestNeighbors, sort_graph_by_row_values

# Precomputed sparse distances where the first sample has no stored
# neighbor and the last row is not sorted by distance.
graph = csr_array([[0.0, 0.0, 0.0],
                   [0.0, 0.0, 2.0],
                   [3.0, 1.0, 0.0]])
print(sort_graph_by_row_values(graph, copy=True).data)

nn = NearestNeighbors(metric="precomputed").fit(graph)
dist, ind = nn.radius_neighbors(graph, radius=5, sort_results=True)
print(dist[2], ind[2])
```

## Expected Results

```
[2. 1. 3.]
[1. 3.] [1 0]
```

## Actual Results

```
[2. 3. 1.]
[3. 1.] [0 1]
```

## Root cause and possible fix

`sklearn/neighbors/_base.py:135-138`:

```python
out_of_order = graph.data[:-1] > graph.data[1:]
line_change = np.unique(graph.indptr[1:-1] - 1)
line_change = line_change[line_change < out_of_order.shape[0]]
return out_of_order.sum() == out_of_order[line_change].sum()
```

A leading empty row gives `line_change == -1`, which numpy reads as the last element. Fix: also filter `line_change >= 0`.

## Versions

```
sklearn: 1.10.dev0, commit f570497251 (no diff in sklearn/neighbors vs upstream/main 289c4d5cfa)
python: 3.13, numpy: 2.5.3, scipy: 1.18.1, platform: linux
```
