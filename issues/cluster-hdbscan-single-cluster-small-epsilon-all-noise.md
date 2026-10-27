# HDBSCAN with allow_single_cluster=True labels every sample as noise for a small cluster_selection_epsilon

```json
{
    "estimators": ["HDBSCAN"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 1,
    "difficulty": 1,
    "decision": 1,
    "decision_tags": ["output-change"]
}
```

## Classification

- **Priority (1)**: needs the non-default combination `allow_single_cluster=True` + `cluster_selection_epsilon > 0`, with an epsilon below the scale of the mutual reachability distances. The result (everything is noise) is easy to notice, but confusing: a smaller epsilon than 0.5 gives fewer cluster members than `epsilon=0`.
- **Difficulty (1)**: cap the threshold, `threshold = min(1 / cluster_selection_epsilon, max sibling lambda)`, one line in `_do_labelling`.
- **Decision (1)**: `output-change` for single-cluster fits with a small epsilon only. The existing tests of this code path (`test_hdbscan_allow_single_cluster_with_epsilon`, `test_labelling_thresholding`) keep passing with the proposed rule, since there `1 / epsilon` is below the max lambda.

## Upstream search

Queries: "allow_single_cluster epsilon", "HDBSCAN alpha", "HDBSCAN store_centers". Related but different: #33219 (open) and its fix #35127 (open PR): `TypeError` crash in `traverse_upwards` with `cluster_selection_epsilon` on NumPy 2 (a size-1 array assigned to a C integer). That is the epsilon search over the cluster tree, not the single-cluster labelling threshold in `_do_labelling` discussed here, and the reproducer below does not crash. `git log -L` on `_do_labelling` only shows the original port (#26385).

## Describe the bug and its user-facing impact

When HDBSCAN selects the root as the single cluster (`allow_single_cluster=True`), a sample that falls out of the root is kept in the cluster only if its lambda (1 / mutual reachability distance at which it leaves) is at least a threshold:

- with `cluster_selection_epsilon=0`, the threshold is the largest lambda among the root's children, so the densest samples are always in the cluster;
- with `cluster_selection_epsilon > 0`, the threshold is `1 / cluster_selection_epsilon`.

If `cluster_selection_epsilon` is smaller than the smallest distance at which a sample leaves the root, `1 / epsilon` is larger than every lambda and no sample passes it: the single cluster is selected but gets no members, and every sample is labelled `-1`. So, as epsilon grows from 0, the cluster first shrinks from 20 samples to 0, then grows again (see below). The docstring describes epsilon as a threshold below which clusters are merged, which should only make clusters larger.

## Steps/Code to Reproduce

```python
import numpy as np
from sklearn.cluster import HDBSCAN
from sklearn.datasets import make_blobs

X, _ = make_blobs(n_samples=200, centers=1, random_state=0)
for eps in [0.0, 0.05, 0.1, 0.5, 1.0]:
    labels = HDBSCAN(
        min_cluster_size=20, allow_single_cluster=True, cluster_selection_epsilon=eps, copy=True
    ).fit(X).labels_
    print(f"cluster_selection_epsilon={eps}: labels={np.unique(labels)}, "
          f"n_noise={(labels == -1).sum()}")
```

## Expected Results

The number of noise points does not increase with `cluster_selection_epsilon`: at most 180 noise points for `eps=0.05` and `eps=0.1`, as for `eps=0`.

## Actual Results

```
cluster_selection_epsilon=0.0: labels=[-1  0], n_noise=180
cluster_selection_epsilon=0.05: labels=[-1], n_noise=200
cluster_selection_epsilon=0.1: labels=[-1], n_noise=200
cluster_selection_epsilon=0.5: labels=[-1  0], n_noise=153
cluster_selection_epsilon=1.0: labels=[-1  0], n_noise=29
```

## Root cause and possible fix

`sklearn/cluster/_hdbscan/_tree.pyx:500-507` (`_do_labelling`):

```python
if cluster_selection_epsilon != 0.0:
    threshold = 1 / cluster_selection_epsilon
else:
    threshold = lambda_array[parent_array == cluster].max()
if parent_lambda >= threshold:
    label = cluster_label_map[cluster]
```

Fix: use `threshold = min(1 / cluster_selection_epsilon, lambda_array[parent_array == cluster].max())`, so that epsilon can only add samples to the single cluster compared to `epsilon=0`. The code comes from the `hdbscan` library port (#26385), which I think has the same behavior.

## Versions

```
sklearn: 1.10.dev0, commit 289c4d5cfa (upstream/main; local checkout f570497251 has no change in sklearn/cluster/_hdbscan)
python: 3.13, numpy: 2.5, scipy: 1.18, platform: linux
```
