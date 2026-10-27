# DBSCAN(metric="minkowski") crashes with a TypeError because of its default p=None

```json
{
    "estimators": ["DBSCAN", "NearestNeighbors"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 2,
    "difficulty": 1,
    "decision": 0,
    "decision_tags": []
}
```

## Classification

- **Priority (2)**: a crash on a documented parameter combination of a widely used estimator. `metric="minkowski"` is a natural choice even though it isn't the default, and the `TypeError` looks unrelated to what the user did. Passing `p=2` works around it. Maintainer calibration, 2026-10-23.
- **Difficulty (1)**: in `DBSCAN.fit`, pass `p=2 if self.p is None else self.p` when the metric is `"minkowski"` (or always: other metrics ignore `p`). Raising an informative error in `NearestNeighbors` for `metric="minkowski", p=None` would be a good complement.
- **Decision (0)**: no trade-off, the docstring already says `None` means 2.

## Upstream search

- #26729 (closed PR, "FIX do not allow for p=None in NN-based algorithm"): @glemaitre noted that `DBSCAN` has `p=None` by default; @jeremiedbb closed it saying `p=None` is valid for other metrics and "we should instead raise an informative error when p=None and metric=minkowski". I found no follow-up issue or PR, and `DBSCAN` itself was not fixed.
- #12437 (open): estimators defaulting to `metric="euclidean"` while exposing `p`/`metric_params` should default to `"minkowski"`. Related API discussion; changing the default would turn this crash into the default path, so it depends on this fix.
- Queries: `DBSCAN minkowski p None`, the `TypeError` message with `minkowski`, `NearestNeighbors p None minkowski TypeError`, `gh issue list`/`gh pr list --search "p=None minkowski"` (these two don't search comments).

## Describe the bug and its user-facing impact

`DBSCAN.p` is documented as "float, default=None. ... If None, then `p=2` (equivalent to the Euclidean distance)". `DBSCAN.fit` passes `p=self.p` to `NearestNeighbors`, which accepts `None` in its parameter validation but then stores `effective_metric_params_["p"] = None` for the minkowski metric and compares it with `< 1`. So `DBSCAN(metric="minkowski")` (with or without `metric_params={"w": ...}`) fails with an unrelated-looking `TypeError`, for every `algorithm`. The `dbscan` function works because its own default is `p=2`.

## Steps/Code to Reproduce

```python
import numpy as np
from sklearn.cluster import DBSCAN, dbscan

X = np.random.RandomState(0).rand(20, 2)

print(dbscan(X, eps=0.3, metric="minkowski")[1])  # function: p=2 by default, works
print(DBSCAN(eps=0.3, metric="minkowski", p=2).fit(X).labels_)
DBSCAN(eps=0.3, metric="minkowski").fit(X)  # p=None is documented to mean p=2
```

## Expected Results

The last line gives the same labels as the first two, as documented for `p=None`.

## Actual Results

```
[ 0  0  0  0  0  0  0 -1  0  0  0  0  0  0  0  0  0  0  0  0]
[ 0  0  0  0  0  0  0 -1  0  0  0  0  0  0  0  0  0  0  0  0]
Traceback (most recent call last):
  File "/tmp/claude-1000/-home-arthur-scikit-learn/d6c0fdce-f3ba-4445-bc1b-31c285ca4f5a/scratchpad/swarm2/kmeans-dbscan/repro_dbscan_p.py", line 8, in <module>
    DBSCAN(eps=0.3, metric="minkowski").fit(X)  # p=None is documented to mean p=2
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^
  File "/home/arthur/scikit-learn/sklearn/base.py", line 1403, in wrapper
    return fit_method(estimator, *args, **kwargs)
  File "/home/arthur/scikit-learn/sklearn/cluster/_dbscan.py", line 440, in fit
    neighbors_model.fit(X)
    ~~~~~~~~~~~~~~~~~~~^^^
  File "/home/arthur/scikit-learn/sklearn/base.py", line 1403, in wrapper
    return fit_method(estimator, *args, **kwargs)
  File "/home/arthur/scikit-learn/sklearn/neighbors/_unsupervised.py", line 175, in fit
    return self._fit(X)
           ~~~~~~~~~^^^
  File "/home/arthur/scikit-learn/sklearn/neighbors/_base.py", line 624, in _fit
    and self.effective_metric_params_["p"] < 1
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
TypeError: '<' not supported between instances of 'NoneType' and 'int'
```

Also reproduced on scikit-learn 1.9.0.

## Root cause and possible fix

- `sklearn/cluster/_dbscan.py:437`: `p=self.p` is forwarded as is, although the docstring maps `None` to 2.
- `sklearn/neighbors/_base.py:527-544`: `effective_p = self.effective_metric_params_.get("p", self.p)` keeps `None`; `pop("p", 2)` then returns that `None` (the key exists), and `:624` / `:646` compare it with 1.

Fix in `DBSCAN.fit`: forward `p=2` when `self.p is None`. Separately, `NearestNeighbors` could raise a clear `ValueError` for `metric="minkowski", p=None` (as suggested on #26729).

## Versions

```
sklearn: 1.10.dev0, commit 289c4d5cfa (also 1.9.0)
python: 3.13, numpy: 2.5.3, scipy: 1.18.1, platform: linux
```
