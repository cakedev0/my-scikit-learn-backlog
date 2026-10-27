# SVC computes `class_weight="balanced"` without `sample_weight`

```json
{
    "estimators": ["SVC"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 2,
    "difficulty": 1,
    "decision": 1,
    "decision_tags": ["output-change"]
}
```

## Classification

- **Priority (2)**: silent and plausible-looking results: with `class_weight="balanced"` and `sample_weight`, `SVC` balances the raw class counts instead of the weighted ones, so weighting is not equivalent to repeating samples. `LinearSVC` and `LogisticRegression` were fixed for exactly this in #30057, `SVC` was left out.
- **Difficulty (1)**: pass `sample_weight` to `compute_class_weight` in `BaseSVC._validate_targets` (it needs to receive it from `fit`).
- **Decision (1)**: `output-change` for `SVC(class_weight="balanced")` fitted with non-uniform `sample_weight`.

## Upstream search

Queries: "SVC balanced sample_weight", "SVC class_weight balanced sample_weight compute_class_weight", "SVC class_weight balanced sample_weight" (issues and PRs).

No issue for `SVC` specifically. #30057 (merged, "Fix linear svc handling sample weights under class_weight='balanced'") fixed `compute_class_weight`, `LinearSVC` and `LogisticRegression`, but did not touch `BaseSVC._validate_targets`. In its review, @antoinebaker pointed at `BaseSVC` as another place needing the same change (per `issues/utils-balanced-class-weight-ignores-sample-weight-others.md`, which covers the same bug in other estimators); I found no follow-up issue or PR for it. The SVC entries in `instance_generator.py` xfail `check_sample_weight_equivalence` with a reference to meta-issue #16298 (sample_weight semantics in general); this file is a narrower, deterministic cause.

## Describe the bug and its user-facing impact

`compute_class_weight("balanced", ..., sample_weight=...)` accounts for sample weights since #30057, and `_fit_liblinear` (used by `LinearSVC`) passes them. `BaseSVC._validate_targets`, used by `SVC` and `NuSVC`, still calls `compute_class_weight(self.class_weight, classes=cls, y=y_)` without `sample_weight`. With `class_weight="balanced"`, `SVC` therefore balances the unweighted class counts.

The reproducer up-weights class 1 by 5 with integer weights. With `class_weight=None`, `SVC` gives the same model with weights or with repeated samples (up to solver tolerance), but with `class_weight="balanced"` the coefficients differ by 0.4, while `LinearSVC` matches to machine precision in both cases.

For `NuSVC`, `class_weight` has no effect at all (see `issues/svm-nusvc-class-weight-ignored.md`), so only `SVC` is visibly affected for now.

The common test `check_sample_weight_equivalence` is xfailed for `SVC` (meta-issue #16298), which is probably why this was not caught; the `LinearSVC` entry in `instance_generator.py` has an explicit `class_weight="balanced"` case, `SVC` has none.

## Steps/Code to Reproduce

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.svm import SVC, LinearSVC

X, y = make_classification(n_samples=60, random_state=0)
sw = np.where(y == 1, 5, 1)  # integer weights: up-weight class 1
X_rep, y_rep = np.repeat(X, sw, axis=0), np.repeat(y, sw)

for est in [SVC(kernel="linear"), LinearSVC(dual=False, tol=1e-10)]:
    for cw in [None, "balanced"]:
        est.set_params(class_weight=cw)
        weighted = est.fit(X, y, sample_weight=sw).coef_.copy()
        repeated = est.fit(X_rep, y_rep).coef_
        print(f"{type(est).__name__:9s} class_weight={cw!s:8s} "
              f"max |coef(weighted) - coef(repeated)| = "
              f"{np.abs(weighted - repeated).max():.1e}")
```

## Expected Results

`SVC(class_weight="balanced")` with integer `sample_weight` gives the same model as with repeated samples, as with `class_weight=None` and as `LinearSVC` does.

## Actual Results

```
SVC       class_weight=None     max |coef(weighted) - coef(repeated)| = 1.8e-04
SVC       class_weight=balanced max |coef(weighted) - coef(repeated)| = 4.0e-01
LinearSVC class_weight=None     max |coef(weighted) - coef(repeated)| = 3.4e-13
LinearSVC class_weight=balanced max |coef(weighted) - coef(repeated)| = 4.3e-15
```

## Root cause and possible fix

`sklearn/svm/_base.py:784` (`BaseSVC._validate_targets`):

```python
self.class_weight_ = compute_class_weight(self.class_weight, classes=cls, y=y_)
```

`_validate_targets` is called from `BaseLibSVM.fit` before `sample_weight` is converted. Validate `sample_weight` first (it is turned into an empty array when None, which `compute_class_weight` would need as None) and pass it as `sample_weight=` here, as `_fit_liblinear` does at `sklearn/svm/_base.py:1237`. Then add a `class_weight="balanced"` case for `SVC` in the `check_sample_weight_equivalence` parameters of `instance_generator.py`.

Note: the zero-weight-class case of `compute_class_weight` (`issues/utils-compute-class-weight-balanced-zero-weight-class.md`) would then also apply to `SVC`.

## Versions

```
sklearn: 1.10.dev0, commit 289c4d5cfa (upstream/main; ran on f570497251, no diff in the files involved)
python: 3.13.14, numpy: 2.5.3, scipy: 1.18.1, platform: linux
```

See also: `issues/utils-balanced-class-weight-ignores-sample-weight-others.md` (same bug in RidgeClassifier, trees, HistGradientBoostingClassifier, SGDClassifier).
