# `class_likelihood_ratios` without positive samples returns a mix of NaN and `replace_undefined_by` values

```json
{
    "estimators": ["class_likelihood_ratios"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 1,
    "difficulty": 1,
    "decision": 1,
    "decision_tags": ["unclear-spec"]
}
```

## Classification

- **Priority (1)**: edge case (no positive sample in `y_true`), and the
  values are undefined anyway, but the output contradicts both the docstring
  and the warning emitted on the same call, and depends on `y_pred`.
- **Difficulty (1)**: return early (or use `elif`) in the `support_pos == 0`
  branch.
- **Decision (1)**: someone must decide whether `replace_undefined_by` applies
  to this case. The docstring says both ratios are NaN and only the `fp == 0`
  / `tn == 0` cases can be replaced; the warning says the opposite ("Use the
  `replace_undefined_by` param to control this behavior").

## Upstream search

Queries: "class_likelihood_ratios nan", "class_likelihood_ratios
replace_undefined_by", "class_likelihood_ratios positive nan",
"class_likelihood_ratios positive class nan", "likelihood ratios no positive
samples".

- #29288 (merged): added `replace_undefined_by`; its discussion does not
  cover the no-positive case, and the added tests only check the warning for
  it.
- #22518 (merged): original implementation.
- #29048 (closed): making `zero_division`-like parameters consistent across
  metrics, not this bug.

No match.

## Describe the bug and its user-facing impact

When `y_true` has no positive sample (`tp + fn == 0`), the function warns
that "`positive_likelihood_ratio` and `negative_likelihood_ratio` are both set
to `np.nan`", sets both to NaN, and then falls through to the `fp == 0` and
`tn == 0` branches, which overwrite them:

- if `fp == 0`, LR+ becomes `replace_undefined_by["LR+"]`, otherwise it is
  computed as `0 / 0` (NaN plus a `RuntimeWarning: invalid value`);
- same for LR- with `tn == 0`.

So the result is `(1.0, nan)`, `(nan, nan)` or `(nan, 0.5)` depending on
`y_pred`, with two or three warnings that disagree with each other.

The `fp == 0` warning text is also cut: the last sentence of `msg_end` sits
on separate statement lines (`sklearn/metrics/_classification.py:2455-2457`)
and is never concatenated, so the message ends with "Use the
`replace_undefined_by` param to ".

## Steps/Code to Reproduce

```python
import warnings
from sklearn.metrics import class_likelihood_ratios

warnings.simplefilter("ignore")
y_true = [0, 0, 0, 0]  # no positive sample
for y_pred, replace in [([0, 0, 0, 0], 1.0),
                        ([0, 1, 0, 0], 1.0),
                        ([1, 1, 1, 1], {"LR+": 5.0, "LR-": 0.5})]:
    print(class_likelihood_ratios(y_true, y_pred, labels=[0, 1],
                                  replace_undefined_by=replace))
```

## Expected Results

The same answer for the three calls, per the docstring ("The sum of true
positives and false negatives is 0 [...]: both likelihood ratios are
undefined", replacement only "For the first two cases"): `(nan, nan)`. Or, if
maintainers prefer, `replace_undefined_by` for both, but consistently.

## Actual Results

```
(1.0, nan)
(nan, nan)
(nan, 0.5)
```

## Root cause and possible fix

`sklearn/metrics/_classification.py:2434-2446` sets both ratios to NaN when
`support_pos == 0`, but the following `if fp == 0` (line 2447) and
`if tn == 0` (line 2470) blocks run unconditionally and overwrite them, or
divide `0 / 0` (lines 2467 and 2486). Fix: return right after the
`support_pos == 0` warning (with NaN, or with the replacement values if that
is the chosen spec), and join the `msg_end` string pieces. The existing test
`test_likelihood_ratios_warnings` only checks that the warning is raised, not
the returned values.

## Versions

```
sklearn: 1.10.dev0, commit 289c4d5cfa (upstream/main; sklearn/metrics/ sources identical to the tested checkout)
python: 3.13, numpy: 2.5.3, scipy: 1.18.1, platform: linux
```
