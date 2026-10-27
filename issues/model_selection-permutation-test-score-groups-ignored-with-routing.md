# `permutation_test_score` permutes `y` across groups when metadata routing is enabled

```json
{
    "estimators": ["permutation_test_score"],
    "existing_issue": null,
    "reproduced": true,
    "priority": 2,
    "difficulty": 2,
    "decision": 0,
    "decision_tags": []
}
```

## Classification

- **Priority (2)**: silently wrong p-value: the null distribution ignores the group structure the user asked for. Only with metadata routing enabled (opt-in), but then there is no way at all to get within-group permutations.
- **Difficulty (2)**: `permutation_test_score` has to consume `groups` from `params` itself while still routing it to the splitter, which needs some care with the routing machinery.
- **Decision (0)**: no trade-off, it makes routing on and off behave the same.

## Upstream search

`gh issue list --search` / `gh pr list --search` (GraphQL, REST search was rate limited): "permutation_test_score groups metadata routing", "permutation_test_score groups".

- #22893 (open): SLEP006 meta-issue, no mention of this.
- #21543 (open): enhancement about shuffling when a group enforces a specific label. Different topic.
- #8127, #12139 (closed): unrelated.

No match.

## Describe the bug and its user-facing impact

`groups` has two roles in `permutation_test_score`. The docstring says:

> Labels to constrain permutation within groups, i.e. `y` values are permuted among samples with the same group identifier. [...] When a grouped cross-validator is used, the group labels are also passed on to the `split` method of the cross-validator.
>
> `groups` can only be passed if metadata routing is not enabled [...]. When routing is enabled, pass `groups` alongside other metadata via the `params` argument instead. E.g.: `permutation_test_score(..., params={'groups': groups})`.

When routing is enabled and the user follows that advice, `groups` from `params` is routed to the splitter only. The permutations are then done across all samples instead of within groups, without any warning. The permutation scores and the p-value change completely, so the test answers a different question than the one asked (typically giving a much smaller p-value).

## Steps/Code to Reproduce

```python
import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, permutation_test_score

rng = np.random.RandomState(0)
groups = np.repeat(np.arange(10), 10)
y = groups % 2  # y is constant within each group
X = y[:, None] + rng.randn(100, 1)
kw = dict(cv=GroupKFold(n_splits=5), n_permutations=20, random_state=0)

# Routing disabled: y is permuted within groups, so it never changes.
score, perm_scores, pvalue = permutation_test_score(
    LogisticRegression(), X, y, groups=groups, **kw
)
print("routing disabled:", score, perm_scores[:5].round(2), pvalue)

# Routing enabled: groups must go through `params`, as the docstring says.
with sklearn.config_context(enable_metadata_routing=True):
    score, perm_scores, pvalue = permutation_test_score(
        LogisticRegression(), X, y, params={"groups": groups}, **kw
    )
print("routing enabled: ", score, perm_scores[:5].round(2), pvalue)
```

## Expected Results

The same permutation scores and p-value with and without routing: `y` is constant within each group, so within-group permutations leave it unchanged and the p-value is 1.0.

## Actual Results

```
routing disabled: 0.71 [0.71 0.71 0.71 0.71 0.71] 1.0
routing enabled:  0.71 [0.38 0.47 0.44 0.47 0.36] 0.047619047619047616
```

## Root cause and possible fix

`sklearn/model_selection/_validation.py:1609` raises if `groups` is passed while routing is enabled, so `groups` is always `None` in that case. At line 1677, `_shuffle(y, groups, random_state)` then uses that `None` and permutes across all samples. The groups passed in `params` only reach `routed_params.splitter.split`.

With a non-group splitter it is worse: `params={"groups": groups}` with `cv=KFold(4)` raises `TypeError: permutation_test_score.fit got unexpected argument(s) {'groups'}, which are not routed to any object.`, so under routing there is no way to ask for within-group permutations at all.

Fix: when routing is enabled, let `permutation_test_score` consume `groups` from `params` itself for `_shuffle`, and still route it to the splitter when the splitter requests it. A test with routing enabled comparing to the routing-disabled result would cover it.

## Versions

```
sklearn: 1.10.dev0, commit f570497251 (local branch; the code involved in
_validation.py is identical on upstream/main 289c4d5cfa)
python: 3.13, numpy: 2.5.3, scipy: 1.18.1, platform: linux
```
