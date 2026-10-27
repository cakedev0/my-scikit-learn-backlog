# <Short title, as it would read on the GitHub issue>

```json
{
    "estimators": [],
    "existing_issue": null,
    "reproduced": true,
    "priority": 0,
    "difficulty": 0,
    "decision": 0,
    "decision_tags": []
}
```

## Classification

- **Priority (N)**: <one or two sentences>
- **Difficulty (N)**: <one or two sentences>
- **Decision (N)**: <one or two sentences, covering each tag>

## Upstream search

<Queries run and what came out: related issues and PRs with their state
(open / closed / merged), and why none of them already covers this bug. "No
match" is a valid result, but list the queries anyway.>

## Describe the bug and its user-facing impact

<What is wrong, under which conditions, and what a user sees: wrong result,
silent misbehavior, crash on valid input, misleading error, etc. Say how likely
a real user is to hit it.>

## Steps/Code to Reproduce

```python
# Minimal, self-contained, public API only, synthetic data.
```

## Expected Results

<What should happen, and why (docstring, user guide, paper, consistency with a
sibling estimator, math).>

## Actual Results

```
<Output or full traceback, pasted from an actual run.>
```

## Root cause and possible fix

<Where the bug lives (`sklearn/path/file.py:LINE` on the commit below), why the
code is wrong, and the fix you have in mind, if any. Mention the trade-offs
that justify the decision level.>

## Versions

```
sklearn: <version>, commit <git rev-parse --short HEAD>
python: <x.y>, numpy: <x.y>, scipy: <x.y>, platform: <linux/macos/...>
```
