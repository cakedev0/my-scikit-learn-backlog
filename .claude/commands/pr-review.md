---
description: Context-aware PR review. Reads the PR description, discussion, review threads, decisions already taken, linked issues and related prior work, then runs /code-review and reconciles its findings with that context.
argument-hint: <pr-number|pr-url> [low|medium|high|xhigh|max] [other /code-review flags]
---

# Context-aware PR review

Paths: `<backlog>` is the root of the my-scikit-learn-backlog clone (the
current directory if it contains `bug-hunting/`, otherwise
`~/my-scikit-learn-backlog`). `<sklearn>` is the scikit-learn checkout with the
`upstream` remote (the current directory if it is one, otherwise
`<backlog>/../scikit-learn`).

Review the PR given in `$ARGUMENTS` like a maintainer who has read the whole
thread first. Plain `/code-review` only sees the diff, so it re-raises points
that were already discussed, contradicts decisions reviewers already took, and
misses "this was done differently last time" inconsistencies. This command
gathers that context first, then reviews.

Never post anything on GitHub (comments, reviews, labels) unless the user
explicitly asks after seeing the report.

## 0. Parse arguments

- PR: a number, `#N` or a GitHub PR URL. If missing, use the PR of the current
  branch (`gh pr view --json number,url`); if there is none, ask.
- Repo defaults to `scikit-learn/scikit-learn`; take it from the URL if one is
  given.
- An effort level (`low` .. `max`) and other flags are forwarded to
  `/code-review`, except `--comment`: drop it and offer posting at the end.
- Note whether the PR author is the user (`gh api user -q .login`). For their
  own PR, frame findings as things to fix, not as review comments to post.

## 1. Fetch the PR thread

```
/usr/bin/python3 <backlog>/scripts/pr-review/fetch_pr_context.py <pr> <scratchpad>/pr-<N>
```

This writes `pr.md` (metadata, description, changed files, commits),
`discussion.md` (comments, reviews and inline threads in chronological order,
each thread tagged RESOLVED / UNRESOLVED / outdated), `refs.txt` and
`refs/<N>.md` (every issue or PR referenced in the description or discussion,
with its own comments). Pass `--repo owner/name` for another repo.

Read `pr.md` and `discussion.md` in full. Then read the refs that matter:
always the issues this PR closes, plus referenced issues/PRs that carry design
discussion (skip drive-by mentions). If a linked issue is itself long and
points to more context (an RFC, a SLEP, a parent issue), follow it one level.

## 2. Look for prior art (when relevant)

Skip this for trivial PRs (typo, doc tweak, CI bump). Otherwise, spawn one
`general-purpose` agent in the background while you finish step 1, with the PR
title, description summary and changed files, asking it to answer: "has
something similar already been done in this repo, and how?" It should:

- `gh search prs --repo <repo> "<keywords>" --limit 20` and `gh search issues`
  with 2-3 keyword variants, including closed/merged and superseded attempts
  at the same thing.
- `git log --oneline -S '<symbol>'` / `git log --grep` on the key symbols,
  and `git log --oneline -15 -- <touched files>` for recent history.
- Find analogous code in the codebase: how the same pattern is implemented
  elsewhere (e.g. other deprecations of a default, other array API ports,
  other estimators gaining `sample_weight` support, other Cython changes of
  the same kind), and the conventions it follows.
- Report back in under ~40 lines: links to the relevant PRs/issues with one
  line each on what was decided or how it was done, plus concrete conventions
  the current PR should match. No file dumps.

## 3. Write the context brief

Write `<scratchpad>/pr-<N>/brief.md` and show a condensed version to the user
before the review starts:

- **Intent**: what the PR does and why, in 2-3 sentences, including the
  problem from the linked issue.
- **Decisions already taken**: design choices agreed in the thread or the
  linked issues, with who agreed and a link. Include things explicitly
  rejected or deferred to a follow-up.
- **Open points**: unresolved threads, unanswered questions, requested changes
  that the latest commits may or may not address.
- **Prior art**: the agent's findings, if run.
- **Review focus**: what deserves the closest look given the above (e.g. the
  part that changed after the last review, a tricky edge case raised in the
  issue but not tested).

## 4. Run the review

Invoke the `code-review` skill with the PR number and the forwarded level and
flags. Read and understand the change well before judging it, especially for
large diffs. Keep the brief in mind while doing so.

Do not switch the branch of `<sklearn>` to look at the PR code. Use `gh pr diff`, or a
worktree on `upstream pull/<N>/head` if you need the files. When a finding is
a potential bug, check the tests don't already cover it and try to trigger it
with a minimal reproducer (step 5b decides what happens to findings without
one). If the PR code is not built locally, say the reproducer was not run
rather than claiming it was.

## 5. Reconcile and filter findings

`/code-review` is tuned for recall: it reports plausible edge cases that cost
a human reviewer much more to check than they are worth. This step is the
filter. Cheap findings may pass freely. Expensive ones have to earn their
place.

### 5a. Against the context

Go through every finding from the review and against the brief:

- **Already raised**: if a reviewer raised it, say so with a link. Keep it
  only if the thread is still unresolved or the fix is incomplete.
- **Already decided**: if it contradicts a decision taken in the thread or
  the issue, drop it, unless there is a new argument the discussion missed;
  then keep it and say what is new.
- **Inconsistent with prior art**: add findings where the PR diverges from how
  the same thing was done before (with a link to the precedent), when the
  divergence is not justified in the thread.
- **Unaddressed requests**: add requested changes from earlier reviews that the
  current diff does not address, and resolved threads whose resolution does
  not match the code.

### 5b. Against real impact

Split the findings in two:

- **Nits**: anything the user can check from the diff alone in under a
  minute: naming, a wrong docstring or comment, dead code, a missing
  changelog entry or test, a convention mismatch. Keep them. They are cheap
  even when the user disagrees.
- **Behavioral findings**: anything claiming the code does the wrong thing
  for some input or state. Each one has to pass all of the gates below.

1. **Argue against it first.** Before keeping a behavioral finding, make the
   strongest case that it does not matter in practice: input validation or
   an upstream check already rules the input out, no public estimator or
   documented workflow reaches this path, the trigger needs a combination
   nobody uses, the consequence is a clear error with an obvious fix. Look
   for that check in the code, don't assume it. Keep the finding only if
   this argument fails.
2. **Classify the impact.** Ask "who hits this, doing what, and what happens
   to them?", not "is this technically wrong?":
   - **Real**: users plausibly hit it in normal use and the consequence
     matters. Keep.
   - **Edge case**: reachable by a sensible user, but needs a niche
     combination of inputs or parameters. Keep only when the consequence is
     silent wrong results (or a crash in a supported, documented setting);
     drop it when there is a clear error or a cheap workaround.
   - **Artifact**: only reachable through contrived inputs (`n_samples=1`,
     NaN in `sample_weight`, float16, a custom estimator written to expose
     it), private API, monkeypatching, or states that validation already
     prevents. Drop.
3. **Reproduce it.** A behavioral finding goes in the main list only if a
   reproducer using the public API with realistic data was actually run and
   showed the problem. Without one (code not built, reproducer too costly,
   reproducer did not trigger it), move it to the "unverified" list.
4. **Check it is introduced by the PR.** Run the reproducer on main, or check
   `git blame` when there is no reproducer. A bug already on main is out of
   scope by default: drop it to the dropped list. Keep it when fixing it
   belongs in this PR: the PR rewrites that code anyway, the PR makes it
   easier to hit, or it is the same bug the PR fixes in a sibling code path.
   Say which in the finding.

The number of findings must come out of the filter, never out of a target.
Zero findings is a normal result for a good PR: say so plainly rather than
promoting nits or edge cases to fill the report. A large PR with real
problems can just as well have 10 or 20, and that is fine too.

## 6. Report

In the terminal:

1. The condensed context brief (intent, decisions, open points, prior art).
2. Behavioral findings that passed 5b, most severe first, each with
   `file:line`, what is wrong, the reproducer that was run (and who hits it,
   in one line), and its context tag (new / still-open from @reviewer /
   diverges from #NNNN / pre-existing but belongs here because ...). If none
   passed, say so in one line.
3. Nits, grouped, one line each.
4. Unverified: behavioral findings that passed the impact gates but could
   not be reproduced, one line each with what blocked the reproducer.
5. Dropped: findings filtered out, one line each with the reason (settled
   in the thread with a link / artifact / edge case with a clear error /
   pre-existing on main), so the user can disagree.
6. Next step: for someone else's PR, offer to draft review comments (load the
   `writing-style` skill for that) and, only on request, post them. For the
   user's own PR, offer to apply the fixes.
