---
description: Context-aware PR review. Reads the PR description, discussion, review threads, decisions already taken, linked issues and related prior work, then runs /code-review and reconciles its findings with that context.
argument-hint: <pr-number|pr-url> [low|medium|high|xhigh|max] [other /code-review flags]
---

# Context-aware PR review

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
/usr/bin/python3 scripts/pr-review/fetch_pr_context.py <pr> <scratchpad>/pr-<N>
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

Do not switch the branch of the scikit-learn checkout to look at the PR code. Use `gh pr diff`, or a
worktree on `upstream pull/<N>/head` if you need the files. When a finding is
a potential bug, check the tests don't already cover it and try to trigger it
with a minimal reproducer. If the PR code is not built locally, say the
reproducer was not run rather than claiming it was.

## 5. Reconcile findings with the context

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

## 6. Report

In the terminal:

1. The condensed context brief (intent, decisions, open points, prior art).
2. Findings, most severe first, each with `file:line`, what is wrong, a
   concrete failure scenario or reproducer, and its context tag (new /
   still-open from @reviewer / diverges from #NNNN).
3. A short list of review findings dropped because the thread already settled
   them, one line each with the link, so the user can disagree.
4. Next step: for someone else's PR, offer to draft review comments (load the
   `writing-style` skill for that) and, only on request, post them. For the
   user's own PR, offer to apply the fixes.
