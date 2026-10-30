---
description: Assess whether a scikit-learn issue or PR is worth maintainer effort. Checks the author's contribution pattern (via /investigate-github-user) and the real user-facing impact (real, edge case, or artifact nobody will hit). Report only, never acts on GitHub.
argument-hint: <issue-or-pr-number|url> [more numbers...]
---

# Triage: is this issue/PR worth our time?

Paths: `<backlog>` is the root of the my-scikit-learn-backlog clone (the
current directory if it contains `bug-hunting/`, otherwise
`~/my-scikit-learn-backlog`). `<sklearn>` is the scikit-learn checkout with the
`upstream` remote (the current directory if it is one, otherwise
`<backlog>/../scikit-learn`).

Maintainers juggle dozens of important subjects. With AI-assisted code
audits, almost every incoming bug report is real: correctness is no longer
the question, and "the bug is real" or "the fix is one line" is not a reason
to keep an issue. Most of these reports are uninteresting, and the user wants
to discourage people from opening them: each one costs triage time, attracts
a queue of "can I work on this?" claimers, and turns into PRs to review.
For each issue or PR in `$ARGUMENTS`, decide whether it deserves any of our
attention compared to everything else, and say why. The user is a
scikit-learn maintainer (`cakedev0`): advise from the triage side.

Never post, label, close or comment on GitHub. Only produce the report; any
action is the user's to take.

## 0. Parse arguments

- Each target is a number, `#N` or a GitHub issue/PR URL. Repo defaults to
  `scikit-learn/scikit-learn`; take it from the URL if given.
- Several targets: run steps 1-5 for each (start all the author
  investigations from step 2 at once), then end with the summary table of
  step 6.

## 1. Fetch the thread

```
gh api repos/<repo>/issues/<N> --jq '{title, user: .user.login, assoc: .author_association, state, is_pr: (.pull_request != null), labels: [.labels[].name], reactions: .reactions.total_count, comments, created_at}'
```

- **Issue**: `gh issue view <N> --repo <repo> --comments`.
- **PR**: `/usr/bin/python3 <backlog>/scripts/pr-review/fetch_pr_context.py <N> <scratchpad>/triage-<N>`,
  then read `pr.md`, `discussion.md`, and `refs/<M>.md` for the issues it
  closes. The impact of a PR is mostly the impact of the problem it solves, so
  assess that issue too (even when it was opened by someone else).
- Linked work, for both: issues and PRs cross-referencing this one
  ```
  gh api --paginate repos/<repo>/issues/<N>/timeline --jq '.[] | select(.event=="cross-referenced") | "\(.source.issue.number)\t\(if .source.issue.pull_request then "PR" else "issue" end)\t\(.source.issue.state)\t\(.source.issue.user.login)\t\(.source.issue.title)"'
  ```
  Note competing PRs for the same issue, duplicates, and downstream projects
  reporting the same problem (a strong sign of real impact).

## 2. Investigate the author (in the background)

Unless the author is a maintainer (`assoc` is `MEMBER` or `OWNER`) or the user
themselves, spawn a `general-purpose` agent in the background, right after
step 1, with this prompt: "Run the `investigate-github-user` skill on
`<login>` and follow its steps. Report in under 25 lines: the concern level
(Low / Medium / High), the key numbers, and 2-3 concrete examples with links."
For a PR on an issue opened by someone else, investigate the PR author only,
unless the issue itself looks farmed too.

Continue with step 3 while it runs. A High concern doesn't settle the
verdict: a farmed account can still report a real bug. It lowers the trust in
claims you couldn't verify yourself, and for a PR it raises the expected
review cost (unread code, unresponsive to reviews).

## 3. Assess the user-facing impact

The question is "who hits this, doing what, and what happens to them?", not
"is this technically wrong?". Assume it is technically right, and spend the
effort on the trigger, not on confirming the bug.

1. **Understand the claim.** Restate in one or two sentences what goes wrong,
   under which exact conditions, and what the user sees (silent wrong result,
   crash, misleading error, warning, slowness, doc mismatch).

2. **Quick sanity check, not a verification.** Read the code involved on
   `upstream/main` (`git -C <sklearn> show upstream/main:<path>`, after
   `git -C <sklearn> fetch upstream`) just enough to know the conditions the bug needs, and check it
   wasn't already fixed (`git log upstream/main --oneline -15 -- <file>`,
   `doc/whats_new/upcoming_changes/`). Run the reproducer (with
   `OMP_NUM_THREADS=2`, under ~10 s) only when it's cheap or when the claim
   looks doubtful; a successful run adds nothing to the verdict.

3. **Find where the trigger comes from.** Look at the conditions the bug
   needs and how a real user would get there:
   - default or commonly used settings, on a widely used estimator?
   - a realistic dataset, or contrived values (constant features with
     `sample_weight` of 1e-300, all-NaN columns, 1 sample, inf in parameters,
     dtypes nobody uses)?
   - public API only, or private helpers / mocked objects / monkeypatching?
   - did the author describe an actual use case, or was it found by reading
     the code ("I noticed that...", a polished "Root cause" section, a
     reproducer that constructs the edge case directly, a custom estimator
     written only to expose the bug, "potential consequences" instead of an
     observed one)? This is the most important signal: no real use case
     means the report has to stand on the trigger alone.
   - can any scikit-learn estimator reach it, or only a third-party/custom
     one? Does it need an opt-in feature (metadata routing, array API,
     callbacks) *combined* with a rarely used method or parameter?
   - evidence of others hitting it: reactions, duplicates, "same here"
     comments, downstream reports, Stack Overflow.
   - is there a cheap workaround? does the user get a clear error, or a
     plausible-looking wrong number?

4. **Classify** into one of:
   - **Real impact**: users plausibly hit it in normal use, and the
     consequence matters (silent wrong results, a broken basic workflow).
   - **Edge case**: real and reachable by a sensible user, but needs a niche
     combination of inputs or parameters, or has a clear error / cheap
     workaround.
   - **Artifact**: only reachable through contrived inputs, private API, or
     conditions no real workflow produces: e.g. a combination of public
     features that no built-in estimator exercises and nobody reported
     using. Technically true, practically irrelevant.
   - **Not a bug**: intended behavior (check the docstring, the user guide,
     tests asserting the current behavior, `git blame`), already fixed on
     main, or a misunderstanding.
   For feature requests and enhancement PRs, the same question applies to the
   need: how many users want this, is it in scope for scikit-learn (see the
   inclusion criteria in the FAQ), and does it duplicate something that
   exists?

5. **Place it on the backlog scale.** Give a priority 0-4 using the scale
   and calibration examples of `<backlog>/bug-hunting/INSTRUCTIONS.md`, and search the
   backlog for the same bug (`grep -ril "<Estimator>\|<function>" <backlog>/issues/`).
   A match gives you its priority and prior analysis; reuse it rather than
   redoing it, but re-check it against what the new report adds.

## 4. For a PR: what would it cost to land?

Skip the deep code review (that's `/pr-review`). Assess the cost side:

- Diff size and spread vs the size of the problem. A 300-line change for an
  artifact is a no, even if correct.
- Does it actually address the issue's root cause, or patch a symptom (a
  special case, a try/except, a new validation that rejects valid input)?
- Tests that would fail without the fix? Changelog entry?
- Behavior changes it introduces (new errors, changed outputs, deprecations),
  and whether they need a maintainer decision first.
- Was the issue triaged and agreed on before the PR was opened? Is there a
  competing or earlier PR, or a maintainer already working on it?
- Author responsiveness to earlier reviews, in this PR or (from step 2)
  elsewhere.

## 5. Verdict

Combine impact, cost and author signals into one of:

- **Worth it**: real impact: users plausibly hit it in normal use and the
  consequence matters. A cheap or obviously right fix is not enough on its
  own. Say what should happen next (review it, ask for a test, label it for
  the next release).
- **Low priority**: an edge case that a real user actually hit (described use
  case, "same here" comments, downstream report). Fine to keep open, label
  "Low Priority", don't spend review time now.
- **Close**: artifact, edge case found by reading the code with no real use
  case behind it, not a bug, duplicate, out of scope, or a PR whose review
  cost exceeds its value. Closing is how we discourage these reports, so
  don't keep one open just because it's real and the fix is small. Give the
  one-line reason a maintainer would write when closing.

The burden is on the report: an issue without a real use case defaults to
Close unless the trigger is plainly common (default settings on a widely used
estimator, silent wrong results). Keeping it open "since it's cheap" invites
more of the same.

## 6. Report

In the terminal, per target:

1. **Header**: `#N` title, link, issue/PR, author and association, age.
2. **Verdict** (Worth it / Low priority / Close) and impact class, in one
   line, plus a backlog priority if available.
3. **Why**: 3-6 bullets covering the trigger conditions, who would hit it,
   whether there's a real use case behind the report, workaround, and for a
   PR the cost side. Don't argue from "it's real" or "the fix is easy". At
   most a short note on the reproducer (ran / not run).
4. **Author**: the concern level from step 2 with its top piece of evidence,
   or "maintainer, skipped".
5. **Suggested action**: label, close, ask a question, review, or redirect
   to an existing issue/PR, with links.

With several targets, end with a table sorted by verdict: `#N | title |
impact | verdict | author concern | action`.

Then offer to draft the comment for any target that needs one (closing
reason, question to the author, pointer to a duplicate), loading the
`writing-style` skill for it. A closing comment should say plainly that
we don't take issues for bugs nobody hits in practice, even when they're
real. Don't invite
the author to come back with a use case, and never suggest example
scenarios: an AI-assisted author will just make one up. When step 2 found a
Medium or High concern, also name the pattern factually (e.g. "N PRs on N
unrelated projects within N hours", "3 PRs for this in 2 hours, 2 closed
within 2 minutes") and link the [Automated Contributions
Policy](https://scikit-learn.org/dev/developers/contributing.html#automated-contributions-policy),
pointing at the rule that the record or the author's own AI disclosure makes
relevant (unreviewed AI output, AI-generated descriptions, missing
disclosure). Stick to verifiable facts (timestamps, counts, what the
disclosure says), without guessing at intent. Never judge how the author's
text reads ("reads as AI-generated", "looks LLM-written"): it's subjective
and invites an argument. Never post it; the user does.

Targets: $ARGUMENTS
