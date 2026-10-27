---
description: Investigate a GitHub user's public activity for low-effort/AI-farmed drive-by contribution patterns (many unrelated repos/topics in implausibly short timeframes). Report only — never blocks/comments/acts on the account.
argument-hint: <github-username>
---

# Investigate a GitHub User's Contribution Pattern

Open-source maintainers are increasingly seeing PRs from accounts (human or bot)
that clearly did not engage with the codebase they're "fixing" — often visible
because the same account is simultaneously blitzing dozens of unrelated repos
across unrelated topics, which is impossible for someone actually reading and
understanding each codebase. Gather evidence of that pattern for the GitHub
handle given in `$ARGUMENTS` and report it. **Never block, report, or comment
on the user itself — only produce findings for the maintainer to act on.**

## Prerequisites

Requires `gh` (GitHub CLI) authenticated (`gh auth status`) and `jq`.

## Steps

1. **Identify the target.** Take the GitHub username from `$ARGUMENTS`, or ask
   for it if missing. Strip any `@` prefix or full profile URL down to the
   bare login.

2. **Fetch the raw data.**
   ```
   bash scripts/investigate-github-user/fetch_activity.sh <username> <outdir>
   ```
   Use a scratch directory for `<outdir>` (e.g. under the session scratchpad).
   This pulls: profile info, last ~90 days of public events, up to 1000
   PRs authored, up to 1000 issues authored, and up to 1000 issue/PR comments
   — all via `gh api`, so it respects the authenticated rate limits (search
   endpoints are capped at 30 req/min; the script paginates but stays well
   under that for a single user).

3. **Run the quantitative analysis.**
   ```
   python3 scripts/investigate-github-user/analyze_activity.py <outdir>
   ```
   This prints a single JSON report with, notably:
   - `pr_volume.distinct_repos_per_pr` — near 1.0 means the account almost
     never returns to the same repo (classic drive-by pattern); a genuine
     contributor usually iterates on a handful of repos they know.
   - `cross_repo_burstiness.smallest_gaps` — the tightest time gaps between
     PRs opened on *different* repos. Minutes-scale gaps between unrelated,
     differently-themed repos are the strongest signal: no human reviews a
     new codebase, forms an understanding, and opens a coherent PR in under
     10-30 minutes, repeatedly, across topics they've never touched before.
   - `outcomes.merge_rate` and `median_hours_to_close_when_unmerged` — high
     volume + low merge rate + maintainers closing fast is consistent with
     low-quality spam PRs being rejected on sight.
   - `body_signals` — empty/near-empty PR descriptions at scale.
   - `duplicate_templated_bodies` — literally identical PR descriptions reused
     across repos, a strong tell for templated/AI-farmed submissions (distinct
     from a person's personal boilerplate reused on their *own* similar PRs).
   - `profile.account_age_days` vs `pr_volume.prs_per_account_age_day` — a
     brand-new account with high throughput is more suspicious than an old
     account with the same throughput.

4. **Sanity-check with qualitative reads — do not skip this.** The numbers
   alone produce false positives (see Calibration below). Read the full body
   text of:
   - the 5-8 PRs listed in `body_signals.sample_short_body_prs`
   - the PRs on both ends of the 5 smallest gaps in `cross_repo_burstiness.smallest_gaps`
   - any groups in `duplicate_templated_bodies`
   For 2-3 of the most suspicious PRs, look at the actual diff (`gh pr diff <url>`
   or open the html_url) — is it a trivial/cosmetic/no-op change dressed up as a
   fix, a plausible-sounding but wrong "fix", or a real substantive change?
   Also skim a handful of entries in `comments.json` for generic filler text
   ("Great point!", "I've fixed this as suggested", boilerplate praise with no
   technical specifics) which is a common LLM-comment tell, especially when it
   doesn't actually respond to what a reviewer asked.

5. **Calibrate against legitimate explanations before concluding anything.**
   Cross-repo activity and topic breadth are *not* inherently suspicious:
   - Fork PRs, companion/benchmark repos, or a personal mirror of the same
     project (e.g. `<user>/scikit-learn`) opened right after/before the
     upstream PR are one coherent unit of work, not two unrelated repos.
   - Professional OSS maintainers, consultants, and dependency-bump bots
     legitimately touch many repos quickly (but usually same narrow topic,
     e.g. version bumps, and often clearly labeled as automated).
   - Long-tenured contributors with broad expertise can plausibly move fast
     across related ecosystems (e.g. numpy <-> scikit-learn).
   - A single busy day is weak evidence; a sustained pattern across weeks is
     much stronger.
   Read the report as evidence to weigh, not a verdict to rubber-stamp.

6. **Write the report.** Present to the user:
   - One-paragraph verdict: Low / Medium / High concern, with the top 3
     pieces of evidence (or "no strong signal found").
   - The key numbers from step 3.
   - The qualitative findings from step 4 (quote 2-3 concrete examples: repo,
     PR link, why it reads as low-effort or templated).
   - Explicit caveats from step 5 that apply to this specific case.
   - End with: this is heuristic evidence for a human decision, not proof —
     recommend the maintainer skim the PR/comments themselves before acting,
     and that any blocking/reporting action on GitHub is theirs to take
     manually (do not offer to do it for them).

## Notes

- The search API only returns up to 1000 results per query and only indexes
  issues/PRs (not raw commits or code review approvals), so extremely
  high-volume accounts will be undercounted, not overcounted — that's fine
  for this purpose, false negatives are safer than false positives here.
- Rerun `fetch_activity.sh` for a fresh look rather than reusing old output
  from a previous investigation in the same session.

GitHub username: $ARGUMENTS
