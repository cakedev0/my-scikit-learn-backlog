---
description: Update my todo list (TODO.md) from my GitHub activity since the last sync, sort the Inbox, and commit. Only reads from GitHub.
argument-hint: [--since YYYY-MM-DD]
---

# Sync the todo list with GitHub

Paths: `<backlog>` is the root of the my-scikit-learn-backlog clone (the
current directory if it contains `bug-hunting/`, otherwise
`~/my-scikit-learn-backlog`). Run git in it with `git -C <backlog>`.

Read `<backlog>/todo/INSTRUCTIONS.md` first: it has the item format, the
sections and the sync rules. Never write to GitHub.

## 1. Fetch

```
/usr/bin/python3 <backlog>/scripts/todo/fetch_activity.py <scratchpad>/todo-sync $ARGUMENTS
```

It prints one row per thread (tracked, move, reasons, waiting since,
updated, URL, title) and writes the details to `threads.json` (author,
roles, last activity of me and of others, who requested my review and when,
review decision, CI, removal date). Takes ~1 min.

If `git -C <backlog> status --short -- TODO.md todo/` shows uncommitted
changes, they are mine: keep them, and include them in the sync commit as
`todo:` lines.

## 2. Look closer where the rows aren't enough

The `move` and `reasons` columns are heuristics. Before adding a thread to
`Now`, or skipping a `new-activity` one, read the comments since my last
activity when the title doesn't make it obvious:

```
gh api repos/<owner>/<repo>/issues/<N>/comments --jq '.[-5:][] | "\(.user.login) \(.created_at[:10]): \(.body[:400])"'
```

For a PR, also `gh pr view <N> --repo <owner>/<repo> --json reviews --jq '.reviews[-3:][] | "\(.author.login) \(.state): \(.body[:400])"'`.
Use what you read to write the action ("answer lorentzenchr's question on
the solver choice").

## 3. Edit

Apply the rules of the "Syncing from GitHub" section of the instructions to
`TODO.md` and `todo/stale.md`: close done items, move items between
sections, add the new threads, sort the Inbox. Keep the existing lines and
their order unless a rule says otherwise.

## 4. Commit and report

Commit `TODO.md` and `todo/` only, in one commit: subject
`sync: <today>`, body with one `add:` / `done:` / `todo:` line per change.
Don't push.

Then report in the terminal, short:

- what changed, per section (`+` added, `-` done, `→` moved);
- the threads you skipped but weren't sure about, and the removed threads
  with new activity that needs me (rule 3), as questions;
- the top of `Now`, as the list stands.
