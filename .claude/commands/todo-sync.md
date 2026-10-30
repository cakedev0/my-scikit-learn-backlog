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

## 0. Pull my edits from GitHub

I may tick or edit items in the GitHub editor. Run `git -C <backlog> fetch`,
and if `origin/main` has commits that aren't in the local branch,
`git -C <backlog> pull --rebase --autostash`. On a conflict, abort the
rebase and ask me.

## 1. Fetch

```
/usr/bin/python3 <backlog>/scripts/todo/fetch_activity.py <scratchpad>/todo-sync $ARGUMENTS
```

It prints one row per thread (tracked, move, reasons, waiting since,
updated, URL, title) and writes the details to `threads.json` (author,
roles, last activity of me and of others, who requested my review and when,
review decision, CI, removal date). Takes ~1 min.

Then list my own edits since the last sync: the uncommitted ones
(`git -C <backlog> diff -- TODO.md todo/`) and the commits after the last
`sync:` one (`git -C <backlog> log -p <last sync>..HEAD -- TODO.md todo/`).
Rule 0 of the instructions says how to record them (ticked items are done,
deleted lines are dropped).

## 2. Read what is expected from me

`move: me` only says someone acted after me. Before adding a thread, adding
one back, or changing an item's action, read the activity since my last
one, for every such thread, whatever the title says:

```
gh api repos/<owner>/<repo>/issues/<N>/comments --jq '.[-5:][] | "\(.user.login) \(.created_at[:10]): \(.body[:400])"'
```

For a PR, also the reviews and the inline review comments:

```
gh api repos/<owner>/<repo>/pulls/<N>/reviews --jq '.[-3:][] | "\(.user.login) \(.state): \(.body[:400])"'
gh api repos/<owner>/<repo>/pulls/<N>/comments --jq '.[-5:][] | "\(.user.login) \(.created_at[:10]): \(.body[:300])"'
```

Then apply the "expected from me" test of the instructions: no item for a
thanks, an approval, a settled answer, others talking. The action says what
is expected ("answer lorentzenchr's concerns on the condition number").

## 3. Edit

Apply the rules of the "Syncing from GitHub" section of the instructions to
`TODO.md`: remove the ticked items, close done items, move items between
sections, add the new threads, sort the Inbox. Keep the existing lines and
their order unless a rule says otherwise.

## 4. Commit and report

Commit `TODO.md` and `todo/` only, in one commit: subject
`sync: <today>`, body with one `add:` / `done:` / `todo:` line per change.
Don't push.

Then report in the terminal, short:

- what changed, per section (`+` added, `-` done, `→` moved);
- the items added in doubt, and the dropped threads added back (rule 3),
  so I know which ones to check first;
- the top of `Now`, as the list stands.
