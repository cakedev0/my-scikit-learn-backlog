# my-scikit-learn-backlog

My triage backoffice for [scikit-learn](https://github.com/scikit-learn/scikit-learn):
a backlog of bugs found by AI agents reading the code, the Claude Code
commands I use to triage upstream issues and PRs, and my work
[todo list](TODO.md).

Nothing here is reported upstream automatically. Each file in `issues/` is a
candidate that I read, re-run and decide on before opening anything.

## Layout

- `issues/`: one bug per file, named `<module>-<short-slug>.md`. Each file
  starts with a JSON block (estimators, existing upstream issue, reproduced,
  priority, difficulty, decision and its tags), then a reproducer with its
  actual output, the root cause and a possible fix.
- `bug-hunting/INSTRUCTIONS.md`: instructions given to the agents that write
  the backlog, including the priority / difficulty / decision scales and my
  calibration examples.
- `bug-hunting/TEMPLATE.md`: the template each backlog file follows.
- `TODO.md`: my work todo list, one line per item, in sections Inbox / Now /
  Reviews / Waiting / Later. Done and abandoned items are removed, and the
  commit message records why (`done:`, `drop:`), so git keeps the history.
- `todo/`: a detail file per item when one line isn't enough, and
  `todo/INSTRUCTIONS.md`, the rules for keeping the list (format, sections,
  commit conventions, how a sync decides what to add).
- `.claude/commands/`: Claude Code commands, available when running `claude`
  from this repo. I also symlink them from `~/.claude/commands/` to use them
  from my scikit-learn checkout.
- `scripts/`: helpers called by the commands (they only read from GitHub).

## Commands

- `/triage <issue-or-pr> [...]`: is this issue or PR worth maintainer time?
  Looks at the real user-facing impact (real, edge case, artifact) rather than
  whether the bug is real, places it on the backlog priority scale, and runs
  `/investigate-github-user` on the author.
- `/investigate-github-user <login>`: looks for drive-by / AI-farmed
  contribution patterns in a GitHub account's public activity (many unrelated
  repos in implausibly short time, templated PR descriptions, low merge rate).
- `/pr-review <pr> [level]`: reads the PR thread, linked issues and prior art
  first, then runs `/code-review` and drops findings the discussion already
  settled.

- `/todo [<item> | done <item> | drop <item>]`: add an item to the todo
  list, remove one, or show `Now`.
- `/todo-sync`: updates the todo list from my GitHub activity since the last
  sync (reviews to answer, review requests, merged PRs...), sorts the Inbox
  and commits. Uses `scripts/todo/fetch_activity.py`.

The first three only produce a report. They never comment, label, close or block
anything on GitHub.

## Setup

- `gh` authenticated, and `jq`.
- This repo cloned at `~/my-scikit-learn-backlog` (where the commands look
  for it when run from elsewhere).
- A scikit-learn checkout next to this repo (`../scikit-learn`), with an
  `upstream` remote on scikit-learn/scikit-learn. Triage reads code on
  `upstream/main` there.

## Using the todo list

- Quick add, without Claude: `scripts/todo/add "some idea"` (appends to the
  Inbox and commits), or edit `TODO.md` on GitHub.
- Find removed items again:
  ```
  git log --grep '^drop:'                          # shelved ideas
  git log --grep '^done:' --since 2026-10-01 --format=%B | grep '^done:'
  git log -S 'radix sort' -p -- TODO.md todo/      # any past item
  ```

## Reading the backlog

The JSON block of each file is valid JSON, so it is easy to load:

```python
import json, re
from pathlib import Path

rows = []
for path in sorted(Path("issues").glob("*.md")):
    meta = json.loads(re.search(r"```json\n(.*?)\n```", path.read_text(), re.S)[1])
    rows.append({"file": path.name, **meta})
```
