# Todo list: how to keep it

Instructions for the agents (and me) that edit `TODO.md` and the files of
`todo/`. The list is my work todo list as a scikit-learn maintainer at
Probabl, mostly on the Intel / Probabl collaboration to make scikit-learn
faster on Intel hardware (see the [announcement][announcement] and the
[progress post][post]). Its "What's next" section is the roadmap the `Later`
items come from.

[announcement]: https://blog.probabl.ai/intel-and-probabl-announce-collaboration-to-accelerate-scikit-learn-on-intel-hardware
[post]: https://github.com/scikit-learn/blog/pull/228

## Giving back

I open many big PRs on scikit-learn, and each one costs other maintainers
review time. So reviews, triage and cleaning up old issues and PRs are part
of the job, not side work, and they get the same care as my own PRs:

- **Reviews I requested myself** (`self-requested` in the sync, often on old
  PRs I picked while cleaning up a backlog) are commitments I chose. They go
  in `Reviews`, above the requests from others unless I reorder them.
- **Threads I reactivated** (an old issue or PR where I just commented) are
  live again: how long they've waited counts from the last event that put
  the ball in my court (`waiting_since`), never from the thread's creation.
- **Triage follow-ups**: when I triaged a thread, what I wrote decides the
  item:
  - I promised something ("I can do that", "I'll review it", "I'll open an
    issue"): an item in `Now`.
  - I handed it to someone (asked a maintainer to take it, asked the author
    to mark the PR ready, suggested closing): an item in `Waiting`, naming
    who it waits on, until it moves.
  - I labeled it, or answered and nothing more is expected from me: no item.

Reviews items start with "review" and triage items with "follow up", so
`done:` lines can be counted, e.g. how much I gave back this month:

```
git log --grep '^done: [^:]*: review' --since 2026-10-01 --oneline | wc -l
```

## It's public

This repo is public. Only write what could be posted on GitHub: public
issues, PRs, benchmarks and plans. No private discussions, no details of
Intel's or Probabl's internal plans, nothing about people beyond their public
GitHub activity. Leave out work that has nothing to do with scikit-learn and
the projects around it (skore, personal repos, old school projects).

## Files

- `TODO.md`: the list, one line per item, in five sections:
  - **Inbox**: quick adds, not sorted yet. Empty after each sync.
  - **Now**: my move, to do in the next days: answer a review, fix a red CI,
    reply to a mention.
  - **Reviews**: other people's PRs I review or was asked to review.
  - **Waiting**: my PRs and reports where the ball is in someone else's
    court. Worth a ping when nothing moves for a few weeks.
  - **Later**: planned work, experiments to upstream or close, and ideas.
- `todo/<slug>.md`: optional details for one item: context, notes, plan,
  dated log. Free-form, starting with a `# Title`. The item's line links to
  it, and the file is deleted in the same commit as its item.

A bug of the backlog (`issues/`) only becomes an item when I decide to act
on it, and its line links to the `issues/` file as its details. Don't copy
the bug backlog into the list.

## Item format

```
- [ ] <Theme>: <action>, [<ref>](<url>) (<few words of context>) → [notes](todo/<slug>.md)
```

- **Theme**, one of: HGB, KMeans, Trees, Linear, Encoders, Preprocessing,
  Threads, Array API, GPU, Docs, Benchmarks, Blog, Upstream (bugs and PRs in
  other projects, e.g. sklearnex, oneDAL, CPython), Maintenance (triage,
  reviews, CI). Add a theme only when none fits.
- **Action**: start with a verb ("answer lorentzenchr on", "fix the CI of",
  "review", "follow up on", "upstream or close"). Name who to answer, or who
  the item waits on.
- **Refs**: Markdown links with a short ref as the text: `sklearn#N`
  (scikit-learn/scikit-learn), `benchmarks#N`
  (probabl-ai/scikit-learn-benchmarks), `fork#N` (cakedev0/scikit-learn),
  `blog#N` (scikit-learn/blog), `joblib#N`, `loky#N`, `threadpoolctl#N`,
  `sklearnex#N` (uxlfoundation/scikit-learn-intelex), `oneDAL#N`
  (uxlfoundation/oneDAL), `raft#N` (NVIDIA/raft), `cpython#N`, `numpy#N`,
  `scipy#N`, otherwise `owner/repo#N`. Always the full URL in the link:
  GitHub doesn't autolink `owner/repo#N` in Markdown files.
- One item can link several threads when they are one piece of work (a PR,
  the issue it fixes and its benchmark PR). Its first link is the main one.
- Keep it to one line. Anything longer goes to a detail file.
- One concrete action that can be ticked when done. No meta items ("go
  through the old threads", "triage the backlog"): list the actions
  themselves. Several threads can share an item only when one action covers
  them (answering the same question in each, closing a group of
  experiments).
- A bigger piece of work (getting back to a paused topic, a multi-PR
  project) can be a parent item with sub-items, indented by two spaces:

  ```
  - [ ] Encoders: go back to the encoders work (paused)
    - [ ] answer lorentzenchr on [sklearn#34387](<url>) (...)
    - [ ] resume or close [sklearn#34678](<url>) (...)
  ```

  The parent says the goal, the sub-items are the concrete actions, without
  a theme. Ticked sub-items are removed like items; the parent goes when
  its last sub-item does (or when I tick it, with all its sub-items). A
  sync adds a new thread of that topic as a sub-item, not as a new item.

Order inside a section is priority, highest first. I reorder by hand; a sync
appends new items at the end of their section and never reorders the rest.

## Removing items: done and drop

I mark an item done by ticking its box (`- [x]`). In VS Code: click it in
the Markdown Preview Enhanced preview, opened with Ctrl+K V from the
`TODO.md` editor. The editor must stay visible next to the preview: the
click edits the line in it, and does nothing when no editor shows the file.
The built-in preview only ticks the box on screen. Or press Ctrl+Shift+Enter
on the line (Markdown Checkbox extension). `.vscode/settings.json` keeps the line plain and saves
the file automatically. On GitHub: with the pencil (GitHub doesn't make
checkboxes clickable in files). I drop
one by deleting its line. Ticked items stay until the next sync, which
removes them. The list itself only holds open items; the commit messages
keep the record:

- `add: <item>`: a new item.
- `done: <item>`: finished (a PR merged, a reply posted, an idea tried).
  Keep the refs of the item (`sklearn#N`): the sync matches them to know a
  thread was done, not dropped.
- `drop: <item>`: abandoned for now, may come back one day.
- `todo: <what>`: anything else (rewording, moving, reordering).

One commit can hold several of these, one per line of the body (the subject
says what the commit is about, e.g. `sync: 2026-10-30`). `git log --grep` is
line-based, so these find them wherever they are:

```
git log --grep '^drop:'                                   # shelved ideas
git log --grep '^done:' --since 2026-10-01 --format=%B | grep '^done:'
git log -S 'radix sort' -p -- TODO.md todo/               # any past item
git log --diff-filter=D --name-only -- todo/              # deleted detail files
```

Only I drop items. An agent may suggest a drop, but doesn't commit one
without my OK.

## Syncing from GitHub (`/todo-sync`)

`scripts/todo/fetch_activity.py` lists the open threads that involve me
(cakedev0) since the last `sync:` commit, plus every thread already tracked
(linked from `TODO.md` or `todo/*.md`), and says whose move it is (`me`,
`them`, `closed`), why (`self-requested`, `review-requested`, `mentioned`,
`new-activity:<logins>`, `ci-failed`, `conflicts`, `assigned`), and since
when the ball is in my court (`waiting_since`).

Apply these rules:

0. **My edits since the last sync**, in the working tree or in commits
   without `add:` / `done:` / `drop:` / `todo:` lines (e.g. "Update TODO.md"
   from the GitHub editor):
   - Ticked items and sub-items (`- [x]` or `- [X]`, maybe wrapped in `~~` or followed by
     a `[date]` by an editor extension):
     remove them and their detail file, as `done:`.
   - Deleted lines: record them as `drop:` (they are already gone).
   - New lines: record them as `add:`, and fix their format if needed.
1. **Tracked threads**:
   - All the threads of an item are `closed`: remove the item as `done:`
     (as `drop:` if my PR was closed without merging; say so in the report).
   - Move an item between `Now`, `Reviews` and `Waiting` when the move
     changed, and update the "answer X" part. Leave items I wrote by hand
     where they are unless their thread clearly changed state.
   - When an item's `waiting_since` gets more than three weeks old, add
     "since <date>" to its context, so I see what has waited too long.
2. **New threads** (not tracked):
   - Add, in `Now`: my PRs with a review or a comment from someone else
     since my last activity, red CI or conflicts on my upstream PRs, my
     reports with a reply from a maintainer or from the project's
     developers, mentions that ask me something.
   - Add, in `Reviews`: review requests (self-requested ones first), and PRs
     I already reviewed that got new commits or replies.
   - Threads I commented on since the last sync without being the author
     are usually triage: read my last comment and apply the "Triage
     follow-ups" rules of "Giving back".
   - Add, in `Waiting`: my new upstream PRs and reports that are waiting on
     others.
   - Skip: threads where I only commented once and the new activity doesn't
     involve me (other users asking to work on an issue, "same here"
     comments, bots), drive-by comments, and anything out of scope (see
     above). In doubt, skip and list it in the report.
   - Fork PRs (`fork#N`) and benchmark comparison PRs are experiments: put
     them on the `Later` line of their theme (create it if needed), not in
     `Now`, whatever their CI says.
   - An old thread where the ball is in my court is a normal item, with
     "since <waiting_since>" in its context. When it isn't worth doing soon
     (an old RFC, a PR to revive), put it in `Later`, with "follow up on" or
     "revive or close" as its action.
3. **Removed threads** (`removed` is set: the thread was on the list before
   and I removed it). `removed.kind` says how, from the `done:` / `drop:`
   line naming it in the removing commit (`unknown` when none does), and
   `removed.back` whether someone acted on it since (a comment, a review,
   new commits, a review request):
   - `done` and `back`: a new round (more commits on a PR I reviewed, a
     reply to my answer). Add it again, as for a new thread.
   - `done` without `back`: leave it out.
   - `drop` or `unknown`: don't add it back. If `back` and the activity
     needs me (a review request, a mention, a review on my PR), list it in
     the report and ask.
4. **Inbox**: sort each item into its section, with its theme, and the
   links resolved to full URLs. Ask when the right section isn't clear.

Then commit everything in one commit: subject `sync: <date>`, body with one
`add:` / `done:` / `todo:` line per change. Don't push.

## Commits

Commit each change to the list (a sync, a quick add, a done), but don't push:
I push myself. Commit only `TODO.md` and `todo/` in these commits.
