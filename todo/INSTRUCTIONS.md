# Todo list: how to keep it

Instructions for the agents (and me) that edit `TODO.md` and the files of
`todo/`. The list is my work todo list as a scikit-learn maintainer at
Probabl, mostly on the Intel / Probabl collaboration to make scikit-learn
faster on Intel hardware (see the [announcement][announcement] and the
[progress post][post]). Its "What's next" section is the roadmap the `Later`
items come from.

[announcement]: https://blog.probabl.ai/intel-and-probabl-announce-collaboration-to-accelerate-scikit-learn-on-intel-hardware
[post]: https://github.com/scikit-learn/blog/pull/228

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
- `todo/stale.md`: threads where the ball has been in my court for more than
  three weeks. It has one item in `Now`, so I review them in one sitting:
  each one either moves to `TODO.md` or gets dropped.

A bug of the backlog (`issues/`) only becomes an item when I decide to act
on it, and its line links to the `issues/` file as its details. Don't copy
the bug backlog into the list.

## Item format

```
- <Theme>: <action>, [<ref>](<url>) (<few words of context>) → [notes](todo/<slug>.md)
```

- **Theme**, one of: HGB, KMeans, Trees, Linear, Encoders, Preprocessing,
  Threads, Array API, GPU, Docs, Benchmarks, Blog, Upstream (bugs and PRs in
  other projects, e.g. sklearnex, oneDAL, CPython), Maintenance (triage,
  reviews, CI). Add a theme only when none fits.
- **Action**: start with a verb when there is something to do ("answer
  lorentzenchr on", "fix the CI of", "upstream or close"). Name who to answer.
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

Order inside a section is priority, highest first. I reorder by hand; a sync
appends new items at the end of their section and never reorders the rest.

## Removing items: done and drop

Items are removed, never ticked. The commit message keeps the record:

- `add: <item>`: a new item.
- `done: <item>`: finished (a PR merged, a reply posted, an idea tried).
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
`them`, `closed`) and why (`review-requested`, `mentioned`,
`new-activity:<logins>`, `ci-failed`, `conflicts`, `assigned`).

Apply these rules:

1. **Tracked threads**:
   - All the threads of an item are `closed`: remove the item as `done:`
     (as `drop:` if my PR was closed without merging; say so in the report).
   - Move an item between `Now`, `Reviews` and `Waiting` when the move
     changed, and update the "answer X" part. Leave items I wrote by hand
     where they are unless their thread clearly changed state.
   - A `Now` item whose thread hasn't moved for more than three weeks goes
     to `todo/stale.md`.
2. **New threads** (not tracked):
   - Add, in `Now`: my PRs with a review or a comment from someone else
     since my last activity, red CI or conflicts on my upstream PRs, my
     reports with a reply from a maintainer or from the project's
     developers, mentions that ask me something.
   - Add, in `Reviews`: review requests, and PRs I already reviewed that got
     new commits or replies.
   - Add, in `Waiting`: my new upstream PRs and reports that are waiting on
     others.
   - Skip: threads where I only commented once and the new activity doesn't
     involve me (other users asking to work on an issue, "same here"
     comments, bots), drive-by comments, and anything out of scope (see
     above). In doubt, skip and list it in the report.
   - Fork PRs (`fork#N`) and benchmark comparison PRs are experiments: put
     them on the `Later` line of their theme (create it if needed), not in
     `Now`, whatever their CI says.
   - A thread older than three weeks where the ball is in my court goes to
     `todo/stale.md` rather than `Now`.
3. **Removed threads** (`removed` is set: the thread was on the list before
   and I removed it): don't add it back. If there was new activity that
   needs me after the removal date (a review request, a mention, a review on
   my PR), list it in the report and ask.
4. **Inbox**: sort each item into its section, with its theme, and the
   links resolved to full URLs. Ask when the right section isn't clear.

Then commit everything in one commit: subject `sync: <date>`, body with one
`add:` / `done:` / `todo:` line per change. Don't push.

## Commits

Commit each change to the list (a sync, a quick add, a done), but don't push:
I push myself. Commit only `TODO.md` and `todo/` in these commits.
