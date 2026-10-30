---
description: Add an item to my todo list (TODO.md), mark one done or dropped, or show what's next.
argument-hint: [<item to add> | done <item> | drop <item>]
---

# Todo list

Paths: `<backlog>` is the root of the my-scikit-learn-backlog clone (the
current directory if it contains `bug-hunting/`, otherwise
`~/my-scikit-learn-backlog`). Run git in it with `git -C <backlog>`.

Read `<backlog>/todo/INSTRUCTIONS.md` for the format and the commit
conventions, then act on `$ARGUMENTS`:

- **Empty**: show `Now` and the first items of `Reviews`, as they are in
  `<backlog>/TODO.md`. Don't edit anything.
- **`done <item>` or `drop <item>`**: find the item (by words or by a
  thread number), remove its line and its detail file if any, and commit
  with `done: <item line>` or `drop: <item line>` as the subject (without
  the URLs, keep the refs). If several items match, ask.
- **Anything else**: a new item.
  - Write it in the item format (`- [ ] <Theme>: ...`), with its theme. Turn issue and PR numbers
    into full URLs (`#N` and bare numbers are scikit-learn/scikit-learn
    unless the text says otherwise), and check their titles with
    `gh issue view` when it helps the context.
  - Put it in the section it obviously belongs to, otherwise in `Inbox`.
  - When I give more than fits on a line (a plan, a list of steps, links),
    put the rest in a new `todo/<slug>.md` and link it.
  - Commit with `add: <item line>` as the subject.

Commit only `TODO.md` and `todo/`, and don't push. Reply in one or two
lines: the item as written and the section.

Request: $ARGUMENTS
