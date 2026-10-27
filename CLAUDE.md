# Backoffice for scikit-learn triage

See `README.md` for the layout and the commands.

- The commands in `.claude/commands/` are also symlinked from
  `~/.claude/commands/`, so they run from the scikit-learn checkout too. Keep
  their paths as `<backlog>/...` and `<sklearn>`, not relative paths.
- The scikit-learn code lives in `../scikit-learn` (remote `upstream` is
  scikit-learn/scikit-learn); `../scikit-learn/issues` is a symlink to
  `issues/` here. Read upstream code with
  `git -C ../scikit-learn show upstream/main:<path>`. Never switch its branch
  or stash in it: other agents use it. Use a worktree if you need PR code.
- Never write to GitHub (comments, labels, reviews, issues, PRs) unless I ask
  after seeing the report.
- New backlog entries follow `bug-hunting/INSTRUCTIONS.md` and
  `bug-hunting/TEMPLATE.md`. Keep the JSON block of each file valid JSON.
  One new backlog entry = one commit (`<module>: <title>`), don't push.
