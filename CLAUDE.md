# Backoffice for scikit-learn triage

See `README.md` for the layout and the commands.

- Run commands and scripts from the root of this repo: the commands call
  `scripts/...` with relative paths.
- The scikit-learn code lives in `../scikit-learn` (remote `upstream` is
  scikit-learn/scikit-learn). Read upstream code with
  `git -C ../scikit-learn show upstream/main:<path>`. Never switch its branch
  or stash in it: other agents use it. Use a worktree if you need PR code.
- Never write to GitHub (comments, labels, reviews, issues, PRs) unless I ask
  after seeing the report.
- New backlog entries follow `bug-hunting/INSTRUCTIONS.md` and
  `bug-hunting/TEMPLATE.md`. Keep the JSON block of each file valid JSON.
