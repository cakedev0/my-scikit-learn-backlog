# Bug hunting in `sklearn/`: instructions for agents

You review scikit-learn code to find bugs and record each one as a Markdown file
in `issues/`, following `bug-hunting/TEMPLATE.md`. Several agents work in
parallel on the same checkout, so read the "Shared checkout" section before
running anything.

Paths like `issues/` and `bug-hunting/` are relative to the backlog repo. The
code you review, `sklearn/...` paths and `git` commands live in the
scikit-learn checkout (`../scikit-learn` from the backlog repo, with an
`upstream` remote on scikit-learn/scikit-learn).

The output is a triaged backlog: a maintainer should be able to read a file,
run the reproducer, and decide whether to open the issue upstream, without
redoing your investigation. A wrong or unverified report costs more than a
missing one: when in doubt, dig deeper or don't file.

## Hard rules

- **Never write to GitHub.** No `gh issue create`, no comments, no PRs, no
  `git push` (the `upstream` remote has push access), including in the
  backlog repo. `gh` is for reading only.
- **Never modify files outside `issues/`.** No edits under `sklearn/`, no
  `pip install`, no rebuild, no `git checkout`/`stash`/`switch` in the shared
  checkout. Reproducers and scratch scripts go in your scratchpad directory.
- **One bug per file, one commit per bug.** Never edit another agent's file,
  except to append a short "See also" line linking your file to it. Commit
  only your own files (step 7), never anything else in the backlog repo.

## What counts as a bug

In scope:

- wrong results (numerical, statistical, or plain logic errors), especially
  silent ones;
- crashes or unhelpful errors on valid input;
- parameters, `sample_weight`, `random_state`, `n_jobs`, metadata routing,
  etc. silently ignored or handled inconsistently (e.g. `fit` vs
  `partial_fit`, dense vs sparse, float32 vs float64, `predict` vs
  `predict_proba`);
- a docstring or user guide promising a behavior the code does not deliver;
- pathological resource usage that is clearly unintended (quadratic memory
  where linear is expected, densifying sparse input, etc.).

Out of scope: feature requests, plain performance improvements, refactoring
or style, typos, code deprecated and scheduled for removal, and private
helpers that cannot be reached from the public API (unless the helper is
"private-public", i.e. widely used by downstream projects).

Before filing, rule out that the behavior is intended: read the docstring, the
user guide, the tests covering the code (a test asserting the current behavior
is a strong hint it's deliberate), and `git log -L` / `git blame` on the lines
involved.

## Workflow for each bug

### 1. Understand the code

Read the surrounding code until you understand what it is meant to do, not
just the suspicious lines. Write down the precise conditions under which the
bug triggers.

### 2. Check for local duplicates

Search `issues/` for the estimator, function, and key identifiers:

```bash
grep -ril "<EstimatorName>\|<function_name>" issues/
```

If another file already covers the same root cause, stop (or add a "See also"
line if yours adds a meaningfully different trigger). Check again right before
writing your file: other agents may have filed it in the meantime.

### 3. Search upstream

Look for existing issues and PRs on `scikit-learn/scikit-learn`, open and
closed. Try several queries: estimator/function name, the error message, the
parameter involved, and a plain-words description of the symptom.

```bash
gh search issues --repo scikit-learn/scikit-learn --include-prs --limit 30 <term1> <term2> ...
gh search issues --repo scikit-learn/scikit-learn --include-prs --match comments <term1> <term2> ...
gh issue view <number> --repo scikit-learn/scikit-learn --comments
gh pr view <number> --repo scikit-learn/scikit-learn
```

Pass the search words to `gh search` as separate, unquoted arguments: a
quoted argument containing spaces is searched as an exact phrase and misses
most matches.

Also check whether it was recently fixed on main:

```bash
git log upstream/main --oneline -15 -- <path/to/file.py>
grep -rl "<EstimatorName>" doc/whats_new/upcoming_changes/
```

Then:

- **open issue describing this bug**: file it anyway, set `existing_issue` to
  its URL, and still verify the reproducer on the current code. Mention any
  open PR attempting a fix.
- **fixed on main by a merged PR**: don't file.
- **closed as "not a bug" / "won't fix" by maintainers**: don't file, unless
  you have a new argument that the discussion missed; then file it, link the
  issue and state the new argument explicitly.
- **related but different** (same estimator, other symptom; or same symptom,
  other cause): list it under "Upstream search" and say why it differs.

The GitHub search API is rate limited (about 30 queries per minute, shared by
all agents). Keep it to a handful of focused queries per bug, and if you hit
the limit, wait a minute rather than retrying in a loop. `gh issue list` and
`gh pr list` with `--state all --search "<terms>"` use a separate quota and
work as a fallback, but they don't search comments: say so under "Upstream
search" when you had to rely on them.

### 4. Write a minimal reproducer

Follow the scikit-learn guidelines
(https://scikit-learn.org/dev/developers/minimal_reproducer.html):

- self-contained and copy-pastable, no external data files: synthetic data
  from `numpy.random` (seeded) or `sklearn.datasets.make_*`;
- public API only. If you had to use private API, explain why, since it lowers
  the user-facing impact;
- as short as possible: remove every line that isn't needed to trigger the bug;
- make the bug obvious: print the wrong value next to the right one, compare
  against a reference (closed form, brute force, scipy, a sibling estimator,
  `sample_weight` vs repeated samples, ...), or show the traceback;
- run it and paste the actual output under "Actual Results". Never write
  output you didn't observe.

If you can't trigger the bug, you were probably wrong: don't file. The only
exception is a bug that cannot be triggered on the available machines
(Windows-only, 32-bit, specific BLAS, race condition, ...) but is certain from
reading the code. Then set `"reproduced": false`, and replace the reproducer by
the most direct evidence you have (an argument with `file:line` references,
and a reproducer that should trigger it on the right platform).

### 5. Classify

Fill the three levels using the scales below, with one or two sentences of
justification each under "Classification". Be calibrated: most real bugs are
priority 1 or 2, and a 4 should be exceptional.

### 6. Write the file

Copy `bug-hunting/TEMPLATE.md` to `issues/<module>-<short-slug>.md`, where
`<module>` is the first-level subpackage the bug lives in (`linear_model`,
`metrics`, `utils`, ...) and `<short-slug>` is a few kebab-case words, e.g.
`issues/linear_model-ridge-sample-weight-intercept.md`. If the name is taken,
read that file first (it may be the same bug), otherwise pick another slug.

Fill every section. Keep the JSON block valid JSON (no comments, double quotes),
as it will be parsed by scripts. Write the prose so it could be pasted into an
upstream issue: concise, user impact first, no filler. If the
`writing-style` skill is available, load it.

### 7. Commit it

Commit the new file, alone, in the backlog repo (`~/my-scikit-learn-backlog`),
with the title of the file as message, prefixed by the module:

```bash
cd ~/my-scikit-learn-backlog
git add issues/<file>.md
git commit -m "<module>: <title>" -- issues/<file>.md
```

Passing the paths after `--` commits only them, even if other agents have
staged files at the same time. If you appended a "See also" line to another
file, add that path to the same commit. If git fails because
`.git/index.lock` exists, another agent is committing: wait a few seconds and
retry, never delete the lock. Don't push.

## Classification scales

### `estimators`

Public estimators or functions touched by the bug, by class/function name
(e.g. `["Ridge", "RidgeCV"]`, `["roc_auc_score"]`). If many, list a few
important ones and describe the full set in the text.

### `existing_issue`

URL of the open upstream issue describing this exact bug, or `null`.

### `reproduced`

`true` if you ran the reproducer and observed the bug, `false` otherwise (see
step 4).

### `priority` (0 to 4)

How much we want this fixed.

- **0**: we never want to solve it, it's an extremely unrealistic edge case.
- **1**: we'd ultimately like to fix it, but probably will never have time.
- **2**: in between 1 and 3.
- **3**: it's quite bad or misleading, on a relatively important estimator;
  let's fix it soon.
- **4**: critical, we should absolutely fix it by the next release (quite
  unlikely to find one).

What pushes priority up: silent wrong results (vs. a loud error), default or
common settings, a widely used estimator, no easy workaround, results that
look plausible so users won't notice. What pushes it down: exotic inputs or
parameter combinations, a clear error message, a cheap workaround, impact only
on numerical noise.

Calibration examples from the maintainer:

- `PolynomialFeatures` giving silently wrong products on CSR input with
  unsorted indices: **2**, not 3. Sparse input is already a minority of uses,
  and unsorted indices are an edge case within it. Silent wrong results don't
  make a 3 on their own: the trigger must be reasonably common.
- HistGradientBoosting applying `interaction_cst` and recursion partial
  dependence to the wrong features when a categorical feature is not first:
  a **low 3**. The parameter combination is somewhat niche, but the results
  are silently wrong on an important estimator.
- Label handling bugs giving wrong results (integer labels not encoded,
  `pos_label` ignored, outliers assigned to the wrong class): **3** is fair.
  Labels are part of every classification call, so these are easy to hit.
- `KMeans.predict` crashing when the input dtype differs from the one seen in
  `fit` (fit on float32, predict on a list or a float64 array): **3**, not 2.
  A loud error doesn't make a bug minor when it breaks the basic fit/predict
  workflow of a widely used estimator on valid input, with an error message
  that doesn't say what to do.
- `LogisticRegression` with `newton-cd` crashing on float32 data with 3+
  classes (reached by `solver="auto"` only with `l1_ratio > 0` and wide data):
  **2**. Same kind of crash as `KMeans.predict` on an even more important
  estimator, but the combination of conditions is too niche for a 3.
- `DBSCAN(metric="minkowski")` crashing with an unrelated-looking `TypeError`
  because of its default `p=None`: **2**, not 1. The metric isn't the default
  but is a natural choice on a widely used estimator.

### `difficulty` (1 to 4)

How hard the fix is to write and to review.

- **1**: a few lines to change; easy to understand why the old lines were
  wrong and why the new ones are correct.
- **2**: might require changes in several functions, or a small change whose
  correctness is not obvious.
- **3**: requires understanding in depth some complex technical, statistical
  or algorithmic aspect of the code.
- **4**: the algorithm is deeply wrong and basically needs a rewrite (and it's
  not a five-line algorithm).

### `decision` (0 to 2) and `decision_tags`

Does the fix imply trade-offs?

- **0**: brainless improvement.
- **1**: minor trade-offs; clearly worth fixing, but there is something to be
  aware of.
- **2**: real trade-off; not sure we even want to fix it.

Tags, listing every trade-off the fix involves (the level weighs them; a
decision 0 usually has no tags, or maybe just something like rng-order):

- Behavior change:
  - `api-change`: a change of some public API is needed (or "private-public"
    API, i.e. private but used by many other projects).
  - `new-error`: something that used to work, possibly giving useful results,
    now raises.
  - `algo-change`: change of algorithm that does not clearly make it match
    better what the paper says or what users would expect.
  - `float-order`: floating-point summation order changes.
  - `rng-order`: random generation changes, due to calling the RNG differently.
  - `output-change`: the fix is correct but changes results noticeably (beyond
    floating-point noise) for the same data and `random_state`, e.g. under
    default settings.
  - `new-warning`: the fix warns instead of raising; a cost if the warning
    fires often in normal use.
  - `deprecation`: the fix needs a deprecation cycle (e.g. a default value
    change with a `FutureWarning`), so the bug stays in released versions for
    two more releases and the code carries both paths meanwhile.
- `unclear-spec`: there's no agreed definition of the correct behavior (e.g.
  what `sample_weight` should mean for this estimator, a metric's convention
  on a constant `y` or a single class); someone must decide what "fixed" means
  first.
- `dependency`: the fix needs a newer minimum numpy/scipy version, a new
  optional dependency, or vendored code.
- `complexity`: the fix adds a lot of code to maintain for the problem it
  solves (special-casing, a second code path, new Cython code). Difficulty is
  how hard the fix is to write; this is whether the extra code is worth
  keeping.
- `perf-low`, `perf-medium`, `perf-high`: performance regression (time or
  memory), or risk of one, of that magnitude.

## Shared checkout

- Use the active Python environment and the existing editable build as is.
- The bug must exist on upstream main. If HEAD is not `upstream/main`, check
  that `git diff upstream/main -- <file>` shows nothing near the buggy code;
  if it does, the bug may come from a local branch: skip it.
- Many agents run at once on the same machine: run reproducers with
  `OMP_NUM_THREADS=2` (unless the bug is about threading) and keep them under
  ~10 s. Don't run the test suite beyond a targeted `pytest -k` on a file.
  Anything CPU-heavy goes under `benchlock` (see `benchlock --help`).

## Final checklist

- [ ] Searched `issues/` and upstream (issues, PRs, recent commits on main).
- [ ] Ruled out intended behavior (docs, tests, blame).
- [ ] Reproducer is minimal, public API, synthetic data, and was actually run.
- [ ] Actual output pasted, not paraphrased.
- [ ] Three levels filled and justified; tags consistent with the decision
      level.
- [ ] JSON block is valid; `existing_issue` and `reproduced` set.
- [ ] Root cause points to `file:line`; versions include the commit hash.
- [ ] Committed alone in the backlog repo, not pushed.
