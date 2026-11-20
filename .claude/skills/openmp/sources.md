# Sources

States as of 2026-11-20. Check with `gh` before quoting one. Local paths are
on this laptop unless a host is given. Several original writeups are untracked files or
live in worktrees: if a path is gone, use the copy in `writeups/` or look for
the branch named next to it.

## Issues and PRs

Wait policy and runtimes:

- scikit-learn/scikit-learn#34764, open: impact of active wait in scikit-learn
  (the overview, with proposed directions).
- scikit-learn/scikit-learn#34437, open: the Linux wheel bundles GCC 8.5's
  libgomp. Comment of 2026-11-13 on the macOS libomp pin.
- scikit-learn/scikit-learn#17334, open: KMeans slow with OpenBLAS. Comments
  of 2026-08-13 (libomp doesn't stall), 2026-10-13 (the two mechanisms, 3
  machines, yield) and 2026-10-21 (OpenBLAS lock fixed, what is left).
- cakedev0/sklearn-contrib-emeans#8, closed: random 5x slower fits on hybrid
  CPUs. Fixed by PR #9 (one region per epoch), whose description has the real
  cause and the benchmarks.
- cakedev0/sklearn-contrib-emeans#15, closed, and
  pypackaging-native/pypackaging-native#50, open: two libomp on macOS.
- probabl-ai/scikit-learn-benchmarks#90, open: a team from one call pollutes
  the next timing.

Cython:

- cython/cython#8120 (closed) and cython/cython#8121 (merged): end barrier
  lost in 3.3, Fix A.
- cython/cython#8129, open: clang hang with the GIL held.
- cakedev0/cython#4, open on the fork: Fix B, no start barrier.

KMeans and HGB:

- scikit-learn/scikit-learn#35140, merged: k-means++ init without
  multithreaded BLAS.
- scikit-learn/scikit-learn#35193, open: no lock, no GIL, reused buffers in
  Lloyd. Benchmarks in probabl-ai/scikit-learn-benchmarks#154.
- scikit-learn/scikit-learn#34935, open: HGB thread heuristics and caps.
  Dashboards in probabl-ai/scikit-learn-benchmarks#81.
- cakedev0/scikit-learn#47, open draft: one region per fit for HGB.
- scikit-learn/scikit-learn#30662, open: HGB 15x slower with threads on small
  data.

BLAS and pools:

- scikit-learn/scikit-learn#35180 (merged) and joblib/threadpoolctl#283
  (merged, in 3.8.0): `sequential_blas_under_openmp`.
- OpenMathLib/OpenBLAS#5958 (closed): crash above 64 concurrent callers.
  OpenMathLib/OpenBLAS#6105 (merged): the per-call lock.
  OpenMathLib/OpenBLAS#5872 (merged, 0.3.34): thread caps for small calls.
  cakedev0/OpenBLAS#8, open on the fork: fewer threads for small calls.
- scikit-learn/scikit-learn#28883, open: OpenBLAS on scikit-learn's OpenMP
  pool.
- joblib/joblib#1869, open: loky thread counts vs physical cores.
- scikit-learn/scikit-learn#35091, open: lock contention with free-threading.
- probabl-ai/scikit-learn-benchmarks#80, #129, #145: multi-socket
  variability, TBB under joblib, laptop throttling.

## Local writeups

The main ones are copied in [writeups/](writeups/README.md), and the code
ideas in [snippets.md](snippets.md). The originals below have the raw logs,
scripts and results.

| What | Where |
| --- | --- |
| KMeans investigation: findings (2 blocking points, 3 OpenBLAS mutex, 4 and 4b hybrid stall and yield, 5 passive on servers, 8 team sizes), experiments E1-E33, traps, status | `~/scikit-learn/kmeans_learnings/` (untracked): `findings.md`, `experiments.md`, `traps.md`, `status.md`, `scripts/` |
| libgomp yield patches | same folder: `libgomp_spin_yield.patch` (GCC 13.3 / 16.1), `libgomp_spin_yield_trunk.patch` (default 100, docs) |
| OpenBLAS patches (memory slot lock, overflow) | same folder, `openblas_*.patch` |
| Cython prange barrier: findings, log of experiments, drafts of the issues and PRs, audit and TSan scripts | `~/scikit-learn-benchmarks/.claude/worktrees/cython-prange-barrier/cython-prange-barrier/` (`FINDINGS.md`, `LOG.md`, `drafts/`, `scripts/`), branch `cython-prange-barrier` |
| HGB parallel regions: wake-up cost per C-state, team regrow, merged regions, deep idle states | `~/scikit-learn-benchmarks/.claude/worktrees/hgb-regions-exp/hgb-regions-experiments/` (`FINDINGS.md`, `LOG.md`), branch `hgb-regions-experiments` |
| GEMM, OpenBLAS vs MKL: thread placement, cliffs, patches | `~/scikit-learn-benchmarks/.claude/worktrees/gemm-experiments/gemm-experiments/FINDINGS.md`, branch `gemm-experiments` (pushed) |
| HGB one region per fit: PR body with all the tables, scripts, raw results | `~/work/hgb-team-bench/` |
| Kernel scheduler report (draft, not sent) | `~/scikit-learn/kernel_sched_report.md`, reproducer `intel-laptop:~/omp-stall/spin_pair` |
| Forest scaling with joblib threads | `~/scikit-learn/tree_parallel_findings.md` |
| macOS libomp issue drafts | `~/sklearn-contrib-emeans/llvm-libomp-macos-exports-issue.md`, `pypackaging-*.md` |

## Code to read

- Handwritten barrier and one region per epoch:
  `~/sklearn-contrib-emeans/sklearn_emeans/_kernels.pyx`, the verbatim C block
  at the top (`emeans_fetch_add`, `emeans_wait`, `emeans_buffers_ready`), then
  `_run_steps` and `_steps_worker`.
- Team of threads driven by Python:
  `~/sklearn-contrib-trees/sklearn_trees/_hist_gradient_boosting/_team.pyx`.
- libgomp: `libgomp/config/linux/wait.h` (`do_spin`),
  `libgomp/config/linux/x86/spincount.h` (hybrid default), `libgomp/env.c`.
- libomp: `openmp/runtime/src/kmp.h` (`KMP_DEFAULT_BLOCKTIME`,
  `KMP_INIT_WAIT`, `KMP_NEXT_WAIT`), `kmp_wait_release.h`.

## On the benchmark machines

- intel-xeon: `~/kmeans-spin/` (worktrees, venvs, `results*.txt`, MKL env in
  `mkl/`, patched libgomp in `gcc-src/build-gomp16/.libs/libgomp.so.1.0.0`,
  yield shim in `gomp-yield/`), `~/kmeans-bench/`, `~/sparse-1t/`. sklearn's
  extensions there have an RPATH to the pixi gcc env: swap libgomp with
  `LD_PRELOAD`.
- intel-laptop: `~/kmeans-spin/` (patched GCC 13.3 libgomp in
  `gcc-src/build-gomp/.libs/`, C tests in `c/`), `~/omp-stall/`, `~/bench26/`.
- `git lfs pull` silently does nothing on intel-xeon. Fetch a record with
  `https://media.githubusercontent.com/media/probabl-ai/scikit-learn-benchmarks/refs/heads/main/results/records/<file>`.
- Branches, worktrees and fork PRs of the KMeans work (cakedev0/scikit-learn
  #23, #24, #26 and the local `../sklearn-kmeans-*` worktrees) are listed in
  `kmeans_learnings/status.md` and `README.md`. Check that they still exist
  before reusing them.

## Keeping this skill current

When a session learns something new on these topics, or finds one of these
claims wrong, edit the reference file (and SKILL.md if the one-page model
changes) in the same session, with the date and the machine. Keep SKILL.md
short: details go to the reference files, raw results stay in the writeups.
