# Traps and failures

Things that went wrong or misled us, so we don't repeat them.

## Measurement traps

- **The "GCC 14 libgomp" that was libomp.** In pixi envs built with
  `_openmp_mutex=*_kmp_llvm` (e.g. `sklearn-mkl`, `sklearn-cf-libomp-openblas`),
  `lib/libgomp.so.1` is a symlink to LLVM `libomp.so`. Its strings contain
  "GCC: (conda-forge gcc 14.3.0)" (the compiler that built it). Always `readlink -f` and
  check the strings (`hybrid core type detected` = libomp) before calling it libgomp.
- **`LD_LIBRARY_PATH` swaps the OpenMP runtime.** Pointing `LD_LIBRARY_PATH` to an env's
  `lib/` to load MKL also loaded that env's `libgomp.so.1`, which confounded a
  MKL-vs-OpenBLAS comparison. `c/blas_region.c` now prints the loaded libgomp. Load the BLAS
  by full path (`dlopen`) and keep the OpenMP runtime fixed.
- **Default `GOMP_SPINCOUNT` differs per libgomp and per CPU.** Old libgomp: 300000.
  Recent libgomp on hybrid CPUs: no spinning by default. Set it explicitly in experiments.
- **strace distorts timings completely** (2 ms -> 6 ms regions, MKL appearing to make 400k
  futex calls). Use it only for syscall counts, and even counts are biased (spinning times
  out). The `LD_PRELOAD` mutex counter was much more informative.
- **Single runs on laptops are noisy.** One "matrix" run showed small Lloyd 1.5-1.8x slower
  on a branch, best-of-2 showed it was noise. Background load (this session, a browser, a
  previous benchmark's load average) changes results; wait for the load to drop and repeat.
- **Laptop power states.** Waking OpenBLAS threads before Lloyd made `main` look ~30%
  faster than a branch, with identical Lloyd code and iterations.
- **Different random init paths change iteration counts.** Compare Lloyd with a fixed
  `init` array and `tol=0`, or report ms/iteration, not total fit time.
- **"lloyd/it" from `(fit - init) / n_iter`** mixes fixed costs into per-iteration numbers.
- **Comparing Lloyd from each build's own k-means++ centers** is confounded: different
  centers give different numbers of empty clusters, whose relocation is expensive (a build
  that doesn't touch Lloyd showed 1.23x). Use the same initial centers for all builds.
- **Same tag for two machines** in a results table (`small14` on the GNR and on the laptop)
  silently merged them. Keep the machine in the tag or tabulate files separately.
- **Hybrid CPUs give bimodal timings** for big Lloyd fits (~1.4x between modes, same build).
  Use 5+ interleaved rounds and look at minima and medians, not one or two runs.
- **NumPy buffers allocated per iteration in the main thread** cost page faults and land on
  the wrong NUMA node for big per-thread buffers. Let each thread allocate its own.
- **Doctests are skipped** in a venv without the optional dependencies (the
  `kmeans_plusplus` example would have failed silently). Run them with
  `doctest.run_docstring_examples` or install the deps.
- **Calibrating a fallback single-threaded** underestimated its multi-threaded cost on a
  hybrid CPU by ~10x (plain loop, E14).
- **Results don't transfer between machines.** The GNR (no hybrid cores, needs spinning)
  and the laptops (hybrid, spinning is dangerous) have different dominant problems. Check
  any conclusion on both kinds before generalizing.

## Mistakes in the reasoning

- "#17334 is not about BLAS": wrong as a general statement. On the GNR the OpenBLAS +
  spinning OpenMP oversubscription is the dominant cause; on hybrid laptops it's blocking
  inside parallel regions. Both are real.
- "Using all cores with spinning is fragile even for pure computation": only true on a busy
  laptop, not on an idle one (intel-laptop).
- "A plain loop isn't much slower than BLAS for small problems": false for d >= 10 with
  baseline-x86-64 compiled code.
- First k-means++ prototype wrote the first center's distances in the buffer read as
  `closest_dist_sq` by step 1 (aliasing), only caught while rewriting.

- **Cython `prange` loops inside a `with parallel()` block are `omp for nowait`.** Two
  consecutive pranges in one parallel region have no barrier between them (checked in the
  generated C, Cython 3.3). A second loop reading the first loop's output races; tests
  passed anyway. Use two separate parallel regions with the same `num_threads`.
- **Splitting the k-means++ gemm by rows when n_chunks < n_threads** (2026-10-12, review of
  #23): with a different team size in the two regions it was 4x slower on 3000 x 784 f32;
  with the same team size still 2.6x slower than the PR there, for 15% on 2000 x 5000 f64.
  Dropped. Wide f64 data with few samples stays ~1.6x slower than main on intel-laptop.

## Tooling traps

- **Socket-slot results are not publishable** (`benchlock` rule): runs on the GNR's `s0`/`s1`
  are for quick A/B only. Numbers for a PR description must come from the `all` slot. The MKL
  table of E23 (s0) was briefly put in #26's description by mistake and removed.

- Git LFS records of scikit-learn-benchmarks: `git lfs pull` silently does nothing on the
  GNR. Fetch via `https://media.githubusercontent.com/media/probabl-ai/scikit-learn-benchmarks/refs/heads/main/results/records/<file>`.
- Don't use `git stash`/`pop` to swap code for benchmarks (used once on the GNR, harmless
  by luck). Use worktrees.
- `pkill -f <pattern>` over ssh matches its own shell command line and kills the ssh
  session. Use a pattern like `"run_smal[l]"`.
- `ssh host 'nohup cmd &'` keeps the ssh session open (inherited fds). Use
  `setsid nohup ... > file 2>&1 < /dev/null & disown`.
- `sed "s|...|$tag|"` breaks when `$tag` contains the delimiter; use `#`.
- A heredoc inside `ssh '...'` with single quotes in the content fails to parse, but parts
  before the error may still have run (a kernel got appended twice).
- `git cherry-pick` has no `-q`.
- `X.size` is `n * d` for arrays and `nnz` for scipy sparse matrices (handy for work
  estimates).
- The GNR has a GitHub Actions runner and other users: check `uptime`/`ps` before
  benchmarking and ask before running heavy things.
- OpenBLAS from the wheels crashes with more than 64 OpenMP threads calling it
  (OpenBLAS#5958): cap experiments at 64 threads.

- **`VAR=x run ...` with `run() { env -u VAR ...; }`** silently drops the assignment
  (`env -u` removes what the prefix just exported). Inherited from `run_spin_exps.sh` into
  the yield scripts, it made `OPENBLAS_NUM_THREADS=1` a no-op and invalidated E27/E28's E
  rows (multi-threaded OpenBLAS inside OpenMP regions). Unset variables once at the top of a
  script, never inside the helper that the caller prefixes.
