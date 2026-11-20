# OpenMP next to BLAS and the other thread pools

Links and file paths are in [sources.md](sources.md).

## Which pools exist in a process

| Install | OpenMP | BLAS |
| --- | --- | --- |
| PyPI wheels | libgomp 8.5 bundled by scikit-learn (libomp 11 on macOS) | two OpenBLAS pthreads pools (numpy's and scipy's `scipy-openblas`, `MAX_THREADS=64`) |
| conda-forge default | recent libgomp | one OpenBLAS, pthreads or OpenMP build |
| conda-forge MKL | LLVM libomp (through `_openmp_mutex`) | MKL on libomp |
| sklearnex | plus TBB for oneDAL | |

Check with `threadpool_info()`, one entry per loaded library, and limit one
pool at a time by filepath when looking for a culprit. gdb breakpoints on
`exec_blas` tell which OpenBLAS call wakes a pool.

## OpenBLAS

- **pthreads pool next to spinning libgomp**: scikit-learn#17334 on servers.
  The OpenBLAS workers also spin after a call (`OPENBLAS_THREAD_TIMEOUT`).
  Fixed in scikit-learn by an init without multithreaded BLAS (#35140, merged).
  Reproducer on PyPI wheels, 8 CPUs of the Core Ultra 5: 12.8 s by default,
  1.28 s with `OMP_WAIT_POLICY=passive`, 4.1 s with
  `OPENBLAS_THREAD_TIMEOUT=4`, 3.8 s with numpy's pool capped, no change with
  scipy's pool capped. So it is libgomp against numpy's pool.
  scikit-learn#28883 (OpenBLAS on scikit-learn's OpenMP pool) is the other
  direction.
- **Global mutex on every call**, single-threaded ones too: 2
  `pthread_mutex_lock` per `gemm`, ~35% contended with 14 callers. MKL
  sequential takes none. Fixed by OpenMathLib/OpenBLAS#6105 (merged
  2026-10-21): check which scipy-openblas release has it. Until then, many
  small BLAS calls from OpenMP threads scale badly (Lloyd stops scaling at ~16
  threads) and stall a spinning team on hybrid CPUs.
- **More than ~64 concurrent callers crash** (buffer table exhausted,
  OpenMathLib/OpenBLAS#5958), even with OpenBLAS limited to 1 thread. Hence
  the `OMP_NUM_THREADS=64` cap for KMeans on the Xeon. Uncapped at 172
  threads (2026-11-04), crashes per env: PyPI 12/12, conda OpenBLAS pthreads
  9/12, conda OpenBLAS OpenMP 11/12, MKL and intel 0/12. At the cap: 0/60. So
  it isn't specific to the PyPI wheel. conda's OpenBLAS warns above 128
  threads, caps between 64 and 128 are untested. Uncapped isn't faster where
  it runs (MKL 0-30% slower per iteration, sklearnex 25-70%). Arthur wants the
  cap on the scikit-learn + OpenBLAS envs only, no limit elsewhere, and thread
  caps must not split cases on the comparison pages (branch
  `kmeans-openblas-only-cap` of scikit-learn-benchmarks).
- **Thread-count cliffs**: `syrk` runs single-threaded when `n_features < 16
  * n_threads` (990 features: 979 GF/s at 60 threads, 81 at 62). No split over
  k for small outputs. Small level 1/2 calls and getrf/gesv use too many
  threads (cakedev0/OpenBLAS#8).
- **`LogisticRegression` (lbfgs) slower with more threads on PyPI installs**:
  not OpenMP (LR runs its loss on 1 thread in 1.8). scipy 1.18.1 vendors
  OpenBLAS 0.3.31.dev, where the `dtrtrs` calls of L-BFGS-B on 10x10 matrices
  wake the whole pool. Its workers then spin and compete with numpy's pool,
  which runs the GEMMs. Fixed in OpenBLAS 0.3.34 (OpenMathLib/OpenBLAS#5872,
  `m < 64` runs on one thread). scipy main has it (scipy#26193, milestone
  2.0.0), and the backport to 1.18.2 was still open on 2026-10-01
  (scipy#26199): check what is released. With one shared OpenBLAS 0.3.34
  (conda-forge): 1.11 s vs 2.37 s on 8 CPUs.
- Single-thread kernels of OpenBLAS and MKL are within ~7%. Gaps above 1.5x
  come from threading: how the work is split, how many threads, where they run.

## threadpoolctl

- scikit-learn limits BLAS to 1 thread inside its OpenMP loops (KMeans,
  ArgKmin) with `limits="sequential_blas_under_openmp"` since #35180.
- threadpoolctl <= 3.7.0 skips that limit for every OpenBLAS, pthreads too:
  GEMMs in the Lloyd loop run multithreaded (7 s to more than 60 s per fit on
  a Xeon socket, 300 s time limit hit in `all_models.py`). MKL builds are not
  affected. Fixed in 3.8.0 (threadpoolctl#283). Benchmark scikit-learn main
  with threadpoolctl >= 3.8, and check the version in the env first. If a
  pixi env of scikit-learn-benchmarks is still locked on 3.7.0: `pixi run -e
  <env> python -m pip install --no-deps --ignore-installed "threadpoolctl>=3.8"`
  (it survives `setup_sklearn_ref.sh`), and say so in the PR, since the
  recorded software env still shows the locked version.
- threadpoolctl can't apply a thread binding: MKL ignores binding set after
  its first `mkl_get/set_num_threads` call.

## MKL and libomp placement

conda-forge MKL on LLVM libomp, unbound, stacks threads on shared cores from
43 threads up on intel-xeon: timings are bimodal from process to process
(1.5-2.5x). Fixed by `OMP_PROC_BIND=close OMP_PLACES=cores`,
`KMP_BLOCKTIME=infinite` or Intel's libiomp5. Same with libgomp. Floating
libomp threads at 86 threads were also 1.6x slower in steady state than pinned
ones on KMeans. So distrust unbound MKL numbers at high thread counts, and add
a bound variant when comparing with OpenBLAS.

## joblib and loky

loky gives each worker `logical_cpus // n_jobs` threads through
`OMP_NUM_THREADS`, and scikit-learn uses that value as is, while its own
default is physical cores only. With few workers this oversubscribes: on the
Xeon, `n_jobs=2` gives 2 x 172 threads on 172 cores, and HGB fits are 3-4x
slower than with `OMP_NUM_THREADS=86` (joblib#1869, FYI issue).

For experiments on HGB, LightGBM or XGBoost where timing isn't the point: few
threads per process, spinning on, and parallelize over processes.

## sklearnex and TBB

- `KMeans.fit` gets 3-20x slower after `KMeans.transform` in the same process
  whenever threads are limited below the machine size (`OMP_NUM_THREADS` or
  `n_jobs`; sklearnex 2026.1.0, reproduced on intel-xeon, intel-gamer-gpu and
  intel-laptop). `control_n_jobs` wraps fit, fit_transform, predict and score
  but not transform, whose `validate_data` runs oneDAL's finiteness check at
  the full thread count. The TBB workers then spin during later capped fits
  (gdb stacks). `sklearnex.utils.validation.assert_all_finite` alone triggers
  it. Flat again with `daal4py.daalinit(limit)` up front.
- sklbench calls `transform` between repeats (for the inertia), and only the
  Xeon had a thread cap. So Xeon sklearnex KMeans records taken with the cap
  understate its speed, by ~10x on some cases: look at the per-repeat
  `time_ms.fit`, not the mean.
- TBB's IPC coordination under joblib is never active (no `libirml`,
  scikit-learn-benchmarks#129).

## Forests (joblib threads, not OpenMP)

At high thread counts the costs are a GIL convoy at task boundaries, joblib's
per-task overhead and page faults or memset on fresh memory. Free-threaded
Python with a plain thread pool is 1.4-3x faster. See
`~/scikit-learn/tree_parallel_findings.md` and scikit-learn#35091 (lock
contention with free-threading).
