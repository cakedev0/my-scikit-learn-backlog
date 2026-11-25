# Findings

Ordered roughly by importance. Each point says on which machine it was measured.

## 1. k-means++ init was most of `KMeans.fit` on big data

On the GNR (64 threads), the init took 75-97% of the fit time for the
scikit-learn-benchmarks datasets (e.g. sift 100k x 128, k=1000: 41.6 s of init,
then ~20 ms per Lloyd iteration). This is where most of the gap with sklearnex was.

Why it was slow:
- for float32, `_euclidean_distances` goes through the upcast path (chunked float64
  BLAS calls on skinny matrices): 365 ms per step on sift 1M with 64 threads vs 2.7 ms
  for the raw gemm. It gets slower with more threads: the init was up to ~2x slower
  with 8 threads than with 1.
- the rest of each step is single-threaded numpy passes (`np.minimum`, matvec, `cumsum`),
  which dominate for low-dimensional data (~10 ms per step on road_network, gemm 0.2 ms).

Fix (cakedev0/scikit-learn#23): one OpenMP kernel per step, one BLAS call per thread
on its block of samples, per-chunk potentials to sample the next candidates without a
global `cumsum`. Small distances (< 1% of `||x||² + ||c||²`) are recomputed exactly to
avoid float32 cancellation. Init: 20-100x faster with 64 threads on the GNR, 10-20x
with 8 threads, 1.5-2.7x with 1 thread.

Variants tried for the kernel:
- direct `sum (x - c)²` loop (no BLAS): exact and scales perfectly, but 2-4x slower
  single-threaded for high d (scalar code, wheels target baseline x86-64).
- one BLAS call per chunk of 256 samples: fast with few threads, but bad with many
  threads (see 3).
- one BLAS call per thread + exact recompute guard: kept.

## 2. Lloyd's parallel region had blocking points

Inside `with nogil, parallel():` of `lloyd_iter_chunked_dense/sparse`, every thread:
- `calloc`/`free` its buffers,
- adds them to the result under `omp_set_lock`,
- passes chunks as memoryview slices (`X[start: end]`): slicing can raise, so Cython
  wraps the whole block in `PyGILState_Ensure()`/`PyGILState_Release()` in every
  thread. Visible as one `gettid` syscall per worker per region (thread state creation).

Fix (cakedev0/scikit-learn#26): buffers allocated before the parallel loop (one row per
thread, padded by a cache line: without padding, false sharing made sparse 1.7x slower),
reduced after it, chunks passed as indices. The generated C has no `PyGILState_Ensure`
in the parallel regions anymore. On the GNR with 64 threads: dense Lloyd iteration
570-610 -> 215-220 us, sparse 720-880 -> 170-190 us, #17334 reproducer 10.1 -> 3.3 s.
On the laptops without active wait: neutral to slightly faster.

First version allocated the per-thread buffers with numpy in the main thread at every
iteration: 1.3x slower on road_network k=1000 at 64 threads (64 x 256 x 1000 doubles =
131 MB of fresh pages per iteration, first-touched on the wrong NUMA node). Reworked
version: each thread `calloc`s its buffers in the parallel block (as on main), stores the
pointer in a shared array, and the reduction happens after the region. Fixes it (1.01x)
and is faster where k * d is large (sift k=1000 0.66x, fashion_mnist 0.62x).

## 3. OpenBLAS takes a global mutex on every single-threaded call

Measured with `c/mutex_count.c` + `c/blas_region.c` (local laptop, 14 threads, each
calling a 256 x 50 x 8 `dgemm`):
- OpenBLAS (scipy-openblas wheel and a 0.3.33 build): exactly 2 `pthread_mutex_lock`
  per call, ~35% contended. With `USE_TLS=1` (0.3.33): 3 per call, still contended.
  Probably `blas_memory_alloc`/`blas_memory_free` (known: OpenMathLib/OpenBLAS#1782).
- MKL (sequential): no `pthread_mutex_lock` at all.
- None of the OpenBLAS builds we use have `USE_TLS` (scipy-openblas, conda-forge
  pthreads and openmp).

Consequences:
- many concurrent BLAS calls from OpenMP threads scale badly (Lloyd stops scaling at
  ~16 threads on the GNR with chunks of 256, worst with OpenBLAS pthreads; bigger chunks
  = fewer calls helped there, see 6),
- with more than 64-ish concurrent callers, OpenBLAS's buffer table is exhausted and it
  crashes (OpenMathLib/OpenBLAS#5958, why the benchmarks cap KMeans at 64 threads),
- with an actively waiting OpenMP runtime on hybrid CPUs, the contended mutex stalls the
  whole team (see 4).

## 4. Hybrid CPUs + libgomp active wait + blocking inside a parallel region = stall

On the hybrid laptops, with libgomp spinning (`GOMP_SPINCOUNT=300000`, the default of
the old libgomp bundled in the PyPI wheel), a parallel region in which threads block
(contended pthread mutex, GIL handoff, OpenBLAS's mutex) costs milliseconds instead of
~0.1 ms:
- standalone C, `pthread_mutex` once per thread: 3.8-7.6 ms/region (225H, 14 threads),
  1.9-5 ms (358H, 16 threads), ~0.1 ms with `GOMP_SPINCOUNT=1`.
- standalone C, single-threaded OpenBLAS `dgemm` per chunk: 2.1-5.4 ms/region vs
  0.07-0.18 ms with MKL or without spinning.
- The same regions without blocking are faster with spinning (~17 us vs ~90 us).
- Not reproduced on the GNR (non-hybrid): mutex region 34-42 us with 14 pinned threads.
- n_cores - 2 threads is not enough to avoid it on the 358H (14 of 16 cores: reproducer
  still 23-29 s).

What fixes it:
- removing the blocking points (no-sync for the Lloyd lock/GIL; a BLAS-free dense Lloyd
  experiment: reproducer 25.7 -> 1.5 s on the 358H with 14 threads),
- not spinning: `GOMP_SPINCOUNT=1`, or a recent libgomp: conda-forge's GCC 16 libgomp does
  not spin by default on hybrid CPUs (empty region ~95 us whatever the default),
- LLVM libomp, even with active wait (`KMP_BLOCKTIME=200ms` or `infinite`): the OpenBLAS
  region stays at 0.1-0.15 ms while the empty region shows it does spin (~50 us). Its
  default on hybrid CPUs is to not spin at all.
- Why libomp's spinning doesn't stall: it calls `sched_yield` while spinning (~85-125
  calls per region in the OpenBLAS test, `c/sync_count.c`), with the same lock contention
  as libgomp (~85k contended locks). libgomp's spin loop (`do_spin` in
  `libgomp/config/linux/wait.h`) only does `pause` and never yields.

## 4b. A libgomp patch that yields while spinning gets the best of both

`libgomp_spin_yield.patch` (GCC 13.3 libgomp): `sched_yield()` every
`GOMP_SPIN_YIELD` iterations of the spin loop (0 = current behavior). Local 225H,
14 threads, `GOMP_SPINCOUNT=300000`, sklearn `main` dev build:

| | GCC 13 libgomp, spinning | patched, yield every 100 | GCC 13, spin 1 |
|---|---|---|---|
| HGB fit | 354-367 ms | 325-393 ms | 897-937 ms |
| dense Lloyd 5000 x 50, k=8 | 5.5 ms | 0.16-0.17 ms | 0.12-0.15 ms |
| #17334 reproducer | 65 s | 2.5 s | 2.1-2.2 s |

Standalone regions: mutex 2.6-5 ms -> 45-48 us, OpenBLAS dgemm 3.9-4.7 ms -> 65-76 us,
empty 29-38 -> 32-50 us (spinning benefit kept). Yield every 1000 helps less, every 10000
not enough.

On the GNR (GCC 16.1 libgomp + same patch, E20): with 14 threads pinned to 14 cores, the
reproducer goes from 19.1 s to 2.45 s (this is the other mechanism: main's init spawns
OpenBLAS threads next to the spinning OpenMP threads, yielding lets them run). HGB,
Lloyd kernels and everything at 64 threads: unchanged. No measured cost anywhere.
So yielding while spinning fixes both #17334 mechanisms in the OpenMP runtime itself,
without losing the spinning gains.

intel-laptop (358H, 16 threads, E27/E28): every period from 10 to 1000 removes the stall
(mutex region 1.8-3.7 ms -> 30-36 us, reproducer 51 s -> 2.2-2.7 s), HGB 192-250 ms vs
1.87 s with spin 1. With yield, the spin count can go down to 3000 without losing HGB's gain
there.

A smaller spin count is not an alternative (225H, E29): without yield, the stall only goes
away at spin counts around 300, where HGB is already 1.45x slower than spinning + yield
(527 vs 364 ms), and at 3000-30000 both the stall (reproducer 3.9-15 s) and HGB (750-1600 ms)
are bad. On large regions (E, BLAS limited), spinning + yield is as fast as spin 1.

Context in GCC: since GCC 14 (PR109812, from Intel), libgomp's default spin count is 1 on
CPUs with the CPUID hybrid bit (`config/linux/x86/spincount.h`), and only when neither
`OMP_WAIT_POLICY` nor `GOMP_SPINCOUNT` is set. That avoids the stall by giving up spinning
on hybrid CPUs, which costs HGB 2.7-8x. Trunk's `do_spin` still never yields. libomp yields
after 512 spin iterations, then every 256 (`KMP_INIT_WAIT`/`KMP_NEXT_WAIT` in `kmp.h`).

GNR period sweep (E30, 14 to 172 threads): yield 100 is neutral to positive everywhere, the
only cost is on empty regions with all SMT threads busy (+10-25% at 172 threads, HGB
unchanged). Short spin counts without yield are bad there too. 100 is the default to propose
(10 and 1000 are not better anywhere).

Trunk version of the patch: `libgomp_spin_yield_trunk.patch` (default 100, documented in
`libgomp.texi`, shown by `OMP_DISPLAY_ENV=verbose`). Applies to trunk `bd261b39de`, builds
and works (checked on GCC 13.3 with the display hunk ported). Still untested: non-Intel CPUs
(no machine available).

## 5. Active wait is essential on the server

On the GNR, `GOMP_SPINCOUNT=1` makes every parallel region cost ~0.8-1.2 ms (wake-up):
HGB fit (20k x 20, 100 iterations, 14 threads) 0.4 s -> 18 s. On the laptops too, active
wait makes HGB 3-9x faster. So "set `GOMP_SPINCOUNT=1`" is not a general fix.

## 6. Chunk size of dense Lloyd: depends on the machine

- GNR (pthreads OpenBLAS, 64 threads): chunks of 1024-4096 samples instead of 256 give
  3-5x faster iterations (fewer contended BLAS calls), as long as the per-thread
  `(chunk, n_clusters)` buffer stays in cache (`2**18 // n_clusters`) and there are
  enough chunks for all threads (`ceil(n / n_threads)`). Spreading tiny workloads over all
  64 threads brings the contention back.
- Hybrid laptops: bigger chunks hurt medium/large fits (100000 x 50, k=10: 3.9 -> 4.4-4.9
  ms/it), probably load imbalance (static schedule, E/LPE cores slower). Requiring at least
  4 chunks per thread didn't fix it.
- Compromise kept (rec2): grow chunks only with many threads, `256 * max(1, n_threads // 16)`
  (unchanged up to 31 threads, 1024 at 64), still capped. No regression anywhere measured,
  2-4x on big Lloyd at 64 threads (a bit less than 4096-sample chunks on sift k=10:
  0.24x vs 0.13x).

## 7. Plain loops are much slower than OpenBLAS kernels

The wheels compile for baseline x86-64 (SSE2), OpenBLAS dispatches AVX2/AVX-512 at
runtime. A plain loop for `X @ C.T` is 1.0-1.4x slower only for d=2, 2-8x slower for
d >= 10 (local laptop, single thread). Using it for "small" problems (work per thread
< 2**18) fixes the active-wait stall but costs ~2.5x on small Lloyd iterations without
active wait on the 225H (250 -> 650 us for 5000 x 50, k=8, 14 threads), ~10x more than
estimated from single-thread numbers. Not kept.

## 8. Thread-count and team-size effects

- `_inertia_dense` right after a Lloyd iteration that used a different number of threads
  (team size 20 vs 64 on the GNR): 13 ms instead of ~0.1 ms. Changing the OpenMP team size
  between consecutive regions is expensive with libgomp at high thread counts.
- A constant team size with idle threads is worse in Lloyd (per-thread setup costs).
- The k-means++ "small data guard" (one thread per 2**16 elements of X) makes the init
  faster on small data alone, but at 64 threads on the GNR the combination with no-sync
  got slower (4.4 s vs 2.5-3.0 s), probably team-size changes between init and Lloyd.

## 8b. Memoryview slicing per (sample, candidate) killed the sparse k-means++ kernel

#23's first sparse kernel sliced `X_data`/`X_indices` once per (sample, candidate) inside a
prange (atomic acquisition count on shared memoryviews): 15-35x slower than main on
intel-laptop, 16 threads (sparse 100k x 1000 d=0.01 f64, k=50: main 198 ms, PR 3360 ms).
Indexing with `X_indptr` and no slicing: 41 ms. Same pattern as Lloyd's blocking points
(finding 2). Benchmark sparse paths even when they "just reuse" an existing helper.

## 9. Misc

- Sparse Lloyd on the no-sync branch was ~15-20% slower single-threaded with GCC 13 builds:
  GCC 13 spills the centers' data pointer and reloads it from the stack at each non-zero
  (`movq 16(%rsp), %rcx` in the f64 inner loop). Tried (2026-10-16, dropped, not in #26):
  computing the dot products with 2 centers per loop over the
  non-zeros (last center twice if odd): 2 independent add chains, shared index/data loads,
  bitwise identical results. 1 thread vs main: 0.69-0.98x intel-laptop (GCC 13, real build),
  ~0.54-0.91x GNR (GCC 14/16) and 0.60-0.92x mac-m2 (clang 21) in the standalone harness.
  4 centers per loop is better on GNR/M2 (down to 0.41x) but GCC 13 spills 2 row pointers in
  f64. Row pointers instead of memoryview indexing: one case 1.06x slower on GCC 13.
  Transposed centers (`K` contiguous values per feature, `dots[j] +=` loop): 0.31-0.59x at
  k=100 but ~1.0x at k<=8, a possible follow-up. Harness: `~/sparse-1t/` on all 3 machines.
  But on the GNR real builds (GCC 16, s1, sparse workloads of benchmarks#144: 20newsgroups
  k=20/50, RCV1 100k k=20 / 50k k=100, `~/sparse-1t/gnr1/`), the 2-centers commit only gains
  at 1 thread (0.82-1.00x of PR head, PR head = main at 1 thread) and is neutral or worse at
  16/32/86 threads (f32 RCV1 100k k=20 at 86 threads: 7.57 -> 9.02 ms/iter). PR head vs main:
  0.15-0.62x at 86 threads, 0.24-0.76x at 32, 0.43-1.02x at 16. Multi-threaded sparse Lloyd is
  dominated by the per-iteration calloc of the k x n_features buffer of each thread (52 MB per
  thread for 20newsgroups k=50 f64), time grows from 16 to 32 threads: the real follow-up.
- On the 358H, `main`'s Lloyd iterations are ~30% faster after `main`'s init than after
  PR 1's init (same iterations). With `OPENBLAS_NUM_THREADS=1` the difference mostly
  disappears: the OpenBLAS threads woken by `main`'s init seem to keep the CPU in a faster
  state. Power management effects make laptop benchmarks of small fits fragile.
- Results of the PyPI 1.9.1 wheel on the #17334 reproducer: 112 s on the 225H (14 threads),
  35/67 s on the 358H (14/16 threads), 15-20 s on the GNR (14 pinned / 64 threads).

## Lloyd thread buffers reused across iterations (2026-10-17, branch `kmeans/lloyd_reuse_buffers`)

Prototype on top of #26, uncommitted in the main checkout (`./`) (intel-xeon: `~/kmeans-spin/p26reuse3`,
results `~/sparse-1t/gnr{2,3,4}/`). `_kmeans_single_lloyd` allocates one `np.zeros((n_threads,
row))` per fit (rows padded and aligned to cache lines), `_reduce_thread_buffers` zeroes what it
read. Why: with #26, each thread `calloc`s k x n_features per iteration; buffers > 32 MiB (glibc
max mmap threshold) are fresh mmaps every iteration: 0.7-1.4M page faults/iteration at 86 threads,
97% of the iteration (f64 20news k=50: 8 ms without center update, 332 ms with). f32 buffers
<= 32 MiB were already recycled by malloc, hence no gain there.
- Zeroing variants: per-value branch (only non-zeros written) is up to 1.22x slower than #26 on
  f32; `memset` of whole 2048-value blocks loses ~30% on f64 big buffers (write traffic). Kept:
  branch-free sum + `memset` per cache line holding a non-zero (rows aligned to 64 B).
- Results vs #26 head (s1, GCC 16): f64 20news k=50 and RCV1 50k k=100: 0.08x at 86 threads,
  0.12-0.16x at 32, 0.20-0.33x at 16; nothing slower than 1.02x; bitwise identical outputs.
- Peak RSS: +17-23% on f64 RCV1 50k k=100 (pages touched by any iteration stay resident), same
  elsewhere; bounded by n_threads x k x n_features, which #26 can also reach in one iteration.
- Dense (benchmarks#144 workloads: sift d=16-128 k=10-3000, nytimes k=1000, fashion_mnist k=100,
  road_network k=1000; 1/16/32/64 threads, `~/sparse-1t/dense1/`): 0.87-1.05x of #26, bitwise
  identical. Dense buffers are small, nothing to gain or lose.
- Test `test_lloyd_iter_reused_thread_buffers` (buffers zeroed after use, reuse == fresh):
  fails on all 6 params if the reduction doesn't zero the lines.

## #26 final (938b98f86c) vs main, 3 machines (2026-10-19)

Committed and pushed to #26. Lloyd ms/iteration, ratio PR/main, `~/bench26/` on laptop and M2,
`~/sparse-1t/{gnr5,dense2}` on intel-xeon:
- intel-xeon (one socket): dense 0.97-1.02 (1 thread), 0.90-1.01 (16), 0.32-0.96 (64); sparse
  0.80-1.01 (1), 0.15-0.88 (16), 0.06-0.53 (32), 0.04-0.18 (86).
- intel-laptop: dense 1.00-1.01 (1), 0.94-1.09 (16); sparse 0.92-1.12 (1, GCC 13), 0.23-1.06 (16;
  RCV1 100k k=20 f64 1.06 is consistent across runs).
- mac-m2: dense 0.80-1.03 (1), 0.94-1.11 (8); sparse 1.00-1.05 (1), 0.84-1.02 (8).
- Main-first fixed order on laptop/M2 first showed fake 1.2-1.7x regressions (throttling of the
  second process); ABBA + 60-90 s idle fixed it.

## #26 on the #17334 reproducer, intel-laptop 16 threads (2026-10-19, `~/bench26/repro*.out`)

Order PR-variant/PR/main/main/PR/PR-variant, 45-60 s idle before each run, `repro_iters.py`:
- Default env (dev builds, Ubuntu GCC 13 libgomp): main 2.7-2.8 s (Lloyd 253-261 us/it), #26
  `938b98f86c` 3.05-3.09 s (285-288 us/it), #26 + `use_threads_if=n_blocks > 1` on the reduction
  prange 1.96-1.98 s (179-181 us/it). The reduction's extra parallel region costs ~100 us per
  iteration on this hybrid laptop when there is a single block (one Lloyd iteration 5000 x 50 k=8:
  update_centers True - False = ~110 us for #26, ~30 us for main, ~45-60 us with the serial variant).
- `GOMP_SPINCOUNT=300000` (mimics old libgomp): main 48.6-50.9 s, #26 99.2-99.6 s, serial
  variant 60.1-62.0 s. #26 doesn't fix the stall and makes it worse (2 regions/iteration).
- PyPI 1.9.1 wheel (bundled GCC 8.5 libgomp), default env: 64.2 s. The 51 s "main" in older
  notes was a dev build with explicit GOMP_SPINCOUNT=300000; dev builds don't stall by default.
- Committed as 01c5612a9e on #26 (2026-10-19). The benchmark workloads of the description all have more than one block (k * d > 2048), so their numbers don't change.

## scikit-learn-benchmarks#154: #26 (01c5612a9e) vs upstream main 4e624cfb16 (checked 2026-10-21)

Raw records fetched with `scripts/fetch_pr_results.py --pr 154`; analysis script in this
session's scratchpad (`analyze154.py`). Lloyd ms/iteration ratio PR/main, median of 5 runs:
xeon OpenBLAS (64 thr) 0.04-0.76 (median 0.27), xeon MKL (86 thr) 0.05-0.82 (0.18), laptop
OpenBLAS 0.24-1.05 (1.01), laptop MKL 0.25-1.08 (1.01). Main sits at an ~18-25 ms/iteration
floor on the xeon for most dense cases (sklbench lets threads float on 172 logical CPUs).
float32 fits can differ by a few iterations (fixed reduction order), float64 identical.

First-run effect: only xeon + MKL env (libomp, 86 threads) + PR side: run 0 is 1.43x (median,
up to 3.1x) slower than runs 1-4. Reproduced with `first_fit_iters.py` (intel-xeon
`~/sparse-1t/`, `venv-mkl-final` = 01c5612a9e): per-iteration time of the first fit starts at
~17 ms and decays to ~6 ms over ~1.5 s of all-core work, steady state 2.76 ms. 3 s of all-core
busy work before the fit (even from separate processes) removes most of it: a CPU warm-up
effect after the single-threaded data loading, plus thread placement of floating libomp
threads (pinned with OMP_PLACES=cores OMP_PROC_BIND=spread: 4.8 ms for the first 10 iterations
then 1.76 ms, i.e. 1.6x faster steady state than floating). Main stays at ~11-13 ms (sync
bound) whether pinned or not, which hides the warm-up. Not a #26 code issue.
- Opened upstream as scikit-learn/scikit-learn#35193 (2026-10-21), changelog fragment 35193.efficiency.rst in commit e69792bf6d.
