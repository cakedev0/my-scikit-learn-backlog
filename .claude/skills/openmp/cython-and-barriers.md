# Cython, barriers and parallel regions

Links and file paths are in [sources.md](sources.md).

## What a region costs

| | spinning | passive |
| --- | --- | --- |
| Empty region | 17-50 us on the laptops (1-4 us back to back in the HGB regions experiments) | ~90 us on the laptops, ~1 ms on the Xeon |
| Barrier inside a region | 1-4 us | ~10 us |
| Region after a serial gap of 30 us or more | same | often 400-600 us |

So in passive mode, a barrier inside a region is 10-50x cheaper than a new
region, and merging regions pays when it removes the serial code between them.
Python code between two regions counts as a serial gap.

## The barriers Cython adds to `prange`

OpenMP wants the `lastprivate` update after all the `firstprivate`
initializations of a variable that is in both clauses. Compilers do it with a
barrier. On `omp for nowait`:

| compiler | `firstprivate` | `lastprivate` | both | `private` |
| --- | --- | --- | --- | --- |
| GCC 7.5 to 15.3 | 0 | 0 | 1 (start) | 0 |
| clang 11 to 21 | 0 | 1 (end) | 2 (start, end) | 0 |
| MSVC (compile only) | 0 | 0 | 0 | 0 |

Cython puts both clauses on the loop target (since 2011) and, since 3.2, on
all the other variables too. So **every `prange` starts with a barrier**: all
the threads must be awake and present before any of them runs an iteration.
With a passive runtime, the slowest wake-up delays everybody.

The end of the loop:

- Cython < 3.3: a barrier at the end when code follows the loop in the
  `parallel` block.
- Cython 3.3: `nowait` on every `omp for`, so no end barrier. Two bugs:
  - cython#8120: code after a `prange` in a `nogil parallel` block can run
    before the other threads finish (GCC). Fixed by cython#8121, merged
    ("Fix A": `nowait` only with the GIL held, on a standalone `prange`, or on
    the last statement of the block).
  - cython#8129, open: a `prange` holding the GIL hangs with clang on GIL
    builds. clang's end barrier from `lastprivate` runs with the GIL held.
    Cython's test suite doesn't enable OpenMP on clang.
- scikit-learn pins `cython<3.2.6` (for another bug, cython#7846), so wheels
  have neither. An audit of the 31 loops in `parallel` blocks on main found no
  wrong result under 3.3.1, and the full test suite is unchanged. What follows
  those loops is a `free`, a reduction under a lock, or a second loop on the
  same static range.

Removing the start barrier ("Fix B", cakedev0/cython#4, open on the fork):
`private(x)` when flow analysis shows nothing can observe the difference, 398
of the 399 `omp for` of scikit-learn. HGB up to ~35% faster with a passive
runtime (`year_prediction_msd`, Xeon socket: 18.1 to 13.2 s), same results,
TSan clean. Not always a gain: 23% slower on the Core Ultra 9 with libomp
(deep idle states), and KMeans in passive mode is 15-21% slower than 3.3.1,
since 3.3.1 skips an end barrier that KMeans happens not to need.

Without touching Cython: manual chunks in a `with nogil, parallel():` block
(each thread computes its own range from `omp_get_thread_num()`) have no start
barrier and reach the same times as Fix B.

## Blocking points inside a region

Each of these is harmless alone and stalls a spinning team on hybrid CPUs (see
[wait-policy.md](wait-policy.md)). All were found in KMeans's Lloyd.

- **Memoryview slicing in `nogil`** (`X[start:end]`): it can raise, so Cython
  wraps the block in `PyGILState_Ensure` / `PyGILState_Release` in every
  thread. Grep the generated C for `PyGILState_Ensure` inside the outlined
  region. Pass indices or raw pointers instead. Slicing per item is worse: the
  atomic acquisition count on the shared memoryview made a sparse kernel
  15-35x slower.
- **A lock for the reduction** (`omp_set_lock`): give each thread its buffer
  and reduce after the region.
- **OpenBLAS's global mutex** on every call, even single-threaded (see
  [blas-and-pools.md](blas-and-pools.md)).
- **Large `calloc` per thread per iteration**: above 32 MiB (glibc's mmap
  threshold) each one is a fresh mmap, 0.7-1.4M page faults per iteration at
  86 threads. Allocate once per fit and zero what was written.

Per-thread buffers:

- Pad and align rows to a cache line (false sharing made sparse Lloyd 1.7x
  slower).
- Let each thread allocate or first-touch its own buffer. NumPy buffers
  allocated by the main thread land on one NUMA node (1.3x slower at 64
  threads).

Result of removing them (scikit-learn#35193): dense Lloyd iteration 570-610 to
215-220 us at 64 threads on the Xeon, sparse up to 25x faster at 86 threads.
With the OpenBLAS lock fix on top, spinning becomes a gain for small KMeans
fits (0.9 vs 2.8 s on the reproducer) instead of a stall.

## Thread counts and guards

- Keep the team size constant between consecutive regions on libgomp. A
  "partial team" should keep the full team and give work to fewer threads.
  But idle threads in a region aren't free either (per-thread setup).
- An extra small region costs ~100 us per call on a passive hybrid laptop:
  use `use_threads_if=` (or a serial branch) when there is a single block.
- Small-input guards and caps: see the threshold method in
  [benchmarking.md](benchmarking.md). HGB's cap table and heuristics are in
  scikit-learn#34935.
- Static schedules wait on the slowest core on hybrid CPUs. Bigger chunks hurt
  there (load imbalance) and help on the Xeon. A dynamic schedule reduced the
  loss with deep idle states but was slower otherwise.
- With `OMP_NUM_THREADS` set (what joblib does in workers), scikit-learn uses
  it as is, SMT threads included. Unset, it uses physical cores.

## One region per fit, with a handwritten barrier

The pattern that makes the wait policy irrelevant. Code excerpts are in
[snippets.md](snippets.md). Reference implementation:
`sklearn_emeans/_kernels.pyx` in sklearn-contrib-emeans
(`~/sklearn-contrib-emeans`), merged in PR #9:

- `_run_steps` opens one `with nogil, parallel(num_threads=n_threads):` for
  all the steps of an epoch. Its docstring explains why.
- `_steps_worker` is run by every thread and goes through the phases of all
  the steps.
- In each phase, threads claim pieces of work (chunks of rows, blocks of
  centers) with an atomic counter (`emeans_fetch_add`), and a second counter
  counts the pieces done.
- `emeans_wait(counter, target)` is the barrier: poll the counter, 100
  `pause` spins, then `sched_yield()` at each check. **A phase waits for its
  work, not for its threads**: a thread that has no CPU delays nothing unless
  it holds a piece, and then the waiters yield their CPU to it.
- `emeans_buffers_ready` is the start barrier: each thread allocates its
  buffers, then all wait, and nobody starts if one allocation failed.
- Serial parts (sort by label, Python callback with the GIL) go to the first
  thread that claims them. The main thread checks signals once per step.
- The atomics are written by hand in a verbatim C block, because MSVC has no
  C11 atomics in C mode: `Interlocked*` and `SwitchToThread` there,
  `__atomic_*` and `sched_yield` elsewhere. The code also builds without
  OpenMP.
- Results are bitwise identical whatever the number of threads: the work is
  split by blocks, not by threads.

Numbers: EMeans reproducer with libgomp spinning, median / max in s: 0.51 /
1.03 to 0.23 / 0.26 on the Core Ultra X7, 0.85 / 1.76 to 0.18 / 0.19 on the
Core Ultra 9, unchanged in passive mode.

Same idea for HGB (`_team.pyx`, cakedev0/scikit-learn#47, now in
sklearn-contrib-trees): `run_in_team(func, n_threads)` opens the region, the
main thread runs the Python boosting loop with the GIL and posts tasks, the
others claim chunks. Geomean speed-up over scikit-learn#34935 in passive mode:
2.69x on a Xeon socket (up to 13.7x), 1.67x on the EPYC, and 1.05-1.16x with
active wait. One region per tree is not enough. Each thread keeps its own
range of features first, then helps the others (free-for-all was ~15% slower
on wide data). A single thread-cap table then works for both wait policies.

Intermediate step, plain OpenMP: merging the regions of one split (7 regions
and 7 barriers down to 2 and 3, "fuse3") gives 0.76-0.90 geomean on the HGB
scalability cases, even with active wait.
