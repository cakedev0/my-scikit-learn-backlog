---
name: openmp
description: What Arthur has measured about OpenMP in scikit-learn and around it. Use when working on or benchmarking thread-parallel code (Cython prange/parallel, KMeans, HistGradientBoosting, EMeans, pairwise distances), when a timing is odd with many threads or on a laptop, or when the topic is libgomp vs libomp, active vs passive wait (GOMP_SPINCOUNT, OMP_WAIT_POLICY, KMP_BLOCKTIME), spinning and yielding, hybrid P/E-core CPUs, barriers, parallel-region overhead, OpenMP next to BLAS (OpenBLAS pthreads, MKL, threadpoolctl), the OpenMP runtime bundled in wheels, or which thread count, pinning and wait policy to use for a benchmark on intel-xeon or a laptop.
---

# OpenMP: what we know

Measured findings from 2026-07 to 2026-11, mostly on Linux x86. This file is
the map. Read the reference file of the topic before acting, and the linked
writeup or issue before quoting a number: figures here are orders of magnitude.
Versions and issue states move: check them when they matter.

| Topic | File |
| --- | --- |
| Runtimes, wait policy, spinning, yielding, hybrid CPUs, scheduler, idle states, wheels | [wait-policy.md](wait-policy.md) |
| Cython `prange` barriers, blocking points, one region per fit, the handwritten barrier | [cython-and-barriers.md](cython-and-barriers.md) |
| OpenMP next to OpenBLAS / MKL / threadpoolctl / joblib / TBB | [blas-and-pools.md](blas-and-pools.md) |
| How to benchmark threaded code, and the traps we fell into | [benchmarking.md](benchmarking.md) |
| Code: the spin-then-yield wait, one region per fit, the libgomp patch, reproducers | [snippets.md](snippets.md) |
| Full writeups behind the numbers (copies), patch and C reproducers | [writeups/](writeups/README.md) |
| Issues, PRs, original folders, where things live on the machines | [sources.md](sources.md) |

## The model in one page

**Two runtimes, two ways to wait.** After a parallel region, idle threads
either spin (active wait) or sleep (passive). Spinning makes the next region
start in microseconds. Sleeping costs a wake-up per region: ~0.1 ms on a
laptop, ~1 ms on the Xeon, more after a deep idle state.

- libgomp (GCC): `GOMP_SPINCOUNT`, default 300000 (3-5 ms), read once at
  load. Its spin loop only does `pause`, it never yields the CPU.
- libomp (LLVM, and Intel's libiomp5): `KMP_BLOCKTIME`, default 200 ms,
  changeable at run time. It calls `sched_yield` while spinning.
- On hybrid CPUs (P and E cores), recent versions of both default to passive
  with no variable set (libgomp since GCC 14). So intel-laptop,
  intel-gamer-gpu and mac-m2 are passive by default, intel-xeon and the EPYC
  VMs are active.
- The scikit-learn Linux wheel bundles a GCC 8.5 libgomp: it always spins,
  on hybrid CPUs too. conda-forge and dev builds have a recent one. Same code,
  opposite behavior on a laptop.

**Passive wait hurts code with many short regions.** HGB opens ~10k regions
per fit: 3-9x slower on laptops without spinning, 50x on the Xeon. So "set
`GOMP_SPINCOUNT=1`" is never a general fix.

**Spinning without yielding hurts when a spinning thread holds a CPU that a
working thread needs.** Four ways to get there:

1. Another pool wants the cores: OpenBLAS pthreads next to spinning libgomp
   threads (the original scikit-learn#17334, on servers).
2. A thread blocks inside the region (mutex, GIL, OpenBLAS's lock) while its
   teammates spin. On hybrid CPUs a region then costs ms instead of ~0.1 ms.
3. Two threads of the team share a core when the team covers all the cores
   (hybrid CPUs, or all SMT threads of a socket). The barrier at the end of
   each region waits for the preempted one: a time slice per region.
4. A larger team from an earlier call keeps spinning during a later, smaller
   one (fit after predict, one benchmark case after another).

libomp doesn't stall in 1 and 2 because it yields. A libgomp patch that
yields every 100 spins removes the stalls and keeps the gains of spinning
(prototype, not sent to GCC).

**What works in our code**

- No blocking inside a parallel region: no lock, no GIL (memoryview slicing
  in `nogil` takes it), no malloc of large buffers, BLAS calls only if needed.
- Fewer, longer regions. Best: one region per fit where threads claim work
  with atomic counters and wait for the work, not for the threads, with a
  short spin then `sched_yield`. Done in EMeans (`_run_steps`) and in the HGB
  experiment (`_team.pyx`). The wait policy then stops mattering.
- A constant team size: libgomp destroys and recreates threads when the team
  shrinks and grows.
- Every Cython `prange` starts with a barrier (all threads awake before any
  starts). Cython 3.3 changed the end barrier and has two bugs around it.

**Where the fix belongs.** Arthur's rule: don't pile workarounds in
scikit-learn for limits of the OpenMP runtime, the BLAS, the compiler or the
kernel. Take a fix there only if it is quick and simple. Otherwise find the
root cause, report it upstream with a standalone reproducer, and offer at most
one cheap mitigation, or advise an alternative (MKL instead of OpenBLAS,
libomp instead of libgomp). Don't iterate on workarounds or tune heuristics
against one library's behavior without asking. Said during the KMeans work
(2026-10-10): the bigger-chunks PR was dropped because it only helped
OpenBLAS, and a 15-20% sparse slowdown specific to GCC 13 was accepted.
Removing blocking points and merging regions count as improving our code, not
as workarounds.

## Before any experiment

1. Print what really runs: `threadpoolctl.threadpool_info()` for the
   libraries, `OMP_DISPLAY_ENV=VERBOSE` for the effective wait policy.
   `env` being empty says nothing. `lib/libgomp.so.1` can be a symlink to
   libomp (`readlink -f`).
2. Set the wait policy and the thread count explicitly on both sides
   (`GOMP_SPINCOUNT=300000` / `1`, `KMP_BLOCKTIME=200` / `0`), and say which
   in the report.
3. Default setup: intel-xeon, 22 threads pinned on one NUMA node. Add a
   hybrid machine when the change touches region counts, guards or waits:
   conclusions often flip there.
4. If the timing is odd, check in this order: half the threads,
   `OPENBLAS_NUM_THREADS=1`, the other wait policy, the other runtime. Then
   look at each pool separately before blaming one.
5. Follow the benchlock rules of `~/.claude/CLAUDE.md` and
   [benchmarking.md](benchmarking.md).

## Open questions (2026-11)

- Why P and E cores, and libomp and libgomp, react differently to deep idle
  states (HGB merged regions lose on the Core Ultra 9 with libomp).
- vcomp (Windows) wait behavior: never measured.
- Non-Intel hybrid CPUs, and the libgomp yield patch outside Intel.
- Whether to send the libgomp patch and the kernel scheduler report.
