# Runtimes, wait policy, spinning and hybrid CPUs

Links and file paths for each claim are in [sources.md](sources.md).

## The runtimes

| | libgomp (GCC) | libomp (LLVM), libiomp5 (Intel) |
| --- | --- | --- |
| Knob | `GOMP_SPINCOUNT` (iterations) | `KMP_BLOCKTIME` (ms, or `infinite`) |
| Default | 300000, 3-5 ms on a P core, longer on E cores | 200 ms |
| Default on hybrid CPUs | 1 since GCC 14 (PR109812, CPUID hybrid bit, `config/linux/x86/spincount.h`), only if neither `OMP_WAIT_POLICY` nor `GOMP_SPINCOUNT` is set | 0 (`KMP_DEFAULT_BLOCKTIME` in `kmp.h`) |
| Change at run time | no, read once at load | `kmp_set_blocktime` |
| While spinning | `pause` only (`do_spin` in `config/linux/wait.h`) | `sched_yield` after 512 spins, then every 256 (`KMP_INIT_WAIT`, `KMP_NEXT_WAIT`) |
| Team shrinks then grows | destroys the extra threads and creates them again: ~290 us per regrow with 23/4 threads, 13 ms seen at 64 vs 20 threads on the Xeon | keeps its threads |

`OMP_WAIT_POLICY=active|passive` works for both and overrides the hybrid
default. Windows wheels use vcomp: nothing measured.

Where each one is found:

- scikit-learn PyPI wheel, Linux: libgomp of GCC 8.5 (AlmaLinux 8 base of
  manylinux_2_28, picked by `auditwheel` instead of the GCC 14 toolset),
  scikit-learn#34437. It predates the hybrid default, so it spins everywhere.
  sklearn-contrib-emeans wheels vendor the same one.
- scikit-learn PyPI wheel, macOS: libomp 11 from 2021, pinned in
  `build_tools/wheels/build_wheels.sh`. Don't update it lightly, see "Two
  libomp on macOS" below.
- conda-forge: recent libgomp by default. With `_openmp_mutex=*_kmp_llvm`
  (MKL envs, `sklearn-cf-libomp-openblas`), `lib/libgomp.so.1` is a symlink to
  `libomp.so`, and its strings still say "GCC".
- Dev builds with the system compiler: the system libgomp. On the Ubuntu
  laptops it is GCC 13's, from before the hybrid default: it spins unless told
  otherwise.

Effective default per machine, nothing set, conda-forge runtimes (checked
2026-11-04): intel-laptop, intel-gamer-gpu, mac-m2 passive. intel-xeon and
aws-l4-1 (EPYC) active. Recorded software envs show `[host]
GOMP_SPINCOUNT=1` on the hybrid machines: that is the runtime's report, not an
environment variable.

## What passive wait costs

- Xeon: every region costs ~0.8-1.2 ms. HGB fit 20k x 20, 100 iterations, 14
  threads: 0.4 s to 18 s.
- Laptops: HGB 3-9x slower, LightGBM and XGBoost similar (2-10x).
- The cost depends on how deep the workers slept (Core Ultra 9 285): waking
  22 threads takes ~12 us from C1, ~200 us from C2/C3, and the woken threads
  then run 2-3x slower for a while. A serial phase of 30 us or more before a
  region is enough for the `menu` governor to go deep: the region after it
  often costs 400-600 us.
- So a change that makes a phase serial (a guard, a cap) can slow down the
  next parallel phase: HGB histograms went from 2.6 to 13.6 s on
  `year_prediction_msd` after two guards turned split finding serial.
- Merging regions or removing a barrier makes threads sleep less often but
  deeper. On the Core Ultra 9 with libomp it lost 23% on one case, and the
  loss disappears with only C1 allowed or with a timer waking each CPU. The
  laptop and the Xeon gain on the same case. Not understood.
- Disabling only C3 is not a valid test: the governor then picks C2 for short
  idles and everything slows down.

## What spinning costs

Spinning helps when regions are short and close together. It hurts when a
spinning thread holds a CPU that a working thread needs, because libgomp never
gives it up before the spin count ends or the scheduler preempts it (a slice
of a few ms).

1. **Another thread pool.** k-means++ on main before scikit-learn#35140
   called multithreaded OpenBLAS (pthreads) between OpenMP regions. Xeon, 14
   threads pinned on 14 cores: 19 s, against 2.0 s with 12 threads on the same
   cores, 2.1 s with an init without multithreaded BLAS.
2. **Blocking inside a region, on hybrid CPUs.** Standalone C, one
   `pthread_mutex` per thread per region: 2-8 ms per region with libgomp
   spinning, ~0.1 ms with `GOMP_SPINCOUNT=1`, 34-42 us on the Xeon. Same with
   a single-threaded OpenBLAS `dgemm` per chunk (its global lock), the GIL, or
   `omp_set_lock`. The #17334 reproducer with the PyPI wheel: 112 s on the
   Core Ultra 5 (14 threads), 1.2 s with 8 threads. Using 2 threads fewer than
   cores is not enough.
3. **Two team threads on one core.** With one spinning thread per CPU, any
   extra task, or a scheduler decision, puts two of them on one CPU. The one
   at the barrier spins while the other can't reach it.
   - EMeans 0.2.0 on hybrid CPUs: half of the fits 1.5-5.5x slower
     (sklearn-contrib-emeans#8). ~450 regions per fit, each thread waiting
     60-200 ms in a runqueue. First guess (serial part stuck on an E core) was
     wrong: timers per phase showed the time was lost inside the regions.
   - Xeon socket, 172 threads (all SMT threads): after a k-means++ init capped
     to few threads, the others sleep, are woken on shared CPUs and spin. Lloyd
     17.8 vs 1.9 ms per iteration. Fine with 86 threads, with
     `OMP_WAIT_POLICY=active` (they never sleep, so never move), or with the
     yield patch.
   - intel-laptop, 16 threads on 16 cores: stalls when serial gaps last ~4-10
     ms (P-core threads already asleep, E-core threads still spinning). The
     kernel moves two threads onto one idle P core by active balance (one from
     an E core, one from an LP-E core), ~30 ms per episode. No stall on core
     subsets (P+E, P+LP-E, E+LP-E) on kernel 7.0. This is the HGB dashboard
     outlier of intel-laptop (bin transform at 16 threads: 79 ms vs 1.2 ms at
     8 threads).
   - Kernel 7.2+ makes it ~100x worse on hybrid x86: `select_idle_capacity()`
     returns no CPU although some are idle (SIS_UTIL scan limit, commit
     61ea17a63719). ~20% of regions stall with 16 threads and 6 ms gaps, and
     12 threads on P+E stall too. `NO_SIS_UTIL` removes it. Report drafted,
     not sent.
   - intel-laptop: each ssh login (any user) makes bluetoothd burn a CPU for
     ~1 s, starting ~0.5 s after the login. That one extra task is enough:
     calls of 100 tiny regions go from ~0.4 ms to ~600 ms.
4. **A leftover team.** HGB `predict` uses all the threads while `fit` is
   capped. After the first predict, the big team keeps spinning and the next
   fits are up to ~1.8x slower (scikit-learn-benchmarks#90).

Binding is not a general fix. `OMP_PROC_BIND=close OMP_PLACES=cores` removed
the EMeans outliers but made every fit as slow as the worst one (1.14 s vs
0.27 s median), with work pinned on slow cores. It does help all-core laptop
runs of other workloads and libomp on the Xeon (see
[blas-and-pools.md](blas-and-pools.md)).

## Yielding while spinning

`sched_yield()` every N spin iterations lets the working thread run and keeps
the short wake-up. This is what libomp does, and why libomp with
`KMP_BLOCKTIME=200` doesn't stall in cases 1 and 2.

Prototype libgomp patch (`GOMP_SPIN_YIELD`, default 100 in the trunk version):

| Core Ultra 5, 14 threads, spinning | libgomp | patched | spin count 1 |
| --- | --- | --- | --- |
| HGB fit | 354-367 ms | 325-393 ms | 897-937 ms |
| dense Lloyd iteration, 5000 x 50 | 5.5 ms | 0.16 ms | 0.12-0.15 ms |
| #17334 reproducer | 65 s | 2.5 s | 2.1 s |

- Xeon: 19.1 to 2.45 s on the pinned reproducer, HGB and 64-thread cases
  unchanged. Only cost seen: +10-25% on empty regions with all SMT threads busy.
- Periods from 10 to 1000 work on the laptops, 100 is the one to propose.
- A smaller spin count without yield is not an alternative: the stall only
  goes at ~300, where HGB is already 1.45x slower, and 3000-30000 is bad for
  both.
- Status: not sent (gcc-patches or Bugzilla first, undecided). Untested on
  non-Intel CPUs. GCC's own answer so far was to stop spinning on hybrid CPUs,
  which costs HGB 2.7-8x.

In our own waits (see [cython-and-barriers.md](cython-and-barriers.md)): spin
~100 `pause`, then `sched_yield` at each check.

## Two libomp on macOS

A recent libomp (conda-forge llvm-openmp 23.1.5) next to PyTorch's PyPI wheel
segfaults or deadlocks, depending on the import order. Both export the same 22
weak C++ template symbols (`__kmp_suspend_64<...>`, ...), dyld coalesces them,
and each runtime calls into the other's state. scikit-learn's libomp 11 shares
only 4 harmless ones, so the old pin protects it. Reproduced without EMeans by
swapping scikit-learn's libomp. See sklearn-contrib-emeans#15 and
pypackaging-native#50.
