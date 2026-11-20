# Benchmarking threaded code

The machine lock (`benchlock`) rules are in `~/.claude/CLAUDE.md`. This file
is what is specific to OpenMP. Links are in [sources.md](sources.md).

## Know what runs

- `threadpool_info()` once per script, printed in the output. Runtime mix-ups
  that happened:
  - `lib/libgomp.so.1` was a symlink to libomp (`_openmp_mutex=*_kmp_llvm`
    envs). `readlink -f` it. Strings containing `hybrid core type detected`
    mean libomp.
  - `LD_LIBRARY_PATH` set to load another BLAS also swapped the OpenMP
    runtime. Load the BLAS by full path, or swap one library with `LD_PRELOAD`.
  - In scikit-learn-benchmarks, calling
    `.pixi/envs/sklearn-dev-libomp/bin/python` directly loads the libgomp of
    `sklearn-dev` (shared editable build, RPATH). Go through `pixi run
    --frozen -e <env> python ...`. Only the threadpoolctl output showed it.
- `OMP_DISPLAY_ENV=VERBOSE` for the effective wait policy. The default depends
  on the libgomp version and on the CPU.
- Baseline for scikit-learn: nightly (or main) with threadpoolctl >= 3.8, not
  the last release. The wheel and a dev build differ in libgomp: don't compare
  a wheel to a source build without setting `GOMP_SPINCOUNT` on both.
- To mimic the wheel's libgomp with a dev build on a hybrid CPU:
  `GOMP_SPINCOUNT=300000`.

## Setup

- Default for benchmarks in general (set 2026-11-12): intel-xeon, 22 threads
  on the first 22 cores of a socket with their SMT siblings, which is one
  NUMA node. In slot `s0`: `benchlock --tag "..." s0 -- numactl
  --physcpubind=0-21,172-193 --membind=0 bash -lc "OMP_NUM_THREADS=22
  OPENBLAS_NUM_THREADS=22 ..."`. In `s1`: `--physcpubind=86-107,258-279
  --membind=4`. Describe it as "intel-xeon (Xeon 6787P), 22 threads on 22
  cores (one NUMA node, with SMT siblings)". Arthur finds it representative
  of laptops, desktops and medium cloud VMs, without the many-thread and
  hybrid problems that can't be fixed in scikit-learn. Brief subagents with
  it. Many-thread runs only when scaling is the topic.
- For PRs, Arthur wants benchmarks from scikit-learn-benchmarks, not asv.
- Thread scaling, guards, region counts, waits: also run a hybrid machine
  (intel-laptop or intel-gamer-gpu), in both wait policies. Add mac-m2 when
  libomp or ARM may differ.
- Report the wait policy with the numbers. For tables: passive is
  `GOMP_SPINCOUNT=1` / `KMP_BLOCKTIME=0`, active is `GOMP_SPINCOUNT=300000` /
  `KMP_BLOCKTIME=200`.
- Benchmark configs of scikit-learn-benchmarks keep the policy and thread
  counts they set (`bench.env`): they are part of what is measured. A config
  that sets no wait variable (`hgb_scalability.py`) runs passive on the hybrid
  machines and lands in the "(no active wait)" HGB tabs, and active on
  intel-xeon. `hgb_scalability_force_active_wait.py` forces it. When comparing
  across machines, go by the effective policy (`has_active_wait`), not by the
  config.
- Ad hoc experiments on HGB, LightGBM or XGBoost where timings aren't the
  point (ablations, sweeps, seeds): prefix with `GOMP_SPINCOUNT=300000`
  (`KMP_BLOCKTIME=200ms` on libomp), set `OMP_NUM_THREADS` to 1 or a few, and
  run several processes side by side. HGB scales poorly on small and medium
  data, so one many-threaded process wastes the cores.

## Method

- A/B sides interleaved in one `benchlock` call. On intel-laptop and mac-m2:
  ABBA order and 60-90 s idle before each process. A fixed order showed fake
  1.2-1.7x regressions (throttling of the second process).
- 5 or more interleaved rounds on hybrid CPUs, and look at min and median:
  big fits are bimodal (~1.4x between modes, same build).
- One team size per process. Cycling thread counts in one loop made some
  10-100x slower on the Xeon (team resizing).
- Time cold calls too (sleep ~50 ms before, so threads are asleep): one-off
  calls at the start of a fit pay the wake-up, hot loops overstate the gain.
- Thresholds for "is OpenMP worth it" guards: crossovers are ~3x later on
  intel-laptop than on the Xeon, and 2 threads may never win there. Pick the
  one that doesn't lose on the laptop.
- Separate calls that use different team sizes (fit and predict, one case and
  the next) into processes when spinning is on (scikit-learn-benchmarks#90).
- KMeans: fixed `init` array and `tol=0`, report ms per iteration measured
  directly. Different inits change iteration counts and empty-cluster
  relocations, and `(fit - init) / n_iter` mixes fixed costs in.
- The first fit after single-threaded data loading can be 1.4-3x slower at 86
  threads (CPU warm-up and placement of floating threads): discard it or warm
  up with all-core work.
- Check results are identical (bitwise when the reduction order is fixed) on
  every A/B.

## Machine traps

- intel-laptop: `sleep 3` after the ssh login before timing, and check `top`
  for bluetoothd (burst on each login, any user; restarting or disabling the
  bluetooth service needs sudo). For all-core runs prefer fewer threads than
  cores, or `OMP_PROC_BIND=close OMP_PLACES=cores`.
- intel-laptop power profile: it changes passive timings a lot. In
  `performance` (2026-10-13 to 16, then back to `balanced`), main without
  active wait ran 2.2-2.4x faster at 8-16 threads, the rest ~10% faster. The
  profile is not in the hardware hash. Since 2026-10-15 each telemetry sample
  has a `power` field (profile, governor, EPP, AC): read it before comparing
  dates. The balanced profile also throttles by ~40% after sustained all-core
  work (scikit-learn-benchmarks#145).
- intel-laptop kernel: Ubuntu 7.3.0-10 since 2026-11-19, installed by hand
  from `~/kernel-7.3/fixed` (no apt updates), with the scheduler regression
  of [wait-policy.md](wait-policy.md). All-core spinning results there are not
  comparable with earlier ones.
- Multi-socket runs vary from process to process
  (scikit-learn-benchmarks#80). Stay on one socket or one NUMA node.
- Laptop power states: waking BLAS threads before a timed section made
  identical code look 30% faster.
- Keep the machine name in result tags: the same tag on two machines merged
  silently once.

## Tools

- `strace` distorts timings (2 ms regions became 6 ms) and even its futex
  counts are biased. Use `LD_PRELOAD` counters instead (`mutex_count.c`,
  `sync_count.c` count locks, contended locks and `sched_yield`).
- `/proc/<pid>/task/<tid>/schedstat`: time each thread waited in a runqueue.
  This is what showed two threads sharing a core in EMeans.
- `perf record --switch-events` and bpftrace (`kfunc:` and tracepoints work
  under Secure Boot lockdown, debugfs doesn't) for scheduler questions.
- turbostat or the cpuidle counters in sysfs for C-state entries (no root
  needed for the latter). A timer thread per CPU keeps cores out of deep
  states without root.
- Timers around each phase before guessing which part is slow.
- TSan with Archer needs clang and libomp builds. It doesn't cover GCC builds.
- Standalone C reproducers (regions with nothing / mutex / `dgemm`,
  `spin_pair.c` for the scheduler) are what made upstream reports possible:
  reduce to one before blaming a runtime.
