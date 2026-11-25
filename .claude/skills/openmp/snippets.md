# Snippets

Enough code to rebuild each idea without the original repository. Excerpts are
trimmed. Originals are listed in [sources.md](sources.md).

## Check what runs

```bash
# effective wait policy and spin count, as the runtime sees them
OMP_DISPLAY_ENV=VERBOSE python -c "import sklearn.cluster" 2>&1 | grep -E "WAIT_POLICY|SPINCOUNT|BLOCKTIME"
# which libraries, and whether "libgomp" really is libgomp
python -c "from threadpoolctl import threadpool_info as t; [print(i['prefix'], i['filepath']) for i in t()]"
readlink -f "$LIBGOMP"; strings "$LIBGOMP" | grep -E "hybrid core type detected|running gcc"
```

`hybrid core type detected` means libomp. `running gcc 8.5.0` is the wheel's
old libgomp.

Wait policy for an A/B: passive is `GOMP_SPINCOUNT=1` or `KMP_BLOCKTIME=0`,
active is `GOMP_SPINCOUNT=300000` or `KMP_BLOCKTIME=200`. Swap a patched
libgomp with `LD_PRELOAD=/path/libgomp.so.1.0.0`, not `LD_LIBRARY_PATH`.

## A wait that spins, then yields (EMeans)

Verbatim C block at the top of `sklearn_emeans/_kernels.pyx`. The MSVC branch
exists because MSVC has no C11 atomics in C mode.

```c
#if defined(_MSC_VER)
#include <windows.h>
#define emeans_fetch_add(p, v) ((Py_ssize_t) InterlockedExchangeAdd64((volatile LONG64*) (p), (v)))
#define emeans_load(p) ((Py_ssize_t) ReadAcquire64((const volatile LONG64*) (p)))
#define emeans_relax() YieldProcessor()
#define emeans_yield() SwitchToThread()
#else
#include <sched.h>
#define emeans_fetch_add(p, v) __atomic_fetch_add((p), (v), __ATOMIC_SEQ_CST)
#define emeans_load(p) __atomic_load_n((p), __ATOMIC_SEQ_CST)
#define emeans_relax() __builtin_ia32_pause()   /* x86 only, no-op elsewhere */
#define emeans_yield() sched_yield()
#endif

/* Wait until *counter >= target. After a short spin, yield the CPU at
   each check: when two threads of the team share a core, the one that
   waits must let the one that works run. */
static void emeans_wait(Py_ssize_t *counter, Py_ssize_t target) {
    int n_spins = 0;
    while (emeans_load(counter) < target) {
        if (n_spins < 100) { n_spins++; emeans_relax(); }
        else emeans_yield();
    }
}

/* Start barrier: each thread calls it once its buffers are allocated.
   Nobody starts to work if one allocation failed. */
static int emeans_buffers_ready(Py_ssize_t *n_ready, Py_ssize_t *n_failed, int failed) {
    if (failed) emeans_fetch_add(n_failed, 1);
    emeans_fetch_add(n_ready, 1);
    emeans_wait(n_ready, emeans_omp_num_threads());
    return emeans_load(n_failed) == 0;
}
```

## One region, phases that wait for their work (EMeans)

```cython
# _run_steps: one region for all the steps of an epoch
with nogil, parallel(num_threads=n_threads):
    distances = <floating*> malloc(...)          # thread-local buffers
    sums = <double*> malloc(...)
    if emeans_buffers_ready(&n_ready, &n_failed, distances == NULL or sums == NULL):
        _steps_worker(...)                       # every thread runs it
    free(distances)
    free(sums)
```

```cython
# _steps_worker, noexcept nogil. counters is shared, zeroed, (n_steps, N_COUNTERS):
# for each phase, one counter of work claimed and one of work done.
for s in range(n_steps):
    if emeans_load(failed):
        return
    if is_main_thread:                # only it can handle Ctrl-C
        with gil:
            try:
                PyErr_CheckSignals()
            except BaseException as exc:
                (<list> errors).append(exc)
                emeans_fetch_add(failed, 1)
                return
    counter = counters + s * N_COUNTERS

    # A parallel phase: claim blocks until none is left, then wait for the work.
    while True:
        first = emeans_fetch_add(&counter[NORMS], block_size)
        if first >= n_clusters:
            break
        last = min(first + block_size, n_clusters)
        for j in range(first, last):
            centers_squared_norms[j] = _squared_norm(centers + j * n_features, n_features)
        emeans_fetch_add(&counter[NORMS + 1], last - first)
    emeans_wait(&counter[NORMS + 1], n_clusters)

    # A serial phase: the first thread to get here does it, the others wait.
    if emeans_fetch_add(&counter[SORT], 1) == 0:
        _sort_by_label(...)
        emeans_fetch_add(&counter[SORT + 1], 1)
    emeans_wait(&counter[SORT + 1], 1)
```

Why it holds: a late thread finds nothing to claim and nothing to wait for in
the steps the others completed. A thread without a CPU delays nobody unless it
holds a block, and then the waiters yield to it. The result doesn't depend on
the number of threads because the work is split by blocks. Block size:
`max(1, n_clusters // (16 * n_threads))`, small enough to balance uneven work.

## A team driven by Python code (HGB `_team.pyx`)

The main thread runs the Python fit loop with the GIL, inside the region. The
kernels post tasks and the other threads take chunks.

```cython
def run_in_team(func, int n_threads):
    if n_threads <= 1 or not hgb_team_acquire(n_spins, is_static):
        return func()                      # a team already runs in the process
    try:
        with nogil, parallel(num_threads=n_threads):
            if hgb_team_thread_num() == 0:
                hgb_team_set_active(1)
                _call(<PyObject*> func, <PyObject*> out)   # `with gil:` inside
                hgb_team_set_active(0)
                hgb_team_stop()
            else:
                hgb_team_worker()          # loop: wait for a task, claim chunks
    finally:
        hgb_team_release()

# in a kernel:
if team_active():
    team_run(fn, ctx, n_items, chunk_size)   # returns when n_done == n_items
else:
    for i in prange(n_items, ...): ...
```

Details that mattered:

- A task is `(fn, ctx, n_items, chunk_size)` plus a generation number. The
  claim word packs `generation << 32 | next_item`, updated by compare and
  swap, so a late worker can't take items of the next task. Fields are read
  with atomic loads and checked against the generation afterwards (needed on
  ARM).
- Static mode: one cache-line-aligned claim word per thread for its own range
  `[t * n / T, (t + 1) * n / T)`. A thread done with its range goes through
  the others' ranges. It keeps a thread on the same features from one task to
  the next.
- Waiting: `n_spins` pauses (100 by default), then `sched_yield` at each
  check, for the workers between tasks and for the main thread until `n_done`
  reaches `n_items`.
- GCC and clang atomics and POSIX only: not portable to MSVC as is.

## Manual chunks: no start barrier, plain OpenMP

Sketch of the `nb` variant of the HGB regions experiments. Nothing is
`firstprivate` and `lastprivate` at once, so the compiler adds no barrier and
each thread starts as soon as it is awake.

```cython
from cython.parallel cimport parallel
cimport openmp

with nogil, parallel(num_threads=n_threads):
    t = openmp.omp_get_thread_num()        # assigned in the block: private
    n_team = openmp.omp_get_num_threads()
    start = t * n // n_team
    stop = (t + 1) * n // n_team
    for i in range(start, stop):           # a plain range, not a prange
        out[i] = f(i)
    # code here runs per thread, before the others are done: only touch
    # thread-local data, or wait explicitly
```

## Per-thread buffers without a lock or the GIL

Sketch of what scikit-learn#35193 does in Lloyd, not its code
(`aligned_calloc` and `_update_chunk` are placeholders).

```cython
# before the region: one pointer per thread, reduction after the region
buffers = <double**> calloc(n_threads, sizeof(double*))
with nogil, parallel(num_threads=n_threads):
    t = openmp.omp_get_thread_num()
    # each thread allocates and first-touches its own buffer (NUMA),
    # rows padded and aligned to 64 bytes (false sharing)
    buffers[t] = <double*> aligned_calloc(row_size_padded)
    for chunk in prange(n_chunks, schedule="static"):
        # pass indices or raw pointers: X[start:end] on a memoryview takes
        # the GIL in every thread (grep PyGILState_Ensure in the generated C)
        _update_chunk(&X[0, 0], chunk * chunk_size, ..., buffers[t])
# serial or parallel reduction of buffers[0..n_threads) here, no omp_set_lock
```

For buffers above 32 MiB, allocate once per fit and zero only what was
written (per cache line holding a non-zero), instead of a `calloc` per
iteration.

## Yield while spinning in libgomp

Core of the patch, in `do_spin` of `libgomp/config/linux/wait.h`. The full
trunk version (env parsing, docs, `OMP_DISPLAY_ENV`) is
[writeups/libgomp_spin_yield_trunk.patch](writeups/libgomp_spin_yield_trunk.patch).

```c
static inline int do_spin (int *addr, int val)
{
  unsigned long long i, count = gomp_spin_count_var;
  unsigned long long yield_every = gomp_spin_yield_var, until_yield = yield_every;
  ...
  for (i = 0; i < count; i++)
    if (__builtin_expect (__atomic_load_n (addr, MEMMODEL_RELAXED) != val, 0))
      return 0;
    else
      {
        cpu_relax ();
        /* Give the CPU back from time to time: a thread of the team that
           blocked (e.g. on a mutex) may be waiting for a CPU to run on.  */
        if (yield_every && --until_yield == 0)
          {
            sched_yield ();
            until_yield = yield_every;
          }
      }
  return 1;
}
```

`GOMP_SPIN_YIELD=100` is the value to propose, 0 keeps the current behavior.
Building only libgomp from a GCC source tree is enough to test it.

## Standalone reproducers

Files in [writeups/](writeups/):

- `regions.c <none|malloc|omplock|mutex>`: time per parallel region when each
  thread does nothing more, allocates, or takes a lock once. The hybrid stall:
  `gcc -O2 -fopenmp regions.c -o regions && GOMP_SPINCOUNT=300000 ./regions
  mutex` gives ms per region on a hybrid laptop, ~0.1 ms with
  `GOMP_SPINCOUNT=1`, tens of us on the Xeon.
- `blas_region.c <lib> <dgemm symbol> <loop|dgemm>`: same with a
  single-threaded `dgemm` per chunk, the library loaded with `dlopen` so that
  the OpenMP runtime stays fixed.
- `sync_count.c`: `LD_PRELOAD` shim that counts `pthread_mutex_lock` calls,
  contended ones and `sched_yield` calls.
- `spin_pair.c`: fork/join with spin-then-futex waits, pthreads only, for the
  scheduler report. `./spin_pair 150 100 6000` (calls, regions per call,
  serial gap in us).
