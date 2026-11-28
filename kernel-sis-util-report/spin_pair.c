// spin_pair.c: fork/join with spin-then-futex waiting (like libgomp), no OpenMP.
//
// Build:  gcc -O2 -pthread spin_pair.c -o spin_pair
// Run:    ./spin_pair [N_CALLS=150 N_REGIONS=100 GAP_US=6000 NTHREADS=#cpus]
//
// One call = N_REGIONS tiny fork/join regions (a few us of work per thread),
// then GAP_US of serial work on the main thread (a busy loop). Workers wait for
// the next region by spinning SPINCOUNT iterations of `pause` (default 300000,
// libgomp's default), then sleep in futex_wait. Set SPINCOUNT=1 to disable the
// spinning, SPIN_PIN=1 to pin thread i on the i-th allowed CPU.
//
// Prints the time per call and how often two threads of the team start a region
// on the same CPU ("stacked"). A stacked region costs ~3 ms (one time slice)
// instead of ~5 us, because each region needs both threads.
// With SPIN_DUMP=file, writes one line per region ("k t0 t1 cpu_of_thread0..."),
// timestamps in CLOCK_MONOTONIC seconds, to be matched with `perf record -k
// CLOCK_MONOTONIC --switch-events`.
#define _GNU_SOURCE
#include <limits.h>
#include <linux/futex.h>
#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

#define ROWS 1000
static atomic_int gen, done, nsleep, msleep;
static int nt, pin;
static long spincount = 300000;
static double x[ROWS], thr[255];
static unsigned char out[ROWS];
static int *cpu_rec;  // cpu_rec[region * nt + thread]
static int tids[1024], cpus_allowed[1024], n_allowed;

static double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}
// Spin hint: `pause` on x86, `isb` on arm64 (`yield` is a nop on Neoverse, so
// 300000 iterations would be much shorter than on x86).
#if defined(__x86_64__) || defined(__i386__)
#define cpu_relax() __builtin_ia32_pause()
#elif defined(__aarch64__)
#define cpu_relax() __asm__ volatile("isb" ::: "memory")
#else
#define cpu_relax() __asm__ volatile("" ::: "memory")
#endif

static void fwait(atomic_int *a, int v) { syscall(SYS_futex, a, FUTEX_WAIT_PRIVATE, v, NULL, NULL, 0); }
static void fwake(atomic_int *a) { syscall(SYS_futex, a, FUTEX_WAKE_PRIVATE, INT_MAX, NULL, NULL, 0); }

static void do_work(int id, int region) {
    cpu_rec[(long)region * nt + id] = sched_getcpu();
    int lo_i = ROWS * id / nt, hi_i = ROWS * (id + 1) / nt;
    for (int i = lo_i; i < hi_i; i++) {  // binary search, like HGB binning
        int lo = 0, hi = 255;
        while (lo < hi) { int m = (lo + hi) / 2; if (x[i] > thr[m]) lo = m + 1; else hi = m; }
        out[i] = lo;
    }
}

static void pin_to(int id) {
    if (!pin) return;
    cpu_set_t s; CPU_ZERO(&s); CPU_SET(cpus_allowed[id % n_allowed], &s);
    sched_setaffinity(0, sizeof s, &s);
}

static void *worker(void *arg) {
    int id = (int)(long)arg, seen = 0, g;
    tids[id] = syscall(SYS_gettid);
    pin_to(id);
    for (;;) {
        for (long i = 0; i < spincount; i++) {  // like libgomp's do_spin
            g = atomic_load_explicit(&gen, memory_order_acquire);
            if (g != seen) goto go;
            cpu_relax();
        }
        atomic_fetch_add(&nsleep, 1);
        while ((g = atomic_load(&gen)) == seen) fwait(&gen, seen);
        atomic_fetch_sub(&nsleep, 1);
    go:
        if (g < 0) return NULL;
        seen = g;
        do_work(id, g);
        if (atomic_fetch_add(&done, 1) + 1 == nt - 1 && atomic_load(&msleep)) fwake(&done);
    }
}

static int cmp(const void *a, const void *b) { double d = *(double *)a - *(double *)b; return (d > 0) - (d < 0); }

int main(int argc, char **argv) {
    int n_calls = argc > 1 ? atoi(argv[1]) : 150, n_regions = argc > 2 ? atoi(argv[2]) : 100;
    int gap_us = argc > 3 ? atoi(argv[3]) : 6000;
    cpu_set_t aff; sched_getaffinity(0, sizeof aff, &aff);
    for (int c = 0; c < CPU_SETSIZE; c++) if (CPU_ISSET(c, &aff)) cpus_allowed[n_allowed++] = c;
    nt = argc > 4 ? atoi(argv[4]) : n_allowed;
    if (getenv("SPINCOUNT")) spincount = atol(getenv("SPINCOUNT"));
    pin = getenv("SPIN_PIN") != NULL;
    for (int i = 0; i < ROWS; i++) x[i] = (double)rand() / RAND_MAX;
    for (int b = 0; b < 255; b++) thr[b] = (b + 1) / 256.0;
    long n_tot = (long)n_calls * n_regions;
    cpu_rec = calloc((n_tot + 1) * nt, sizeof(int));
    double *t0 = malloc(sizeof(double) * (n_tot + 1)), *t1 = malloc(sizeof(double) * (n_tot + 1));
    double *call_ms = malloc(sizeof(double) * n_calls);

    pthread_t th[1024];
    tids[0] = syscall(SYS_gettid);
    pin_to(0);
    for (int i = 1; i < nt; i++) pthread_create(&th[i], NULL, worker, (void *)(long)i);
    usleep(100000);

    volatile double sink = 0;
    int g = 0;
    for (int c = 0; c < n_calls; c++) {
        double tg = now();
        while ((now() - tg) * 1e6 < gap_us) sink += 1;  // serial work between calls
        double tc = now();
        for (int r = 0; r < n_regions; r++) {
            long k = (long)c * n_regions + r;
            g++;
            atomic_store(&done, 0);
            t0[k] = now();
            atomic_store(&gen, g);
            if (atomic_load(&nsleep)) fwake(&gen);
            do_work(0, g);
            long i;
            for (i = 0; i < spincount; i++) {
                if (atomic_load_explicit(&done, memory_order_acquire) == nt - 1) break;
                cpu_relax();
            }
            if (i == spincount) {
                int d;
                atomic_store(&msleep, 1);
                while ((d = atomic_load(&done)) < nt - 1) fwait(&done, d);
                atomic_store(&msleep, 0);
            }
            t1[k] = now();
        }
        call_ms[c] = (now() - tc) * 1e3;
    }
    atomic_store(&gen, -1);
    fwake(&gen);
    for (int i = 1; i < nt; i++) pthread_join(th[i], NULL);

    // Stacked regions: two threads of the team started the region on the same CPU.
    long n_slow = 0, n_stack = 0, n_ep = 0; double ep_total = 0, longest = 0, ep_start = 0; int in_ep = 0;
    for (long k = 0; k < n_tot; k++) {
        int seen[1024] = {0}, stacked = 0;
        for (int t = 0; t < nt; t++) if (seen[cpu_rec[(k + 1) * nt + t]]++) stacked = 1;
        if ((t1[k] - t0[k]) * 1e3 > 1.0) n_slow++;
        if (stacked) { n_stack++; if (!in_ep) { in_ep = 1; ep_start = t0[k]; n_ep++; } }
        if ((!stacked || k == n_tot - 1) && in_ep) {
            double d = (t1[k - !stacked] - ep_start) * 1e3; ep_total += d; if (d > longest) longest = d; in_ep = 0;
        }
    }
    double *s = malloc(sizeof(double) * n_calls); memcpy(s, call_ms, sizeof(double) * n_calls);
    qsort(s, n_calls, sizeof(double), cmp);
    printf("threads=%d spincount=%ld gap=%dus: call median %.3f p90 %.3f max %.1f ms (a call is %d regions); "
           "regions > 1 ms: %ld/%ld; regions with 2 threads on one CPU: %ld; episodes: %ld, total %.0f ms, longest %.0f ms\n",
           nt, spincount, gap_us, s[n_calls / 2], s[n_calls * 9 / 10], s[n_calls - 1], n_regions, n_slow, n_tot,
           n_stack, n_ep, ep_total, longest);

    if (getenv("SPIN_DUMP")) {
        FILE *f = fopen(getenv("SPIN_DUMP"), "w");
        fprintf(f, "tids");
        for (int t = 0; t < nt; t++) fprintf(f, " %d", tids[t]);
        fprintf(f, "\n");
        for (long k = 0; k < n_tot; k++) {
            fprintf(f, "%ld %.9f %.9f", k, t0[k], t1[k]);
            for (int t = 0; t < nt; t++) fprintf(f, " %d", cpu_rec[(k + 1) * nt + t]);
            fprintf(f, "\n");
        }
        fclose(f);
    }
    return 0;
}
