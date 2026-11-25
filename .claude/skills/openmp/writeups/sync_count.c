// LD_PRELOAD shim counting pthread_mutex_lock calls (and contended ones) and
// sched_yield calls made through the PLT (e.g. by OpenBLAS, MKL, libomp).
#define _GNU_SOURCE
#include <dlfcn.h>
#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#include <stdio.h>

static atomic_long n_lock, n_contended, n_yield;

int pthread_mutex_lock(pthread_mutex_t *m) {
    static int (*real_lock)(pthread_mutex_t *);
    static int (*real_trylock)(pthread_mutex_t *);
    if (!real_lock) {
        real_lock = dlsym(RTLD_NEXT, "pthread_mutex_lock");
        real_trylock = dlsym(RTLD_NEXT, "pthread_mutex_trylock");
    }
    atomic_fetch_add(&n_lock, 1);
    if (real_trylock(m) == 0) return 0;
    atomic_fetch_add(&n_contended, 1);
    return real_lock(m);
}

int sched_yield(void) {
    static int (*real_yield)(void);
    if (!real_yield) real_yield = dlsym(RTLD_NEXT, "sched_yield");
    atomic_fetch_add(&n_yield, 1);
    return real_yield();
}

__attribute__((destructor)) static void report(void) {
    fprintf(stderr, " [mutex_lock=%ld contended=%ld sched_yield=%ld]",
            n_lock, n_contended, n_yield);
}
