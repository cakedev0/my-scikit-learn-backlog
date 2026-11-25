#include <omp.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Usage: regions <kind> ; kind in none, malloc, omplock, mutex */
int main(int argc, char **argv) {
    const char *kind = argv[1];
    int n_rep = 5000, n = 5000 * 50;
    double *x = malloc(n * sizeof(double));
    for (int i = 0; i < n; i++) x[i] = i * 1e-6;
    omp_lock_t lock;
    omp_init_lock(&lock);
    pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
    double total = 0;
    double t0 = 0;
    for (int rep = 0; rep < 2 * n_rep; rep++) {
        if (rep == n_rep) t0 = omp_get_wtime();
        double acc = 0;
        #pragma omp parallel reduction(+:acc)
        {
            double *buf = NULL;
            if (!strcmp(kind, "malloc")) buf = calloc(400, sizeof(double));
            #pragma omp for schedule(static)
            for (int i = 0; i < n; i++) acc += x[i] * x[i];
            if (!strcmp(kind, "omplock")) { omp_set_lock(&lock); total += 1; omp_unset_lock(&lock); }
            if (!strcmp(kind, "mutex")) { pthread_mutex_lock(&mutex); total += 1; pthread_mutex_unlock(&mutex); }
            if (buf) { buf[0] = acc; acc += buf[0] * 0; free(buf); }
        }
        total += acc;
    }
    printf("%8.1f us/region", (omp_get_wtime() - t0) / n_rep * 1e6);
    return total < 0;
}
