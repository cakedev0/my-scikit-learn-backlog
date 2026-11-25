// Parallel regions where each chunk does a small matrix product, either with
// BLAS dgemm (single threaded, called from OpenMP threads) or a plain loop.
// Usage: blas_region <blas library> <dgemm symbol> <loop|dgemm>
#include <dlfcn.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef void (*dgemm_t)(const char *, const char *, const int *, const int *,
                        const int *, const double *, const double *, const int *,
                        const double *, const int *, const double *, double *,
                        const int *);

#define N_CHUNKS 20
#define M 256  // samples per chunk
#define K 50   // features
#define N 8    // clusters

int main(int argc, char **argv) {
    void *lib = dlopen(argv[1], RTLD_NOW | RTLD_GLOBAL);
    if (!lib) { fprintf(stderr, "%s\n", dlerror()); return 1; }
    dgemm_t dgemm = (dgemm_t)dlsym(lib, argv[2]);
    int use_blas = !strcmp(argv[3], "dgemm");
    double *A = malloc(N_CHUNKS * M * K * sizeof(double));
    double *B = malloc(K * N * sizeof(double));
    double *C = malloc(N_CHUNKS * M * N * sizeof(double));
    for (int i = 0; i < N_CHUNKS * M * K; i++) A[i] = (i % 97) * 1e-2;
    for (int i = 0; i < K * N; i++) B[i] = (i % 13) * 1e-2;
    int m = M, n = N, k = K;
    double alpha = 1.0, beta = 0.0;
    int n_rep = 3000;
    double t0 = 0;
    for (int rep = 0; rep < 2 * n_rep; rep++) {
        if (rep == n_rep) t0 = omp_get_wtime();
        #pragma omp parallel for schedule(static)
        for (int c = 0; c < N_CHUNKS; c++) {
            double *a = A + c * M * K, *out = C + c * M * N;
            if (use_blas) {
                // out (M x N, col-major) = a (M x K, col-major) @ B (K x N)
                dgemm("N", "N", &m, &n, &k, &alpha, a, &m, B, &k, &beta, out, &m);
            } else {
                for (int j = 0; j < N; j++)
                    for (int i = 0; i < M; i++) {
                        double s = 0;
                        for (int l = 0; l < K; l++) s += a[i + l * M] * B[l + j * K];
                        out[i + j * M] = s;
                    }
            }
        }
    }
    printf("%8.1f us/region", (omp_get_wtime() - t0) / n_rep * 1e6);
    FILE *maps = fopen("/proc/self/maps", "r");
    char line[4096];
    while (fgets(line, sizeof(line), maps)) {
        char *p = strstr(line, "libgomp");
        if (p && strstr(line, "r-xp")) {
            line[strcspn(line, "\n")] = 0;
            fprintf(stderr, " (%s)", strrchr(line, ' ') + 1);
            break;
        }
    }
    fclose(maps);
    return C[0] < -1;
}
