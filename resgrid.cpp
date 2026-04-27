#include "cholmod.h"
#include <stdio.h>
#include <iostream>
#include <vector>
#include <stdint.h>
#include <memory>
#include <algorithm>
#include <chrono>
#include <stdlib.h>
#include "Header.h"

using namespace std::chrono;

static void my_handler(int status, const char* file, int line, const char* message) {
    fprintf(stderr, "\n[CHOLMOD ERROR] Status %d: %s\n", status, message);
}

int main(void) {
    // --- 1. NEW PARALLEL STRATEGY ---
    const char* threads = "12"; 
    _putenv_s("MKL_NUM_THREADS", threads);
    _putenv_s("OMP_NUM_THREADS", threads);
    _putenv_s("MKL_DYNAMIC", "FALSE");
    
    // Bypasses CPU-specific dispatching bugs that can cause alignment crashes
    _putenv_s("MKL_SKIP_CPU_CHECK", "1"); 
    // Ensure thread stack is large enough for MKL's dense kernels
    _putenv_s("KMP_STACKSIZE", "32M");

    printf("--- SuiteSparse Diagnostic: Testing %s Threads (Strategy Shift) ---\n", threads);

    // --- 2. INITIALIZE ---
    cholmod_common c;
    cholmod_l_start(&c);
    c.error_handler = my_handler;
    
    c.supernodal = CHOLMOD_SUPERNODAL; 
    
    // Shift threshold: Larger blocks are more stable for MKL threading
    c.supernodal_switch = 100.0; 
    
    c.final_ll = 1;
    c.nmethods = 1;
    c.method[0].ordering = CHOLMOD_AMD;

    const SuiteSparse_long X = 1000, Y = 1000;
    const SuiteSparse_long n = X * Y + 1;
    const SuiteSparse_long nnz2 = compute_nnz2(X, Y);

    auto Tgrid = std::make_unique<double[]>(X * Y);
    auto Rgrid = std::make_unique<double[]>(X * Y);
    auto Gvals = std::make_unique<double[]>(nnz2);
    auto Rowlind = std::make_unique<SuiteSparse_long[]>(nnz2);
    auto Colcoords = std::make_unique<SuiteSparse_long[]>(nnz2);

    Tgridset(Tgrid.get(), X, Y);
    Rgridset(Rgrid.get(), X, Y, 1000.0);
    
    constructRowlind(Rowlind.get(), X, Y);
    constructColind(Colcoords.get(), X, Y);
    fillGvals(Gvals.get(), Rgrid.get(), X, Y);
    fillGvalsSourceDiag(Gvals.get(), Y);

    cholmod_triplet* T = cholmod_l_allocate_triplet(n, n, nnz2, -1, CHOLMOD_REAL, &c);
    for (SuiteSparse_long k = 0; k < nnz2; k++) {
        ((SuiteSparse_long*)T->i)[k] = Rowlind[k];
        ((SuiteSparse_long*)T->j)[k] = Colcoords[k];
        ((double*)T->x)[k] = Gvals[k];
    }
    T->nnz = nnz2;
    cholmod_sparse* A = cholmod_l_triplet_to_sparse(T, nnz2, &c);
    A->stype = -1;
    cholmod_l_free_triplet(&T, &c);

    // CSC Map
    SuiteSparse_long* Ap = (SuiteSparse_long*)A->p;
    SuiteSparse_long* Ai = (SuiteSparse_long*)A->i;
    auto triplet_to_csc = std::make_unique<SuiteSparse_long[]>(nnz2);
    for (SuiteSparse_long k = 0; k < nnz2; k++) {
        triplet_to_csc[k] = -1;
        SuiteSparse_long r = Rowlind[k], col = Colcoords[k];
        if (r < col) std::swap(r, col);
        SuiteSparse_long lo = Ap[col], hi = Ap[col + 1] - 1;
        while (lo <= hi) {
            SuiteSparse_long mid = lo + (hi - lo) / 2;
            if (Ai[mid] < r) lo = mid + 1;
            else if (Ai[mid] > r) hi = mid - 1;
            else { triplet_to_csc[k] = mid; break; }
        }
    }

    // Symbolic Analysis
    printf("Analyzing... ");
    fflush(stdout);
    cholmod_factor* L = cholmod_l_analyze(A, &c);
    printf("DONE (Supernodes: %lld)\n", (long long)L->nsuper);

    printf("Starting Simulation Loop...\n");
    cholmod_dense *x = NULL, *b = NULL;

    for (int step = 0; step <= 2; step++) {
        auto step_start = high_resolution_clock::now();
        
        double temp = 200.0 + (double)step;
        for (SuiteSparse_long i = 0; i < X * Y; i++)
            if (temp >= Tgrid[i]) Rgrid[i] = 1.0;

        fillGvals(Gvals.get(), Rgrid.get(), X, Y);
        fillGvalsSourceDiag(Gvals.get(), Y);

        double* Ax = (double*)A->x;
        std::fill(Ax, Ax + A->nzmax, 0.0); 
        for (SuiteSparse_long k = 0; k < nnz2; k++) {
            if (triplet_to_csc[k] >= 0) Ax[triplet_to_csc[k]] += Gvals[k];
        }

        printf("Step %d: Factoring...", step);
        fflush(stdout);
        
        if (!cholmod_l_factorize(A, L, &c)) {
            printf(" FAILED.\n");
            break;
        }

        printf(" Solving...");
        fflush(stdout);

        if (b) cholmod_l_free_dense(&b, &c);
        b = cholmod_l_zeros(n, 1, CHOLMOD_REAL, &c);
        ((double*)b->x)[0] = 1.0; 

        if (x) cholmod_l_free_dense(&x, &c);
        x = cholmod_l_solve(CHOLMOD_A, L, b, &c);

        auto step_end = high_resolution_clock::now();
        if (x) {
            printf(" SUCCESS! (%.2f ms)\n", duration<double, std::milli>(step_end - step_start).count());
        }
        fflush(stdout);
    }

    cholmod_l_free_factor(&L, &c);
    cholmod_l_free_sparse(&A, &c);
    cholmod_l_finish(&c);
    return 0;
}