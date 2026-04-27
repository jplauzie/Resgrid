#include "cholmod.h"
#include <stdio.h>
#include <iostream>
#include <vector>
#include <stdint.h>
#include <memory>
#include <algorithm>
#include <chrono>
#include <stdlib.h> // for _putenv_s
#include "Header.h"

using namespace std::chrono;

// Custom error handler to catch SuiteSparse internal issues
static void my_handler(int status, const char* file, int line, const char* message) {
    fprintf(stderr, "\n[CHOLMOD ERROR] Status %d: %s (at %s:%d)\n", status, message, file, line);
    fflush(stderr);
}

int main(void) {
    // ---------------------------------------------------------
    // 1. PERFORMANCE & THREADING CONFIGURATION
    // ---------------------------------------------------------
    // Based on testing, "1" is likely your fastest setting due to memory bandwidth.
    const char* threads = "4"; 
    _putenv_s("MKL_NUM_THREADS", threads);
    _putenv_s("OMP_NUM_THREADS", threads);
    _putenv_s("MKL_DYNAMIC", "FALSE");

    if (sizeof(void*) < 8) {
        printf("CRITICAL ERROR: This program must be compiled for x64.\n");
        return 1;
    }

    printf("--- SuiteSparse 1000x1000 Optimized Solver (%s Thread) ---\n", threads);

    // ---------------------------------------------------------
    // 2. CHOLMOD INITIALIZATION
    // ---------------------------------------------------------
    cholmod_common c;
    cholmod_l_start(&c);
    c.error_handler = my_handler;
    
    // Core performance settings
    c.supernodal = CHOLMOD_SUPERNODAL; 
    c.supernodal_switch = 40.0; // Matches MATLAB threshold
    c.final_ll = 1;             // Force LL' factorization
    c.nmethods = 1;
    c.method[0].ordering = CHOLMOD_AMD; 

    const SuiteSparse_long X = 1000, Y = 1000;
    const SuiteSparse_long n = X * Y + 1;
    const SuiteSparse_long nnz2 = compute_nnz2(X, Y);

    // ---------------------------------------------------------
    // 3. HEAP MEMORY ALLOCATION
    // ---------------------------------------------------------
    auto Tgrid = std::make_unique<double[]>(X * Y);
    auto Rgrid = std::make_unique<double[]>(X * Y);
    auto Gvals = std::make_unique<double[]>(nnz2);
    auto Rowlind = std::make_unique<SuiteSparse_long[]>(nnz2);
    auto Colcoords = std::make_unique<SuiteSparse_long[]>(nnz2);

    Tgridset(Tgrid.get(), X, Y);
    Rgridset(Rgrid.get(), X, Y, 1000.0);
    
    printf("Setting up matrix structure... ");
    constructRowlind(Rowlind.get(), X, Y);
    constructColind(Colcoords.get(), X, Y);
    fillGvals(Gvals.get(), Rgrid.get(), X, Y);
    fillGvalsSourceDiag(Gvals.get(), Y);
    printf("Done.\n");

    // ---------------------------------------------------------
    // 4. MATRIX CONSTRUCTION (Triplet to CSC)
    // ---------------------------------------------------------
    cholmod_triplet* T = cholmod_l_allocate_triplet(n, n, nnz2, -1, CHOLMOD_REAL, &c);
    for (SuiteSparse_long k = 0; k < nnz2; k++) {
        ((SuiteSparse_long*)T->i)[k] = Rowlind[k];
        ((SuiteSparse_long*)T->j)[k] = Colcoords[k];
        ((double*)T->x)[k] = Gvals[k];
    }
    T->nnz = nnz2;

    cholmod_sparse* A = cholmod_l_triplet_to_sparse(T, nnz2, &c);
    A->stype = -1; // Lower symmetric
    cholmod_l_free_triplet(&T, &c);

    // ---------------------------------------------------------
    // 5. CSC MAPPING (For high-speed numeric updates)
    // ---------------------------------------------------------
    printf("Mapping Triplet to CSC... ");
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
    printf("Done.\n");

    // 6. Symbolic Factorization (Once per simulation)
    printf("Performing Symbolic Analysis... ");
    fflush(stdout);
    cholmod_factor* L = cholmod_l_analyze(A, &c);
    printf("Done.\n");

    // ---------------------------------------------------------
    // 7. SIMULATION LOOP
    // ---------------------------------------------------------
    printf("Starting Simulation...\n");
    cholmod_dense *x = NULL, *b = NULL;

    for (int step = 0; step <= 5; step++) {
        auto step_start = high_resolution_clock::now();
        double temp = 200.0 + (double)step;

        // Physics: Update resistance values
        for (SuiteSparse_long i = 0; i < X * Y; i++)
            if (temp >= Tgrid[i]) Rgrid[i] = 1.0;

        fillGvals(Gvals.get(), Rgrid.get(), X, Y);
        fillGvalsSourceDiag(Gvals.get(), Y);

        // Fast Matrix Update: Update A->x without reallocating
        double* Ax = (double*)A->x;
        std::fill(Ax, Ax + A->nzmax, 0.0); 
        for (SuiteSparse_long k = 0; k < nnz2; k++) {
            if (triplet_to_csc[k] >= 0) Ax[triplet_to_csc[k]] += Gvals[k];
        }

        // Numeric Factorization
        if (!cholmod_l_factorize(A, L, &c)) {
            printf("Factorization Failed at Step %d\n", step);
            break;
        }

        // Prepare RHS vector
        if (b) cholmod_l_free_dense(&b, &c);
        b = cholmod_l_zeros(n, 1, CHOLMOD_REAL, &c);
        ((double*)b->x)[0] = 1.0; // 1V Source

        // Solve
        if (x) cholmod_l_free_dense(&x, &c);
        x = cholmod_l_solve(CHOLMOD_A, L, b, &c);

        auto step_end = high_resolution_clock::now();
        double duration_ms = duration<double, std::milli>(step_end - step_start).count();

        if (x && x->x) {
            printf("Step %d: %.2f ms | V[0]: %.4f\n", step, duration_ms, ((double*)x->x)[0]);
        }
        fflush(stdout);
    }

    // ---------------------------------------------------------
    // 8. CLEANUP
    // ---------------------------------------------------------
    if (x) cholmod_l_free_dense(&x, &c);
    if (b) cholmod_l_free_dense(&b, &c);
    cholmod_l_free_factor(&L, &c);
    cholmod_l_free_sparse(&A, &c);
    cholmod_l_finish(&c);
    return 0;
}