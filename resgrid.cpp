#include "cholmod.h"
#include <stdio.h>
#include <iostream>
#include <vector>
#include <stdint.h>
#include <memory>
#include <algorithm>
#include <chrono>
#include <stdlib.h>
#include <set>
#include "Header.h"
#include "visualization.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

using namespace std::chrono;

int main(void) {
    // --- MKL DIAGNOSTIC MODE ---
    const char* threads = "4"; 
    _putenv_s("MKL_PARDISO_ALIGNMENT", "64"); 
    _putenv_s("MKL_INTERFACE_LAYER", "ILP64");
    _putenv_s("MKL_THREADING_LAYER", "INTEL");
    _putenv_s("MKL_NUM_THREADS", "1");

    printf("--- SuiteSparse Diagnostic: MKL_VERBOSE enabled (%s Threads) ---\n", threads);

    cholmod_common c;
    cholmod_l_start(&c);
    c.supernodal = CHOLMOD_SUPERNODAL; 
    c.supernodal_switch = 80.0;
    c.final_ll = 1;
    c.nmethods = 1;
    c.method[0].ordering = CHOLMOD_NESDIS;
    c.method[0].nd_small = 100;
    c.nthreads_max = 1;
    printf("cholmodthreads: %ld\n\n", c.nthreads_max);

    const SuiteSparse_long X = 1000, Y = 1000;
    // +1: source node 0. The explicit sink node is completely removed (implicit ground).
    const SuiteSparse_long n    = X * Y + 1;          
    const SuiteSparse_long nnz2 = compute_nnz2(X, Y);

    auto Tgrid     = std::make_unique<double[]>(X * Y);
    auto Rgrid     = std::make_unique<double[]>(X * Y);
    auto Gvals     = std::make_unique<double[]>(nnz2);
    auto Rowlind   = std::make_unique<SuiteSparse_long[]>(nnz2);
    auto Colcoords = std::make_unique<SuiteSparse_long[]>(nnz2);

    Tgridset(Tgrid.get(), X, Y);
    Rgridset(Rgrid.get(), X, Y, 1000.0);
    
    // Precompute the cascading trigger temperatures
    printf("Precomputing trigger temperature cascades...\n");
    precomputeTriggerTemps(Tgrid.get(), X, Y, 15.0);

    constructRowlind(Rowlind.get(), X, Y);
    constructColind(Colcoords.get(), X, Y);
    fillGvals(Gvals.get(), Rgrid.get(), X, Y);
    fillGvalsSourceDiag(Gvals.get(), Y);

    // Build triplet and convert to CSC
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

    // Build triplet_to_csc mapping
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

    // Free setup-only arrays
    Rowlind.reset();
    Colcoords.reset();
    Gvals.reset();

    cholmod_factor* L = cholmod_l_analyze(A, &c);

    // Allocate b once. b[0]=1 (current into source). 
    // Grid nodes implicitly have b[k]=0. Ground is naturally defined by the math.
    cholmod_dense* b = cholmod_l_zeros(n, 1, CHOLMOD_REAL, &c);
    ((double*)b->x)[0] = 1.0;

    cholmod_dense* x = NULL;
    cholmod_dense *Ywork = NULL, *Ework = NULL;

    printf("\n>>> ENTERING SIMULATION LOOP. MKL VERBOSE OUTPUT WILL BEGIN NOW. <<<\n\n");
    fflush(stdout);

    FILE* f = fopen("results.dat", "w");
    if (!f) { printf("ERROR: could not open results.dat\n"); exit(1); }
    fprintf(f, "# temp x0 loop_time_ms\n");

    // --- Automated, Physics-Biased Image Scheduler ---
    double start_temp = 300.0;
    double end_temp   = 375.0;
    double mean_temp  = 345.0; // The threshold distribution mean

    // Generate 15 target temperatures strongly clustered around 345K
    std::set<int> temp_set = generateHighlyBiasedTemps(start_temp, end_temp, mean_temp, 15, 5.0);
    std::vector<int> sorted_temps(temp_set.begin(), temp_set.end());
    size_t next_img_idx = 0;

    printf("Image capture schedule: ");
    for(int t : sorted_temps) printf("%d ", t);
    printf("\n\n");

    int total_steps = static_cast<int>(end_temp - start_temp);

    for (int step = 0; step <= total_steps; step++) {
        auto step_start = high_resolution_clock::now();
        double temp = start_temp + (double)step;
        printf("temp: %.2f\n", temp);

        // --- TIMING: fillG ---
        auto t_fillG_start = high_resolution_clock::now();

        // The precomputation baked the cascade into Tgrid, so this is just a fast check
        for (SuiteSparse_long i = 0; i < X * Y; i++) {
            if (temp >= Tgrid[i]) Rgrid[i] = 1.0;
        }

        updateAx((double*)A->x, (SuiteSparse_long)A->nzmax,
                 Rgrid.get(), triplet_to_csc.get(), X, Y);

        // (No pinning code required here anymore)

        auto t_fillG_end = high_resolution_clock::now();
        double fillG_ms = duration<double, std::milli>(t_fillG_end - t_fillG_start).count();

        // --- TIMING: factorize ---
        auto t_factor_start = high_resolution_clock::now();
        cholmod_l_factorize(A, L, &c);
        auto t_factor_end = high_resolution_clock::now();
        double factor_ms = duration<double, std::milli>(t_factor_end - t_factor_start).count();

        // --- TIMING: solve2 ---
        auto t_solve_start = high_resolution_clock::now();
        cholmod_l_solve2(CHOLMOD_A, L, b, NULL, &x, NULL, &Ywork, &Ework, &c);
        auto t_solve_end = high_resolution_clock::now();
        double solve_ms = duration<double, std::milli>(t_solve_end - t_solve_start).count();

        // --- Check and Save Image ---
        if (next_img_idx < sorted_temps.size() && temp >= sorted_temps[next_img_idx]) {
            char filename[64];
            sprintf(filename, "rgrid_%d.png", sorted_temps[next_img_idx]);
            save_rgrid_png(filename, Rgrid.get(), X, Y);
            printf("Saved image: %s\n", filename);
            next_img_idx++; // Move to the next target
        }

        auto step_end = high_resolution_clock::now();
        fprintf(f, "%f %f\n", temp, ((double*)x->x)[0]);
        double total_ms = duration<double, std::milli>(step_end - step_start).count();

        if (x && x->x) {
            printf("Step %d | V[0] (Total Resistance): %.4f\n", step, ((double*)x->x)[0]);
            printf("  fillG:     %8.3f ms\n", fillG_ms);
            printf("  factorize: %8.3f ms\n", factor_ms);
            printf("  solve:     %8.3f ms\n", solve_ms);
            printf("  total:     %8.3f ms\n", total_ms);
        }
        fflush(stdout);
    }

    fclose(f);

    // --- Free all resources ---
    if (Ywork) cholmod_l_free_dense(&Ywork, &c);
    if (Ework) cholmod_l_free_dense(&Ework, &c);
    if (x)     cholmod_l_free_dense(&x, &c);
    if (b)     cholmod_l_free_dense(&b, &c);

    cholmod_l_free_factor(&L, &c);
    cholmod_l_free_sparse(&A, &c);
    cholmod_l_finish(&c);
    return 0;
}