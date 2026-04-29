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

// --- ADD: stb_image_write ---
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
    printf("cholmodthreads: %ld\n", c.nthreads_max);
    printf("\n");

    const SuiteSparse_long X = 1000, Y = 1000;
    const SuiteSparse_long n    = X * Y + 2;          // +2: source node 0, sink node X*Y+1
    const SuiteSparse_long sink = sink_idx(X, Y);      // = X*Y+1
    const SuiteSparse_long nnz2 = compute_nnz2(X, Y);

    auto Tgrid     = std::make_unique<double[]>(X * Y);
    auto Rgrid     = std::make_unique<double[]>(X * Y);
    auto Gvals     = std::make_unique<double[]>(nnz2);
    auto Rowlind   = std::make_unique<SuiteSparse_long[]>(nnz2);
    auto Colcoords = std::make_unique<SuiteSparse_long[]>(nnz2);

    Tgridset(Tgrid.get(), X, Y);
    Rgridset(Rgrid.get(), X, Y, 1000.0);
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

    // --- Free setup-only arrays ---
    Rowlind.reset();
    Colcoords.reset();
    Gvals.reset();

    // --- Precompute CSC indices for the sink column so we can pin it quickly
    // each iteration. The sink is the last column; Ap[sink]..Ap[sink+1]-1 are
    // all its entries in the lower-triangle CSC. We record the diagonal position
    // separately since that gets set to 1.0; all others get set to 0.0. ---
    SuiteSparse_long sink_col_start = Ap[sink];
    SuiteSparse_long sink_col_end   = Ap[sink + 1]; // exclusive
    SuiteSparse_long sink_diag_csc  = -1;
    for (SuiteSparse_long p = sink_col_start; p < sink_col_end; p++) {
        if (Ai[p] == sink) { sink_diag_csc = p; break; }
    }
    if (sink_diag_csc < 0) {
        printf("ERROR: could not find sink diagonal in CSC structure\n");
        return 1;
    }

    cholmod_factor* L = cholmod_l_analyze(A, &c);

    // Allocate b once. b[0]=1 (current into source), b[sink]=0 (sink pinned to 0V).
    // Neither entry changes across iterations.
    cholmod_dense* b = cholmod_l_zeros(n, 1, CHOLMOD_REAL, &c);
    ((double*)b->x)[0] = 1.0;
    // b[sink] is already 0 from cholmod_l_zeros — no explicit set needed.

    cholmod_dense* x = NULL;

    // solve2 workspaces: allocated on first call, reused every subsequent call
    cholmod_dense *Ywork = NULL, *Ework = NULL;

    printf("\n>>> ENTERING SIMULATION LOOP. MKL VERBOSE OUTPUT WILL BEGIN NOW. <<<\n\n");
    fflush(stdout);

    FILE* f = fopen("results.dat", "w");
    if (!f) { printf("ERROR: could not open results.dat\n"); exit(1); }
    fprintf(f, "# temp x0 loop_time_ms\n");

    // --- temperatures at which to save images ---
    std::set<int> save_temps = {305,340,345,350,375};

    for (int step = 0; step <= 75; step++) {
        auto step_start = high_resolution_clock::now();
        double temp = 300.0 + (double)step;
        printf("temp: %.2f\n", temp);

        // --- TIMING: fillG ---
        auto t_fillG_start = high_resolution_clock::now();

        for (SuiteSparse_long i = 0; i < X * Y; i++) if (temp >= Tgrid[i]) Rgrid[i] = 1.0;

        updateAx((double*)A->x, (SuiteSparse_long)A->nzmax,
                 Rgrid.get(), triplet_to_csc.get(), X, Y);

        // --- Pin sink node to 0V ---
        // Replace the sink column in Ax with a unit row: zero all off-diagonals,
        // set diagonal to 1.0. This enforces V[sink]=0 without making the matrix
        // singular. b[sink] is already 0, so the system correctly solves for
        // x[sink]=0 and leaves all other nodes unaffected.
        double* Ax = (double*)A->x;
        for (SuiteSparse_long p = sink_col_start; p < sink_col_end; p++)
            Ax[p] = 0.0;
        Ax[sink_diag_csc] = 1.0;

        auto t_fillG_end = high_resolution_clock::now();
        double fillG_ms = duration<double, std::milli>(t_fillG_end - t_fillG_start).count();

        // --- TIMING: factorize ---
        auto t_factor_start = high_resolution_clock::now();
        printf("--- Calling cholmod_l_factorize ---\n"); fflush(stdout);
        cholmod_l_factorize(A, L, &c);
        auto t_factor_end = high_resolution_clock::now();
        double factor_ms = duration<double, std::milli>(t_factor_end - t_factor_start).count();

        // --- TIMING: solve2 ---
        auto t_solve_start = high_resolution_clock::now();
        printf("--- Calling cholmod_l_solve2 ---\n"); fflush(stdout);
        cholmod_l_solve2(CHOLMOD_A, L, b, NULL, &x, NULL, &Ywork, &Ework, &c);
        auto t_solve_end = high_resolution_clock::now();
        double solve_ms = duration<double, std::milli>(t_solve_end - t_solve_start).count();

        // --- save image if temp matches ---
        int temp_int = static_cast<int>(temp + 0.5);
        if (save_temps.count(temp_int)) {
            char filename[64];
            sprintf(filename, "rgrid_%d.png", temp_int);
            save_rgrid_png(filename, Rgrid.get(), X, Y);
            printf("Saved image: %s\n", filename);
        }

        auto step_end = high_resolution_clock::now();
        fprintf(f, "%f %f\n", temp, ((double*)x->x)[0]);
        double total_ms = duration<double, std::milli>(step_end - step_start).count();

        if (x && x->x) {
            printf("Step %d | V[0]: %.4f\n", step, ((double*)x->x)[0]);
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
