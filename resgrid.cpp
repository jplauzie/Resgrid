#include "cholmod.h"
#include <stdio.h>
#include <iostream>
#include <vector>
#include <memory>
#include <chrono>
#include <stdlib.h>
#include <set>
#include "Header.h"
#include "visualization.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

using namespace std::chrono;

int main(void) {
    // --- MKL DIAGNOSTICS & SETUP ---
    _putenv_s("MKL_INTERFACE_LAYER", "ILP64");
    _putenv_s("MKL_THREADING_LAYER", "INTEL");
    _putenv_s("MKL_NUM_THREADS", "1");

    cholmod_common c;
    cholmod_l_start(&c);
    c.supernodal = CHOLMOD_SUPERNODAL;
    c.nthreads_max = 1;

    const SuiteSparse_long X = 100, Y = 100;
    const SuiteSparse_long n = X * Y + 1;
    const SuiteSparse_long nnz2 = compute_nnz2(X, Y);

    auto T_base     = std::make_unique<double[]>(X * Y);
    auto Tgrid_up   = std::make_unique<double[]>(X * Y);
    auto Tgrid_down = std::make_unique<double[]>(X * Y);
    auto Rgrid      = std::make_unique<double[]>(X * Y);

    // Initial random distribution (Shifted mean to account for J-pull)
    Tgridset(T_base.get(), X, Y, 345.0, 10.0); 
    std::copy(T_base.get(), T_base.get() + X * Y, Tgrid_up.get());
    std::copy(T_base.get(), T_base.get() + X * Y, Tgrid_down.get());

    printf("Precomputing Heating & Cooling Cascades (J=15.0)...\n");
    precomputeHeatingMap(Tgrid_up.get(), X, Y, 20.0);
    precomputeCoolingMap(Tgrid_down.get(), X, Y, 20.0);

    Rgridset(Rgrid.get(), X, Y, 1000.0);

    auto Rowlind = std::make_unique<SuiteSparse_long[]>(nnz2);
    auto Colcoords = std::make_unique<SuiteSparse_long[]>(nnz2);
    constructRowlind(Rowlind.get(), X, Y);
    constructColind(Colcoords.get(), X, Y);

    // Matrix setup
    cholmod_triplet* T = cholmod_l_allocate_triplet(n, n, nnz2, -1, CHOLMOD_REAL, &c);
    for (SuiteSparse_long k = 0; k < nnz2; k++) {
        ((SuiteSparse_long*)T->i)[k] = Rowlind[k];
        ((SuiteSparse_long*)T->j)[k] = Colcoords[k];
        ((double*)T->x)[k] = 0.0;
    }
    T->nnz = nnz2;
    cholmod_sparse* A = cholmod_l_triplet_to_sparse(T, nnz2, &c);
    A->stype = -1;
    cholmod_l_free_triplet(&T, &c);

    // Build triplet_to_csc mapping
    auto triplet_to_csc = std::make_unique<SuiteSparse_long[]>(nnz2);
    SuiteSparse_long* Ap = (SuiteSparse_long*)A->p;
    SuiteSparse_long* Ai = (SuiteSparse_long*)A->i;
    for (SuiteSparse_long k = 0; k < nnz2; k++) {
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

    cholmod_factor* L = cholmod_l_analyze(A, &c);
    cholmod_dense* b = cholmod_l_zeros(n, 1, CHOLMOD_REAL, &c);
    ((double*)b->x)[0] = 1.0;
    cholmod_dense *x = NULL, *Ywork = NULL, *Ework = NULL;

    double start_temp = 300.0, end_temp = 375.0;
    int total_steps = (int)(end_temp - start_temp);

    // --- HEATING LOOP ---
    printf("\n>>> STARTING HEATING CYCLE <<<\n");
    FILE* f1 = fopen("results_up.dat", "w");
    std::set<int> h_set = generateHighlyBiasedTemps(start_temp, end_temp, 345.0, 15, 5.0);
    std::vector<int> sched_up(h_set.begin(), h_set.end());
    size_t up_idx = 0;

    for (int step = 0; step <= total_steps; step++) {
        auto s_start = high_resolution_clock::now();
        double temp = start_temp + (double)step;

        double current_insulator_R = getSemiconductorR(temp);

        // Update the entire grid based on current temp
        for (int i = 0; i < X * Y; i++) {
            if (temp >= Tgrid_up[i]) {
                Rgrid[i] = 1.0; // Metallic phase
            } else {
                Rgrid[i] = current_insulator_R; // Semiconductor phase
            }
        }

        auto t_fill = high_resolution_clock::now();
        updateAx((double*)A->x, A->nzmax, Rgrid.get(), triplet_to_csc.get(), X, Y);
        auto t_fact = high_resolution_clock::now();
        cholmod_l_factorize(A, L, &c);
        auto t_solve = high_resolution_clock::now();
        cholmod_l_solve2(CHOLMOD_A, L, b, NULL, &x, NULL, &Ywork, &Ework, &c);
        auto t_end = high_resolution_clock::now();

        fprintf(f1, "%f %f\n", temp, ((double*)x->x)[0]);
        
        if (up_idx < sched_up.size() && temp >= sched_up[up_idx]) {
            char fn[64]; sprintf(fn, "heat_%d.png", sched_up[up_idx]);
            save_rgrid_png(fn, Rgrid.get(), X, Y);
            up_idx++;
        }

        printf("Step %d (H) | T: %.1f | R_tot: %.4f | Fact: %.1fms | Total: %.1fms\n", 
               step, temp, ((double*)x->x)[0], 
               duration<double, std::milli>(t_solve - t_fact).count(),
               duration<double, std::milli>(t_end - s_start).count());
    }
    fclose(f1);

    // --- COOLING LOOP ---
    printf("\n>>> STARTING COOLING CYCLE <<<\n");
    FILE* f2 = fopen("results_down.dat", "w");
    std::set<int> c_set = generateHighlyBiasedTemps(start_temp, end_temp, 335.0, 15, 5.0);
    std::vector<int> sched_down(c_set.begin(), c_set.end());
    std::reverse(sched_down.begin(), sched_down.end());
    size_t dn_idx = 0;

    for (int step = total_steps; step >= 0; step--) {
        auto s_start = high_resolution_clock::now();
        double temp = start_temp + (double)step;

        double current_insulator_R = getSemiconductorR(temp);

        auto rgrid = high_resolution_clock::now();
        for (int i = 0; i < X * Y; i++) {
            if (temp < Tgrid_down[i]) {
                Rgrid[i] = current_insulator_R; // Reverted to semiconductor
            } else {
                Rgrid[i] = 1.0; // Still metallic
            }
        }
        auto rgrid_end = high_resolution_clock::now();
        printf("Rgrid  %.1fms\n", duration<double, std::milli>(rgrid_end - rgrid).count());

        auto t_fill = high_resolution_clock::now();
        updateAx((double*)A->x, A->nzmax, Rgrid.get(), triplet_to_csc.get(), X, Y);
        auto t_fact = high_resolution_clock::now();
        cholmod_l_factorize(A, L, &c);
        auto t_solve = high_resolution_clock::now();
        cholmod_l_solve2(CHOLMOD_A, L, b, NULL, &x, NULL, &Ywork, &Ework, &c);
        auto t_end = high_resolution_clock::now();

        fprintf(f2, "%f %f\n", temp, ((double*)x->x)[0]);

        if (dn_idx < sched_down.size() && temp <= sched_down[dn_idx]) {
            char fn[64]; sprintf(fn, "cool_%d.png", (int)temp);
            save_rgrid_png(fn, Rgrid.get(), X, Y);
            dn_idx++;
        }

        printf("Step %d (C) | T: %.1f | R_tot: %.4f | Fact: %.1fms | Total: %.1fms\n", 
               step, temp, ((double*)x->x)[0], 
               duration<double, std::milli>(t_solve - t_fact).count(),
               duration<double, std::milli>(t_end - s_start).count());
    }
    fclose(f2);

    if (Ywork) cholmod_l_free_dense(&Ywork, &c);
    if (Ework) cholmod_l_free_dense(&Ework, &c);
    if (x)     cholmod_l_free_dense(&x, &c);
    if (b)     cholmod_l_free_dense(&b, &c);
    cholmod_l_free_factor(&L, &c);
    cholmod_l_free_sparse(&A, &c);
    cholmod_l_finish(&c);
    return 0;
}