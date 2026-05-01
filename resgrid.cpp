#include <petscksp.h>
#include <petscmat.h>
#include <vector>
#include <memory>
#include <chrono>
#include <algorithm>
#include <set>
#include <stdio.h>
#include <iostream>
#include "Header.h"
#include "visualization.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

using namespace std::chrono;
using Clock = high_resolution_clock;
using std::cout;
using std::endl;

static void print_elapsed(PetscMPIInt rank, const char* label,
                          time_point<Clock> t0, time_point<Clock> t1)
{
    if (rank == 0)
        printf("  [init] %-45s %.1f ms\n", label,
               duration<double, std::milli>(t1 - t0).count());
}

// ============================================================
// Build local CSR arrays for rows [r_start, r_end).
// Column indices are sorted within each row — required so that
// PETSc's internal storage order matches our v_csr order,
// enabling direct pointer writes for per-step value updates.
// i_csr : row pointers,   length (local_rows + 1)
// j_csr : column indices, length nnz_local
// v_csr : values,         length nnz_local (filled per-step)
// ============================================================
static void buildLocalCSR(
    PetscInt X, PetscInt Y,
    PetscInt r_start, PetscInt r_end,
    std::vector<PetscInt>& i_csr,
    std::vector<PetscInt>& j_csr,
    std::vector<double>&   v_csr)
{
    PetscInt local_rows = r_end - r_start;
    i_csr.resize(local_rows + 1);
    j_csr.clear();
    j_csr.reserve(local_rows * 6);

    std::vector<std::vector<PetscInt>> row_cols(local_rows);

    for (PetscInt r = r_start; r < r_end; r++) {
        PetscInt lr = r - r_start;
        auto& cols = row_cols[lr];

        if (r == 0) {
            cols.push_back(0);
            for (PetscInt j = 0; j < Y; j++) cols.push_back(j + 1);
        } else {
            PetscInt grid_idx = r - 1;
            PetscInt i = grid_idx / Y;
            PetscInt j = grid_idx % Y;

            cols.push_back(r);

            if (i == 0) cols.push_back(0);
            else        cols.push_back(r - Y);

            if (j > 0)     cols.push_back(r - 1);
            if (j < Y - 1) cols.push_back(r + 1);
            if (i < X - 1) cols.push_back(r + Y);
        }

        std::sort(cols.begin(), cols.end());
    }

    i_csr[0] = 0;
    for (PetscInt lr = 0; lr < local_rows; lr++) {
        for (PetscInt c : row_cols[lr]) j_csr.push_back(c);
        i_csr[lr + 1] = (PetscInt)j_csr.size();
    }

    v_csr.assign(j_csr.size(), 0.0);
}

// ============================================================
// Fill v_csr with conductance values for the current Rgrid.
// ============================================================
static void fillCSRValues(
    PetscInt X, PetscInt Y,
    PetscInt r_start, PetscInt r_end,
    const double* Rgrid,
    const std::vector<PetscInt>& i_csr,
    const std::vector<PetscInt>& j_csr,
    std::vector<double>& v_csr)
{
    std::fill(v_csr.begin(), v_csr.end(), 0.0);

    for (PetscInt r = r_start; r < r_end; r++) {
        PetscInt lr = r - r_start;
        PetscInt rs = i_csr[lr];

        auto set_val = [&](PetscInt col, double val) {
            for (PetscInt k = rs; k < i_csr[lr + 1]; k++) {
                if (j_csr[k] == col) { v_csr[k] += val; return; }
            }
        };

        if (r == 0) {
            double diag = 0.0;
            for (PetscInt j = 0; j < Y; j++) {
                double gval = 2.0 / Rgrid[j];
                set_val(j + 1, -gval);
                diag += gval;
            }
            set_val(0, diag);
        } else {
            PetscInt grid_idx = r - 1;
            PetscInt i = grid_idx / Y;
            PetscInt j = grid_idx % Y;
            double diag = 0.0;

            if (i == 0) {
                double gval = 2.0 / Rgrid[grid_idx];
                set_val(0, -gval); diag += gval;
            } else {
                double gval = g(Rgrid[grid_idx], Rgrid[(i-1)*Y+j]);
                set_val(r - Y, -gval); diag += gval;
            }

            if (j > 0) {
                double gval = g(Rgrid[grid_idx], Rgrid[i*Y+(j-1)]);
                set_val(r - 1, -gval); diag += gval;
            }

            if (j < Y - 1) {
                double gval = g(Rgrid[grid_idx], Rgrid[i*Y+(j+1)]);
                set_val(r + 1, -gval); diag += gval;
            }

            if (i < X - 1) {
                double gval = g(Rgrid[grid_idx], Rgrid[(i+1)*Y+j]);
                set_val(r + Y, -gval); diag += gval;
            } else {
                diag += 2.0 / Rgrid[grid_idx];
            }

            set_val(r, diag);
        }
    }
}

// ============================================================
// MAIN
// ============================================================
int main(int argc, char **argv)
{
    std::chrono::steady_clock::time_point begin1 = std::chrono::steady_clock::now();
    _putenv_s("MKL_INTERFACE_LAYER", "ILP64");
    _putenv_s("MKL_THREADING_LAYER", "INTEL");
    _putenv_s("MKL_NUM_THREADS", "1");

    auto t_prog_start = Clock::now();
    PetscInitialize(&argc, &argv, NULL, NULL);

    PetscMPIInt rank, nprocs;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    MPI_Comm_size(PETSC_COMM_WORLD, &nprocs);

    if (rank == 0) printf("=== Initialization (nprocs=%d) ===\n", nprocs);

    PetscInt X = 300, Y = 100;
    PetscOptionsGetInt(NULL, NULL, "-X", &X, NULL);
    PetscOptionsGetInt(NULL, NULL, "-Y", &Y, NULL);
    if (rank == 0) printf("  Grid: %d x %d  (n = %d unknowns)\n",
                          (int)X, (int)Y, (int)(X*Y+1));

    const PetscInt n = X * Y + 1;

    // --- Grid arrays (replicated on all ranks) ---
    auto t0 = Clock::now();
    auto Tgrid_up   = std::make_unique<double[]>(X * Y);
    auto Tgrid_down = std::make_unique<double[]>(X * Y);
    auto Rgrid      = std::make_unique<double[]>(X * Y);

    Tgridset(Tgrid_up.get(), X, Y, 345.0, 10.0);
    std::copy(Tgrid_up.get(), Tgrid_up.get() + X*Y, Tgrid_down.get());
    Rgridset(Rgrid.get(), X, Y, 1000.0);
    print_elapsed(rank, "Grid arrays", t0, Clock::now());

    t0 = Clock::now();
    precomputeHeatingMap(Tgrid_up.get(), X, Y, 20.0);
    precomputeCoolingMap(Tgrid_down.get(), X, Y, 20.0);
    print_elapsed(rank, "Heating/cooling cascades", t0, Clock::now());

    std::chrono::steady_clock::time_point end1 = std::chrono::steady_clock::now();
	std::cout << "Time difference1 = " << std::chrono::duration_cast<std::chrono::microseconds>(end1 - begin1).count() << "[micros]" << std::endl;

    // -------------------------------------------------------------------
    // MatCreate + MatSetUp: finalizes the parallel layout and gives us
    // the ownership range WITHOUT allocating value storage.
    // Then MatMPIAIJSetPreallocationCSR allocates exactly what is needed.
    // This avoids the OOM from MatCreateAIJ(... Y+1 ...) on large grids.
    // -------------------------------------------------------------------
    std::chrono::steady_clock::time_point begin2 = std::chrono::steady_clock::now();
    t0 = Clock::now();
    Mat A;
    MatCreate(PETSC_COMM_WORLD, &A);
    MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, n, n);
    MatSetType(A, MATAIJ);
    MatSetUp(A);  // finalizes layout, no value allocation yet

    PetscInt r_start, r_end;
    MatGetOwnershipRange(A, &r_start, &r_end);
    print_elapsed(rank, "MatCreate + MatSetUp + ownership", t0, Clock::now());

    t0 = Clock::now();
    std::vector<PetscInt> i_csr, j_csr;
    std::vector<double>   v_csr;
    buildLocalCSR(X, Y, r_start, r_end, i_csr, j_csr, v_csr);
    print_elapsed(rank, "Local CSR structure built", t0, Clock::now());

    t0 = Clock::now();
    fillCSRValues(X, Y, r_start, r_end, Rgrid.get(), i_csr, j_csr, v_csr);
    MatMPIAIJSetPreallocationCSR(A, i_csr.data(), j_csr.data(), v_csr.data());
    MatSeqAIJSetPreallocationCSR(A, i_csr.data(), j_csr.data(), v_csr.data());
    MatSetOption(A, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_TRUE);
    print_elapsed(rank, "CSR preallocation + initial fill", t0, Clock::now());

    std::chrono::steady_clock::time_point end2 = std::chrono::steady_clock::now();
	std::cout << "Time difference2 = " << std::chrono::duration_cast<std::chrono::microseconds>(end2 - begin2).count() << "[micros]" << std::endl;

    std::chrono::steady_clock::time_point begin3 = std::chrono::steady_clock::now();
    // --- Vectors ---
    t0 = Clock::now();
    Vec b, x_vec;
    MatCreateVecs(A, &x_vec, &b);
    VecZeroEntries(b);
    if (r_start == 0) {
        PetscScalar one = 1.0;
        VecSetValue(b, 0, one, INSERT_VALUES);
    }
    VecAssemblyBegin(b); VecAssemblyEnd(b);
    print_elapsed(rank, "Vectors", t0, Clock::now());

    // --- KSP ---
    t0 = Clock::now();
    KSP ksp;
    KSPCreate(PETSC_COMM_WORLD, &ksp);
    KSPSetOperators(ksp, A, A);
    KSPSetType(ksp, KSPGMRES);
    KSPGMRESSetRestart(ksp, 100);
    PC pc;
    KSPGetPC(ksp, &pc);
    PCSetType(pc, PCBJACOBI);
    KSPSetFromOptions(ksp);
    print_elapsed(rank, "KSP/PC setup", t0, Clock::now());

    if (rank == 0) {
        printf("  Total init: %.1f ms\n\n",
               duration<double, std::milli>(Clock::now() - t_prog_start).count());
    }

    std::chrono::steady_clock::time_point end3 = std::chrono::steady_clock::now();
	std::cout << "Time difference3 = " << std::chrono::duration_cast<std::chrono::microseconds>(end3 - begin3).count() << "[micros]" << std::endl;

    // -------------------------------------------------------------------
    // Per-step reassembly: recompute v_csr then write directly into
    // PETSc's internal value buffer via MatSeqAIJGetArray.
    // No MatZeroEntries, no MatSetValue, no assembly begin/end.
    // Safe because sparsity pattern is fixed and column order matches.
    // -------------------------------------------------------------------
    std::chrono::steady_clock::time_point begin4 = std::chrono::steady_clock::now();
    auto reassemble = [&]() {
    fillCSRValues(X, Y, r_start, r_end, Rgrid.get(), i_csr, j_csr, v_csr);

    // Insert values row by row using precomputed CSR structure.
    // MAT_NEW_NONZERO_ALLOCATION_ERR is set so no reallocation occurs —
    // this is purely a value update into the existing sparsity pattern.
    for (PetscInt lr = 0; lr < r_end - r_start; lr++) {
        PetscInt r = r_start + lr;
        PetscInt ncols = i_csr[lr + 1] - i_csr[lr];
        const PetscInt* cols = j_csr.data() + i_csr[lr];
        const double*   vals = v_csr.data() + i_csr[lr];
        MatSetValues(A, 1, &r, ncols, cols, vals, INSERT_VALUES);
    }
    MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);
    };

    double start_temp = 300.0, end_temp = 375.0;
    int total_steps = (int)(end_temp - start_temp);

    FILE* f1 = nullptr;
    FILE* f2 = nullptr;
    std::vector<int> sched_up, sched_down;
    if (rank == 0) {
        f1 = fopen("results_up.dat", "w");
        std::set<int> h = generateHighlyBiasedTemps(start_temp, end_temp, 345.0, 15, 5.0);
        sched_up.assign(h.begin(), h.end());

        f2 = fopen("results_down.dat", "w");
        std::set<int> c = generateHighlyBiasedTemps(start_temp, end_temp, 335.0, 15, 5.0);
        sched_down.assign(c.begin(), c.end());
        std::reverse(sched_down.begin(), sched_down.end());

        printf(">>> STARTING HEATING CYCLE <<<\n");
    }

    size_t up_idx = 0;

    std::chrono::steady_clock::time_point end4 = std::chrono::steady_clock::now();
	std::cout << "Time difference4 = " << std::chrono::duration_cast<std::chrono::microseconds>(end4 - begin4).count() << "[micros]" << std::endl;


    // --- HEATING LOOP ---
    for (int step = 0; step <= total_steps; step++) {
        std::cout << "Step " << step << " / " << total_steps << std::endl;
        std::chrono::steady_clock::time_point begin5 = std::chrono::steady_clock::now();
        auto s_start = Clock::now();
        double temp = start_temp + (double)step;
        double ins_R = getSemiconductorR(temp);

        for (PetscInt i = 0; i < X * Y; i++)
            Rgrid[i] = (temp >= Tgrid_up[i]) ? 1.0 : ins_R;

        auto t_asm = Clock::now();
        reassemble();
        auto t_slv = Clock::now();
        KSPSetOperators(ksp, A, A);
        KSPSolve(ksp, b, x_vec);
        auto t_end = Clock::now();

        PetscScalar R_tot = 0.0;
        if (r_start == 0) { PetscInt idx = 0; VecGetValues(x_vec, 1, &idx, &R_tot); }
        MPI_Bcast(&R_tot, 1, MPI_DOUBLE, 0, PETSC_COMM_WORLD);

        if (rank == 0) {
            fprintf(f1, "%f %f\n", temp, PetscRealPart(R_tot));
            if (up_idx < sched_up.size() && temp >= sched_up[up_idx]) {
                //char fn[64]; sprintf(fn, "heat_%d.png", sched_up[up_idx]);
               // save_rgrid_png(fn, Rgrid.get(), X, Y);
                //up_idx++;
            }
            PetscInt its; KSPGetIterationNumber(ksp, &its);
            printf("Step %3d (H) | T:%.1f | R:%.4e | Asm:%.1fms | Slv:%.1fms | Tot:%.1fms | It:%d\n",
                step, temp, PetscRealPart(R_tot),
                duration<double,std::milli>(t_slv-t_asm).count(),
                duration<double,std::milli>(t_end-t_slv).count(),
                duration<double,std::milli>(t_end-s_start).count(), (int)its);
        }
        std::chrono::steady_clock::time_point end5 = std::chrono::steady_clock::now();
	    std::cout << "Time difference5 = " << std::chrono::duration_cast<std::chrono::microseconds>(end5 - begin5).count() << "[micros]" << std::endl;

    }

    if (rank == 0) { fclose(f1); printf("\n>>> STARTING COOLING CYCLE <<<\n"); }
    size_t dn_idx = 0;

    // --- COOLING LOOP ---
    for (int step = total_steps; step >= 0; step--) {
        auto s_start = Clock::now();
        double temp = start_temp + (double)step;
        double ins_R = getSemiconductorR(temp);

        for (PetscInt i = 0; i < X * Y; i++)
            Rgrid[i] = (temp < Tgrid_down[i]) ? ins_R : 1.0;

        auto t_asm = Clock::now();
        reassemble();
        auto t_slv = Clock::now();
        KSPSetOperators(ksp, A, A);
        KSPSolve(ksp, b, x_vec);
        auto t_end = Clock::now();

        PetscScalar R_tot = 0.0;
        if (r_start == 0) { PetscInt idx = 0; VecGetValues(x_vec, 1, &idx, &R_tot); }
        MPI_Bcast(&R_tot, 1, MPI_DOUBLE, 0, PETSC_COMM_WORLD);

        if (rank == 0) {
            fprintf(f2, "%f %f\n", temp, PetscRealPart(R_tot));
            if (dn_idx < sched_down.size() && temp <= sched_down[dn_idx]) {
                char fn[64]; sprintf(fn, "cool_%d.png", (int)temp);
                save_rgrid_png(fn, Rgrid.get(), X, Y);
                dn_idx++;
            }
            PetscInt its; KSPGetIterationNumber(ksp, &its);
            printf("Step %3d (C) | T:%.1f | R:%.4e | Asm:%.1fms | Slv:%.1fms | Tot:%.1fms | It:%d\n",
                step, temp, PetscRealPart(R_tot),
                duration<double,std::milli>(t_slv-t_asm).count(),
                duration<double,std::milli>(t_end-t_slv).count(),
                duration<double,std::milli>(t_end-s_start).count(), (int)its);
        }
    }

    if (rank == 0) fclose(f2);

    KSPDestroy(&ksp);
    MatDestroy(&A);
    VecDestroy(&b);
    VecDestroy(&x_vec);
    PetscFinalize();
    return 0;
}