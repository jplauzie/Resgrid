#include <petsc.h>
#include <petscmat.h>
#include <petscksp.h>
#include <vector>
#include <iostream>
#include <fstream>
#include "header.h"

// Explicitly define the mapping to ensure no index conflicts with header.h
// Node 0: Source, Nodes 1 to N: Grains
#define GET_NODE(i, j, Y) ((i) * (Y) + (j) + 1)

struct GridCtx {
    PetscInt X, Y;
    PetscInt r_start, r_end;
    Vec x_seq;
    VecScatter scatter;
    std::vector<double> Rgrid;
};

// --- Physics: Matrix-Vector Multiplication ---
PetscErrorCode matvec(Mat M, Vec x, Vec y) {
    GridCtx* ctx;
    MatShellGetContext(M, (void**)&ctx);
    
    VecScatterBegin(ctx->scatter, x, ctx->x_seq, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(ctx->scatter, x, ctx->x_seq, INSERT_VALUES, SCATTER_FORWARD);

    const PetscScalar* x_arr;
    PetscScalar* y_arr;
    VecGetArrayRead(ctx->x_seq, &x_arr);
    VecGetArray(y, &y_arr);

    for (PetscInt r = ctx->r_start; r < ctx->r_end; r++) {
        PetscInt lr = r - ctx->r_start;
        double result = 0.0, diag_val = 0.0;

        if (r == 0) { // Source Node connects to the first row of grains
            for (PetscInt j = 0; j < ctx->Y; j++) {
                double G = 2.0 / ctx->Rgrid[j];
                diag_val += G;
                result -= G * x_arr[GET_NODE(0, j, ctx->Y)];
            }
        } else {
            PetscInt idx = r - 1;
            PetscInt i = idx / ctx->Y;
            PetscInt j = idx % ctx->Y;
            
            // Connection to Source (Row 0) or Grain above
            if (i == 0) {
                double G = 2.0 / ctx->Rgrid[idx];
                diag_val += G;
                result -= G * x_arr[0];
            } else {
                double G = g(ctx->Rgrid[idx], ctx->Rgrid[(i - 1) * ctx->Y + j]);
                diag_val += G;
                result -= G * x_arr[GET_NODE(i - 1, j, ctx->Y)];
            }
            // Connection to Grain below or Ground (Last Row)
            if (i < ctx->X - 1) {
                double G = g(ctx->Rgrid[idx], ctx->Rgrid[(i + 1) * ctx->Y + j]);
                diag_val += G;
                result -= G * x_arr[GET_NODE(i + 1, j, ctx->Y)];
            } else { 
                diag_val += 2.0 / ctx->Rgrid[idx]; // Ground connection
            }
            // Horizontal connections
            if (j > 0) {
                double G = g(ctx->Rgrid[idx], ctx->Rgrid[i * ctx->Y + (j - 1)]);
                diag_val += G;
                result -= G * x_arr[GET_NODE(i, j - 1, ctx->Y)];
            }
            if (j < ctx->Y - 1) {
                double G = g(ctx->Rgrid[idx], ctx->Rgrid[i * ctx->Y + (j + 1)]);
                diag_val += G;
                result -= G * x_arr[GET_NODE(i, j + 1, ctx->Y)];
            }
        }
        y_arr[lr] = result + (diag_val * x_arr[r]);
    }
    VecRestoreArrayRead(ctx->x_seq, &x_arr);
    VecRestoreArray(y, &y_arr);
    return 0;
}

PetscErrorCode GetDiagonal(Mat M, Vec diag) {
    GridCtx* ctx;
    MatShellGetContext(M, (void**)&ctx);
    PetscScalar* d_arr;
    VecGetArray(diag, &d_arr);
    for (PetscInt r = ctx->r_start; r < ctx->r_end; r++) {
        double dv = 0.0;
        if (r == 0) { for (int j=0; j<ctx->Y; j++) dv += 2.0/ctx->Rgrid[j]; }
        else {
            int idx = r-1, i = idx/ctx->Y, j = idx%ctx->Y;
            dv += (i==0) ? 2.0/ctx->Rgrid[idx] : g(ctx->Rgrid[idx], ctx->Rgrid[(i-1)*ctx->Y+j]);
            dv += (i<ctx->X-1) ? g(ctx->Rgrid[idx], ctx->Rgrid[(i+1)*ctx->Y+j]) : 2.0/ctx->Rgrid[idx];
            if (j>0) dv += g(ctx->Rgrid[idx], ctx->Rgrid[i * ctx->Y + (j - 1)]);
            if (j<ctx->Y-1) dv += g(ctx->Rgrid[idx], ctx->Rgrid[i * ctx->Y + (j + 1)]);
        }
        d_arr[r - ctx->r_start] = dv;
    }
    VecRestoreArray(diag, &d_arr);
    return 0;
}

// This is your matrix-free preconditioner logic
PetscErrorCode MyPreconditionerApply(PC pc, Vec r, Vec z) {
    GridCtx* ctx;
    PCShellGetContext(pc, (void**)&ctx);

    const PetscScalar* r_arr;
    PetscScalar* z_arr;
    VecGetArrayRead(r, &r_arr);
    VecGetArray(z, &z_arr);

    // Simple Jacobi-style smoothing: 
    // z = r * (approximate diagonal inverse)
    // Since we don't have the matrix, we use your Rgrid to approximate conductance
    for (PetscInt i = 0; i < (ctx->r_end - ctx->r_start); i++) {
        double diag_approx = 1.0; 
        if (i < ctx->Rgrid.size()) diag_approx = 1.0 / (ctx->Rgrid[i] + 1e-12);
        z_arr[i] = r_arr[i] * diag_approx;
    }

    VecRestoreArrayRead(r, &r_arr);
    VecRestoreArray(z, &z_arr);
    return 0;
}

int main(int argc, char **argv) {
    PetscInitialize(&argc, &argv, NULL, NULL);
    int rank;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

    PetscInt X = 100, Y = 100;
    PetscOptionsGetInt(NULL, NULL, "-X", &X, NULL);
    PetscOptionsGetInt(NULL, NULL, "-Y", &Y, NULL);
    PetscInt N = X * Y + 1;

    std::vector<double> Tgrid_heat(X * Y), Tgrid_cool(X * Y);
    Tgridset(Tgrid_heat.data(), X, Y, 350.0, 10.0);
    Tgridset(Tgrid_cool.data(), X, Y, 350.0, 10.0);
    precomputeHeatingMap(Tgrid_heat.data(), X, Y, 100.0);
    precomputeCoolingMap(Tgrid_cool.data(), X, Y, 100.0);

    GridCtx ctx;
    ctx.X = X; ctx.Y = Y;
    ctx.Rgrid.assign(X * Y, 1000.0);

    Vec x, b;
    VecCreate(PETSC_COMM_WORLD, &x);
    VecSetSizes(x, PETSC_DECIDE, N);
    VecSetFromOptions(x);
    VecDuplicate(x, &b);
    VecGetOwnershipRange(x, &ctx.r_start, &ctx.r_end);
    VecScatterCreateToAll(x, &ctx.scatter, &ctx.x_seq);

    Mat A;
    MatCreateShell(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, N, N, &ctx, &A);
    MatShellSetOperation(A, MATOP_MULT, (void(*)(void))matvec);
    MatShellSetOperation(A, MATOP_GET_DIAGONAL, (void(*)(void))GetDiagonal);

    KSP ksp;
    KSPCreate(PETSC_COMM_WORLD, &ksp);
    KSPSetOperators(ksp, A, A);
    // --- ADD THIS BLOCK ---
    PC pc;
    KSPGetPC(ksp, &pc);
    PCSetType(pc, PCSHELL);
    PCShellSetContext(pc, &ctx); // Pass your grid context so the PC can access Rgrid
    PCShellSetApply(pc, MyPreconditionerApply); // Register the function above
    // ----------------------
    KSPSetTolerances(ksp, 1e-5, PETSC_DEFAULT, PETSC_DEFAULT, 10000);
    KSPSetFromOptions(ksp);

    

    std::ofstream resFile;
    if (rank == 0) resFile.open("resistance.dat");

    auto solve_loop = [&](int T_start, int T_end, int step, const std::vector<double>& T_map, const char* phase) {
        PetscPrintf(PETSC_COMM_WORLD, ">>> Starting %s Phase (%dK to %dK)\n", phase, T_start, T_end);
        for (int T = T_start; (step > 0) ? (T <= T_end) : (T >= T_end); T += step) {
            
            for (int i = 0; i < X * Y; i++) {
                ctx.Rgrid[i] = (T >= T_map[i]) ? 1.0 : getSemiconductorR((double)T);
            }

            VecSet(b, 0.0);
            if (ctx.r_start == 0) VecSetValue(b, 0, 1.0, INSERT_VALUES);
            VecAssemblyBegin(b); VecAssemblyEnd(b);

            KSPSolve(ksp, b, x);
            
            KSPConvergedReason reason;
            KSPGetConvergedReason(ksp, &reason);

            if (rank == 0) {
                PetscScalar v_source;
                PetscInt row = 0;
                VecGetValues(x, 1, &row, &v_source);
                resFile << T << " " << v_source << std::endl;
                std::cout << "  T: " << T << " K | V_source: " << v_source << " | Reason: " << (int)reason << std::endl;
            }
        }
    };

    solve_loop(300, 375, 1, Tgrid_heat, "Heating");
    solve_loop(375, 300, -1, Tgrid_cool, "Cooling");

    if (rank == 0) resFile.close();
    VecScatterDestroy(&ctx.scatter); VecDestroy(&ctx.x_seq);
    VecDestroy(&x); VecDestroy(&b); MatDestroy(&A); KSPDestroy(&ksp);
    PetscFinalize();
    return 0;
}