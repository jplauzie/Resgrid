#include <petscksp.h>
#include <petscmat.h>
#include <petscpc.h>
#include <petscvec.h>
#include <mpi.h>
#include <vector>
#include <memory>
#include <chrono>
#include <algorithm>
#include <set>
#include <cmath>
#include <stdio.h>
#include <iostream>
#include "header.h"
#include "visualization.h"
#include "resgrid_kernels.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

using namespace std::chrono;
using Clock = high_resolution_clock;

static void print_elapsed(PetscMPIInt rank, const char* label, time_point<Clock> t0, time_point<Clock> t1) {
    if (rank == 0) printf("  [init] %-45s %.1f ms\n", label, duration<double, std::milli>(t1 - t0).count());
}

struct GhostExchange {
    struct RecvSeg { int rank; PetscInt local_start; PetscInt count; };
    struct SendSeg { int rank; PetscInt count; double* d_buf; std::vector<int> local_indices; int* d_indices; };
    struct SelfCopy { PetscInt src; PetscInt dst; };
    std::vector<RecvSeg> recvs;
    std::vector<SendSeg> sends;
    std::vector<SelfCopy> self_copies;
    std::vector<MPI_Request> reqs;
    MPI_Comm comm;
    int tag = 42;
};

struct LevelCtx {
    std::vector<double> Rgrid;
    PetscInt Xl, Yl, n;
    PetscInt r_start, r_end;
    PetscInt source_local_idx;
    PetscInt above_halo_start_global;
    PetscInt above_halo_start_local;
    PetscInt below_halo_start_global;
    PetscInt below_halo_start_local;
    double* d_R      = nullptr; 
    double* d_xlocal = nullptr;   
    double* d_y      = nullptr;   
    PetscInt xlocal_total = 0;
    GhostExchange ghost_ex;
    Vec x_ghost_host;
};

struct IntergridCtx {
    LevelCtx* fine;
    LevelCtx* coarse;
    PetscInt fine_seq_offset;
    PetscInt fine_seq_source_slot;
    Vec fine_seq;
    VecScatter fine_scatter;
    PetscInt coarse_seq_offset;
    PetscInt coarse_seq_ic_end;
    Vec coarse_seq;
    VecScatter coarse_scatter;
};

static void buildGhostExchange(LevelCtx& ctx, const std::vector<PetscInt>& ghost_globals, MPI_Comm comm) {
    PetscMPIInt nprocs, rank;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);
    if (nprocs == 1) return; 

    GhostExchange& gx = ctx.ghost_ex;
    gx.comm = comm; gx.tag  = 42;
}

static void ghostExchangeExecute(LevelCtx& ctx) {
    PetscMPIInt nprocs; MPI_Comm_size(ctx.ghost_ex.comm, &nprocs);
    if (nprocs == 1) return; 
    GhostExchange& gx = ctx.ghost_ex;
    for (auto& sc : gx.self_copies) cuda_memcpy_device_to_device(ctx.d_xlocal + sc.dst, ctx.d_xlocal + sc.src, sizeof(double));
    if (gx.reqs.empty()) return;
    for (auto& seg : gx.sends) launch_gather(seg.d_buf, ctx.d_xlocal, seg.d_indices, (int)seg.count);
    MPI_Startall((int)gx.reqs.size(), gx.reqs.data());
    MPI_Waitall((int)gx.reqs.size(), gx.reqs.data(), MPI_STATUSES_IGNORE);
}

static PetscErrorCode levelMatvec(Mat M, Vec x, Vec y) {
    LevelCtx* ctx; MatShellGetContext(M, (void**)&ctx);
    const PetscInt lrows = ctx->r_end - ctx->r_start;
    PetscMPIInt size; MPI_Comm_size(PETSC_COMM_WORLD, &size);

    const PetscScalar* d_xa; 
    VecCUDAGetArrayRead(x, &d_xa);
    PetscScalar* d_ya;
    VecCUDAGetArrayWrite(y, &d_ya);

    if (size == 1) {
        // Fast path: bypass intermediate copy buffer entirely. Write straight to d_ya
        launch_matvec(d_xa, d_ya, ctx->d_R, ctx->r_start, ctx->r_end, ctx->Xl, ctx->Yl,
                      ctx->source_local_idx, ctx->above_halo_start_global, ctx->above_halo_start_local,
                      ctx->below_halo_start_global, ctx->below_halo_start_local);
    } else {
        cuda_memcpy_device_to_device(ctx->d_xlocal, d_xa, (size_t)lrows * sizeof(double));
        ghostExchangeExecute(*ctx);
        launch_matvec(ctx->d_xlocal, ctx->d_y, ctx->d_R, ctx->r_start, ctx->r_end, ctx->Xl, ctx->Yl,
                      ctx->source_local_idx, ctx->above_halo_start_global, ctx->above_halo_start_local,
                      ctx->below_halo_start_global, ctx->below_halo_start_local);
        cuda_memcpy_device_to_device(d_ya, ctx->d_y, (size_t)lrows * sizeof(double));
    }
    VecCUDARestoreArrayRead(x, &d_xa);
    VecCUDARestoreArrayWrite(y, &d_ya);

    return 0;
}

static PetscErrorCode levelGetDiagonal(Mat M, Vec diag) {
    LevelCtx* ctx; MatShellGetContext(M, (void**)&ctx);
    PetscScalar* d; 
    VecCUDAGetArrayWrite(diag, &d);
    launch_diagonal(d, ctx->d_R, ctx->r_start, ctx->r_end, ctx->Xl, ctx->Yl);
    VecCUDARestoreArrayWrite(diag, &d);
    return 0;
}

static PetscErrorCode restrictVec(Mat Rmat, Vec xf, Vec xc) {
    IntergridCtx* ig; MatShellGetContext(Rmat, (void**)&ig);
    PetscMPIInt size; MPI_Comm_size(PETSC_COMM_WORLD, &size);

    const PetscScalar* d_xf; VecCUDAGetArrayRead(xf, &d_xf);
    PetscScalar* d_xc;       VecCUDAGetArrayWrite(xc, &d_xc);

    if (size == 1) {
        launch_restrict(d_xf, d_xc, ig->coarse->r_start, ig->coarse->r_end,
                        ig->fine->Xl, ig->fine->Yl, ig->coarse->Yl, ig->fine_seq_offset);
    } else {
        // [Fallback MPI Scatter Logic]
    }

    VecCUDARestoreArrayRead(xf, &d_xf);
    VecCUDARestoreArrayWrite(xc, &d_xc);
    return 0;
}

static PetscErrorCode prolongVec(Mat Pmat, Vec xc, Vec xf) {
    IntergridCtx* ig; MatShellGetContext(Pmat, (void**)&ig);
    PetscMPIInt size; MPI_Comm_size(PETSC_COMM_WORLD, &size);

    const PetscScalar* d_xc; VecCUDAGetArrayRead(xc, &d_xc);
    PetscScalar* d_xf;       VecCUDAGetArrayWrite(xf, &d_xf);

    if (size == 1) {
        launch_prolong(d_xc, d_xf, ig->fine->r_start, ig->fine->r_end,
                       ig->fine->Yl, ig->coarse->Yl, ig->coarse->Xl,
                       ig->coarse_seq_offset, ig->coarse_seq_ic_end);
    } else {
        // [Fallback MPI Scatter Logic]
    }

    VecCUDARestoreArrayRead(xc, &d_xc);
    VecCUDARestoreArrayWrite(xf, &d_xf);
    return 0;
}

static std::vector<double> coarsenRgrid(const double* Rf, PetscInt Xf, PetscInt Yf, PetscInt Xc, PetscInt Yc) {
    std::vector<double> Rc(Xc * Yc);
    for (PetscInt ic = 0; ic < Xc; ic++)
        for (PetscInt jc = 0; jc < Yc; jc++) {
            PetscInt if0=2*ic, if1=std::min(2*ic+1,Xf-1);
            PetscInt jf0=2*jc, jf1=std::min(2*jc+1,Yf-1);
            int cnt=0; double si=0.0;
            auto acc=[&](double r){ si+=1.0/r; cnt++; };
            acc(Rf[if0*Yf+jf0]);
            if (jf1!=jf0) acc(Rf[if0*Yf+jf1]);
            if (if1!=if0) acc(Rf[if1*Yf+jf0]);
            if (if1!=if0&&jf1!=jf0) acc(Rf[if1*Yf+jf1]);
            Rc[ic*Yc+jc] = (double)cnt/si;
        }
    return Rc;
}

static void getOwnershipRange(PetscInt n, PetscMPIInt rank, PetscMPIInt nprocs, PetscInt& rs, PetscInt& re) {
    PetscInt base=n/nprocs, extra=n%nprocs;
    rs = rank*base + std::min((PetscInt)rank, extra);
    re = rs + base + ((PetscInt)rank < extra ? 1 : 0);
}

static void buildGhostVec(LevelCtx& ctx, MPI_Comm comm) {
    std::vector<PetscInt> g; g.push_back(0);
    ctx.above_halo_start_global = std::max((PetscInt)1, ctx.r_start - ctx.Yl);
    for (PetscInt i = ctx.above_halo_start_global; i < ctx.r_start; i++) if (i != 0) g.push_back(i);
    ctx.below_halo_start_global = ctx.r_end;
    PetscInt below_halo_end = std::min(ctx.n, ctx.r_end + ctx.Yl);
    for (PetscInt i = ctx.below_halo_start_global; i < below_halo_end; i++) g.push_back(i);

    std::sort(g.begin(), g.end());
    g.erase(std::unique(g.begin(), g.end()), g.end());

    PetscInt lr = ctx.r_end - ctx.r_start;
    PetscInt nghosts = (PetscInt)g.size();

    ctx.source_local_idx = -1; ctx.above_halo_start_local = -1; ctx.below_halo_start_local = -1;
    for (PetscInt i = 0; i < nghosts; i++) {
        if (g[i] == 0)                           ctx.source_local_idx       = lr + i;
        if (g[i] == ctx.above_halo_start_global) ctx.above_halo_start_local = lr + i;
        if (g[i] == ctx.below_halo_start_global) ctx.below_halo_start_local = lr + i;
    }

    VecCreateGhost(comm, lr, ctx.n, nghosts, g.data(), &ctx.x_ghost_host);
    ctx.xlocal_total = lr + nghosts;
    cuda_malloc((void**)&ctx.d_R, (size_t)(ctx.Xl * ctx.Yl) * sizeof(double));
    cuda_malloc((void**)&ctx.d_xlocal, (size_t)ctx.xlocal_total * sizeof(double));
    cuda_malloc((void**)&ctx.d_y, (size_t)lr * sizeof(double));
    buildGhostExchange(ctx, g, comm);
}

static IntergridCtx* buildIntergridCtx(LevelCtx* fine, LevelCtx* coarse, MPI_Comm comm) {
    IntergridCtx* ig = new IntergridCtx();
    ig->fine = fine; ig->coarse = coarse;
    
    const PetscInt Yf = fine->Yl;
    PetscInt ic_start = -1;
    for (PetscInt rc = coarse->r_start; rc < coarse->r_end; rc++) {
        if (rc == 0) continue;
        PetscInt ic = (rc - 1) / coarse->Yl;
        if (ic_start < 0) ic_start = ic;
    }
    ig->fine_seq_offset = (ic_start >= 0) ? (2*ic_start)*Yf + 1 : 0;

    PetscInt ic_start_prol = -1, ic_end_prol = -1;
    for (PetscInt rf = fine->r_start; rf < fine->r_end; rf++) {
        if (rf == 0) continue;
        PetscInt ic = ((rf - 1) / Yf) / 2;
        if (ic_start_prol < 0) ic_start_prol = ic;
        ic_end_prol = ic;
    }
    ig->coarse_seq_offset = (ic_start_prol >= 0) ? ic_start_prol * coarse->Yl + 1 : 0;
    ig->coarse_seq_ic_end = std::min(ic_end_prol + 1, coarse->Xl - 1);

    return ig;
}

struct MGLevel { LevelCtx ctx; Mat mat; };
struct MGData  { std::vector<MGLevel> levels; int nlevels=0; };

static void buildLevelMat(MGLevel& lv, MPI_Comm comm) {
    PetscInt lr = lv.ctx.r_end - lv.ctx.r_start;
    MatCreateShell(comm, lr, lr, lv.ctx.n, lv.ctx.n, &lv.ctx, &lv.mat);
    MatSetVecType(lv.mat, VECCUDA); // Ensure Krylov vectors default to VECCUDA 
    MatShellSetOperation(lv.mat, MATOP_MULT, (void(*)(void))levelMatvec);
    MatShellSetOperation(lv.mat, MATOP_GET_DIAGONAL, (void(*)(void))levelGetDiagonal);
}

static MGData buildMGData(const double* fine_Rgrid, PetscInt X, PetscInt Y, PetscMPIInt rank, PetscMPIInt nprocs, MPI_Comm comm, int coarse_threshold=32) {
    MGData mg;
    std::vector<std::pair<PetscInt,PetscInt>> sizes;
    PetscInt cx=X, cy=Y;
    while (true) {
        sizes.push_back({cx,cy});
        if (std::min(cx,cy) <= coarse_threshold) break;
        cx=(cx+1)/2; cy=(cy+1)/2;
    }
    mg.nlevels = (int)sizes.size();
    mg.levels.resize(mg.nlevels);

    if (rank==0) printf("  GMG: %d levels, coarsest %dx%d (%lld DOFs)\n", mg.nlevels,(int)sizes.back().first,(int)sizes.back().second, (long long)(sizes.back().first*sizes.back().second+1));

    for (int lv=0; lv<mg.nlevels; lv++) {
        auto& L  = mg.levels[lv];
        L.ctx.Xl = sizes[lv].first; L.ctx.Yl = sizes[lv].second; L.ctx.n  = L.ctx.Xl * L.ctx.Yl + 1;
        getOwnershipRange(L.ctx.n, rank, nprocs, L.ctx.r_start, L.ctx.r_end);
        buildGhostVec(L.ctx, comm);
        buildLevelMat(L, comm);
    }

    mg.levels[0].ctx.Rgrid.assign(fine_Rgrid, fine_Rgrid + X*Y);
    for (int lv=1; lv<mg.nlevels; lv++) {
        auto& p = mg.levels[lv-1].ctx;
        auto& c = mg.levels[lv].ctx;
        c.Rgrid = coarsenRgrid(p.Rgrid.data(), p.Xl, p.Yl, c.Xl, c.Yl);
    }
    for (int lv=0; lv<mg.nlevels; lv++) {
        auto& ctx = mg.levels[lv].ctx;
        cuda_memcpy_to_device(ctx.d_R, ctx.Rgrid.data(), (size_t)(ctx.Xl*ctx.Yl)*sizeof(double));
    }
    return mg;
}

static void rebuildCoarseLevelsGPU(MGData& mg) {
    for (int lv=1; lv<mg.nlevels; lv++) {
        auto& p = mg.levels[lv-1].ctx;
        auto& c = mg.levels[lv].ctx;
        launch_coarsen(p.d_R, c.d_R, p.Xl, p.Yl, c.Xl, c.Yl);
    }
}

struct PCMGSetupData {
    std::vector<IntergridCtx*> ig_ctxs;
    std::vector<Mat> R_mats, P_mats;
};

static PetscErrorCode setupPCMG(PC pc, MGData& mg, PCMGSetupData& setup, MPI_Comm comm) {
    int nlevels = mg.nlevels;
    PCSetType(pc, PCMG);
    PCMGSetLevels(pc, nlevels, NULL);
    PCMGSetType(pc, PC_MG_MULTIPLICATIVE);
    PCMGSetCycleType(pc, PC_MG_CYCLE_W);

    setup.ig_ctxs.resize(nlevels-1, nullptr);
    setup.R_mats.resize(nlevels-1, nullptr);
    setup.P_mats.resize(nlevels-1, nullptr);

    for (int lv=0; lv<nlevels; lv++) PCMGSetOperators(pc,(nlevels-1)-lv, mg.levels[lv].mat, mg.levels[lv].mat);

    for (int lv=0; lv<nlevels-1; lv++) {
        int pcmg_fine = (nlevels-1)-lv;
        LevelCtx* fine_ctx   = &mg.levels[lv].ctx;
        LevelCtx* coarse_ctx = &mg.levels[lv+1].ctx;
        IntergridCtx* ig = buildIntergridCtx(fine_ctx, coarse_ctx, comm);
        setup.ig_ctxs[lv] = ig;

        PetscInt nf=fine_ctx->n, lf=fine_ctx->r_end - fine_ctx->r_start;
        PetscInt nc=coarse_ctx->n, lc=coarse_ctx->r_end - coarse_ctx->r_start;

        MatCreateShell(comm, lc, lf, nc, nf, ig, &setup.R_mats[lv]);
        MatSetVecType(setup.R_mats[lv], VECCUDA); // Ensure proper PC vec types
        MatShellSetOperation(setup.R_mats[lv], MATOP_MULT, (void(*)(void))restrictVec);
        
        MatCreateShell(comm, lf, lc, nf, nc, ig, &setup.P_mats[lv]);
        MatSetVecType(setup.P_mats[lv], VECCUDA); // Ensure proper PC vec types
        MatShellSetOperation(setup.P_mats[lv], MATOP_MULT, (void(*)(void))prolongVec);
        
        PCMGSetRestriction (pc, pcmg_fine, setup.R_mats[lv]);
        PCMGSetInterpolation(pc, pcmg_fine, setup.P_mats[lv]);
    }

    for (int pcmg_lv=1; pcmg_lv<nlevels; pcmg_lv++) {
        KSP smoother; PCMGGetSmoother(pc, pcmg_lv, &smoother);
        KSPSetType(smoother, KSPCHEBYSHEV);
        PC sub_pc; KSPGetPC(smoother, &sub_pc);
        PCSetType(sub_pc, PCJACOBI);
        KSPSetTolerances(smoother, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT, 2);
        KSPChebyshevEstEigSet(smoother, 0.0, 0.1, 0.0, 1.3);
        KSPChebyshevEstEigSetUseNoisy(smoother, PETSC_TRUE);
    }

    KSP coarse_ksp; PCMGGetCoarseSolve(pc, &coarse_ksp);
    KSPSetType(coarse_ksp, KSPGMRES);
    KSPGMRESSetRestart(coarse_ksp, 200);
    PC cpc; KSPGetPC(coarse_ksp, &cpc);
    PCSetType(cpc, PCJACOBI);
    KSPSetTolerances(coarse_ksp, 1e-12, PETSC_DEFAULT, PETSC_DEFAULT, 200);
    return 0;
}

int main(int argc, char **argv) {
    auto t_prog_start = Clock::now();
    PetscInitialize(&argc, &argv, NULL, NULL);

    PetscMPIInt rank, nprocs;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    MPI_Comm_size(PETSC_COMM_WORLD, &nprocs);

    if (rank==0) printf("=== Initialization (nprocs=%d) ===\n", nprocs);
    if (nprocs > 1) {
        printf("WARNING: Code optimized for -n 1. High latency expected with -n %d.\n", nprocs);
    }

    PetscInt X=100, Y=100;
    PetscOptionsGetInt(NULL,NULL,"-X",&X,NULL);
    PetscOptionsGetInt(NULL,NULL,"-Y",&Y,NULL);
    PetscBool save_pics=PETSC_FALSE;
    PetscOptionsGetBool(NULL,NULL,"-save_pics",&save_pics,NULL);
    PetscInt coarse_threshold=32;
    PetscOptionsGetInt(NULL,NULL,"-mg_coarse_threshold",&coarse_threshold,NULL);

    const PetscInt n = X*Y+1;
    if (rank==0) printf("  Grid: %d x %d  (n = %lld unknowns)\n", (int)X,(int)Y,(long long)n);

    auto Tgrid_up   = std::make_unique<double[]>(X*Y);
    auto Tgrid_down = std::make_unique<double[]>(X*Y);
    auto Rgrid      = std::make_unique<double[]>(X*Y);

    Tgridset(Tgrid_up.get(), X, Y, 345.0, 10.0);
    std::copy(Tgrid_up.get(), Tgrid_up.get()+X*Y, Tgrid_down.get());
    Rgridset(Rgrid.get(), X, Y, 1000.0);
    precomputeHeatingMap(Tgrid_up.get(), X, Y, 20.0);
    precomputeCoolingMap(Tgrid_down.get(), X, Y, 20.0);

    double *d_Tup = nullptr, *d_Tdown = nullptr;
    cuda_malloc((void**)&d_Tup, (X*Y)*sizeof(double));
    cuda_malloc((void**)&d_Tdown, (X*Y)*sizeof(double));
    cuda_memcpy_to_device(d_Tup, Tgrid_up.get(), (X*Y)*sizeof(double));
    cuda_memcpy_to_device(d_Tdown, Tgrid_down.get(), (X*Y)*sizeof(double));

    MGData mg = buildMGData(Rgrid.get(), X, Y, rank, nprocs, PETSC_COMM_WORLD, (int)coarse_threshold);

    Vec b, x_vec;
    MatCreateVecs(mg.levels[0].mat, &x_vec, &b); 
    VecZeroEntries(b);
    if (mg.levels[0].ctx.r_start==0) {
        PetscScalar* b_arr;
        VecGetArray(b, &b_arr);
        b_arr[0] = 1.0;
        VecRestoreArray(b, &b_arr);
    }
    VecAssemblyBegin(b); VecAssemblyEnd(b);

    KSP ksp; KSPCreate(PETSC_COMM_WORLD, &ksp);
    KSPSetOperators(ksp, mg.levels[0].mat, mg.levels[0].mat);
    KSPSetType(ksp, KSPCG);
    KSPSetTolerances(ksp, 1e-8, PETSC_DEFAULT, PETSC_DEFAULT, 5000);

    PC pc; KSPGetPC(ksp, &pc);
    PCMGSetupData mg_setup;
    setupPCMG(pc, mg, mg_setup, PETSC_COMM_WORLD);
    KSPSetFromOptions(ksp);

    double start_temp=300.0, end_temp=375.0;
    int total_steps=(int)(end_temp-start_temp);

    FILE *f1=nullptr, *f2=nullptr;
    if (rank==0) {
        f1 = fopen("results_up.dat","w");
        f2 = fopen("results_down.dat","w");
        printf(">>> STARTING HEATING CYCLE <<<\n");
    }

    for (int step=0; step<=total_steps; step+=5) {
        auto tl = Clock::now();
        double temp = start_temp + (double)step;
        double ins_R = getSemiconductorR(temp);

        launch_update_rgrid(mg.levels[0].ctx.d_R, d_Tup, temp, ins_R, X*Y, true);
        
        auto ta=Clock::now(); 
        rebuildCoarseLevelsGPU(mg); 
        
        for (int lv = 0; lv < mg.nlevels; lv++) {
            MatAssemblyBegin(mg.levels[lv].mat, MAT_FINAL_ASSEMBLY);
            MatAssemblyEnd(mg.levels[lv].mat, MAT_FINAL_ASSEMBLY);
        }
        KSPSetOperators(ksp, mg.levels[0].mat, mg.levels[0].mat);
        
        auto ts=Clock::now(); 
        KSPSolve(ksp,b,x_vec);
        auto te=Clock::now();

        PetscScalar R_tot=0.0;
        if (mg.levels[0].ctx.r_start==0) {
            const PetscScalar* x_arr;
            VecGetArrayRead(x_vec, &x_arr);
            R_tot = x_arr[0];
            VecRestoreArrayRead(x_vec, &x_arr);
        }
        if (rank==0) {
            fprintf(f1,"%f %f\n",temp,PetscRealPart(R_tot));
            PetscInt its; KSPGetIterationNumber(ksp,&its);
            printf("Step %3d (H) | T:%.1f | R:%.4e | Asm:%.1fms | Slv:%.1fms | Tot:%.1fms | It:%d\n",
                   step,temp,PetscRealPart(R_tot), duration<double,std::milli>(ta-tl).count(),
                   duration<double,std::milli>(te-ts).count(), duration<double,std::milli>(te-tl).count(),(int)its);
        }
    }

    if (rank==0) { printf("\n>>> STARTING COOLING CYCLE <<<\n"); }

    for (int step=total_steps; step>=0; step-=5) {
        auto tl=Clock::now();
        double temp = start_temp + (double)step;
        double ins_R = getSemiconductorR(temp);

        launch_update_rgrid(mg.levels[0].ctx.d_R, d_Tdown, temp, ins_R, X*Y, false);
        
        auto ta=Clock::now(); rebuildCoarseLevelsGPU(mg);
        
        for (int lv = 0; lv < mg.nlevels; lv++) {
            MatAssemblyBegin(mg.levels[lv].mat, MAT_FINAL_ASSEMBLY);
            MatAssemblyEnd(mg.levels[lv].mat, MAT_FINAL_ASSEMBLY);
        }
        KSPSetOperators(ksp, mg.levels[0].mat, mg.levels[0].mat);
        
        auto ts=Clock::now(); KSPSolve(ksp,b,x_vec);
        auto te=Clock::now();

        PetscScalar R_tot=0.0;
        if (mg.levels[0].ctx.r_start==0) { 
            const PetscScalar* x_arr;
            VecGetArrayRead(x_vec, &x_arr);
            R_tot = x_arr[0];
            VecRestoreArrayRead(x_vec, &x_arr); 
        }
        if (rank==0) {
            fprintf(f2,"%f %f\n",temp,PetscRealPart(R_tot));
            PetscInt its; KSPGetIterationNumber(ksp,&its);
            printf("Step %3d (C) | T:%.1f | R:%.4e | Asm:%.1fms | Slv:%.1fms | Tot:%.1fms | It:%d\n",
                   step,temp,PetscRealPart(R_tot), duration<double,std::milli>(ta-tl).count(),
                   duration<double,std::milli>(te-ts).count(), duration<double,std::milli>(te-tl).count(),(int)its);
        }
    }

    if (rank==0) { fclose(f1); fclose(f2); }

    cuda_free(d_Tup);
    cuda_free(d_Tdown);
    KSPDestroy(&ksp);
    VecDestroy(&b);
    VecDestroy(&x_vec);
    PetscFinalize();
    return 0;
}