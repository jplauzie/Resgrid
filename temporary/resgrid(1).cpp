#include <petscksp.h>
#include <petscmat.h>
#include <petscpc.h>
#include <petscvec.h>
#include <petscveccuda.h>
#include "resgrid_kernels.h"
#include <vector>
#include <memory>
#include <chrono>
#include <algorithm>
#include <set>
#include <cmath>
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

static inline double gmean(double R1, double R2)
{
    return 2.0 / (R1 + R2);
}

// ============================================================
// Per-level context for matrix-free matvec.
// ============================================================
struct LevelCtx {
    std::vector<double> Rgrid;
    double* d_Rgrid = nullptr;   // device copy of Rgrid
    PetscInt Xl, Yl, n;
    PetscInt r_start, r_end;
    Vec      x_ghost;
    
    // NEW: Fast mapping offsets
    PetscInt source_local_idx;
    PetscInt above_halo_start_global;
    PetscInt above_halo_start_local;
    PetscInt below_halo_start_global;
    PetscInt below_halo_start_local;
};

// ============================================================
// Context for mapping between fine and coarse grids
// ============================================================
struct IntergridCtx {
    LevelCtx* fine;
    LevelCtx* coarse;

    PetscInt fine_seq_offset;
    PetscInt fine_seq_source_slot;
    Vec fine_seq;
    VecScatter fine_scatter;

    PetscInt coarse_seq_offset;
    PetscInt coarse_seq_ic_end;   // NEW: last ic row in coarse_seq (for bounds check)
    Vec coarse_seq;
    VecScatter coarse_scatter;
};

static std::vector<PetscInt> buildGhostIndices(
    PetscInt n, PetscInt r_start, PetscInt r_end)
{
    std::vector<PetscInt> g;
    g.reserve(3);
    if (r_start > 1) g.push_back(r_start - 1);
    if (r_end   < n) g.push_back(r_end);
    if (r_start > 0) g.push_back(0);
    std::sort(g.begin(), g.end());
    g.erase(std::unique(g.begin(), g.end()), g.end());
    return g;
}

static inline PetscInt globalToLocal(const LevelCtx* ctx, PetscInt global) {
    if (global >= ctx->r_start && global < ctx->r_end)
        return global - ctx->r_start; // Owned
    if (global == 0) 
        return ctx->source_local_idx; // Source node
    if (global < ctx->r_start) 
        return ctx->above_halo_start_local + (global - ctx->above_halo_start_global);
    return ctx->below_halo_start_local + (global - ctx->below_halo_start_global);
}
// ============================================================
// Matrix-free matvec: y = A*x  (GPU version)
//
// Strategy:
//   1. Copy owned values of x into the ghost vec (CPU, needed for MPI scatter)
//   2. MPI ghost update (CPU boundary exchange)
//   3. Get raw device pointer to the ghost vec local form
//   4. Get raw device pointer to y
//   5. Launch CUDA kernel — all arithmetic stays on GPU
// ============================================================
static PetscErrorCode levelMatvec(Mat M, Vec x, Vec y)
{
    LevelCtx* ctx;
    MatShellGetContext(M, (void**)&ctx);

    const PetscInt r_start = ctx->r_start;
    const PetscInt r_end   = ctx->r_end;
    const PetscInt lrows   = r_end - r_start;

    // --- Step 1: fill owned part of ghost vec from x ---
    // Must go through CPU because VecGhostUpdateBegin/End uses MPI.
    {
        const PetscScalar* xa;
        VecGetArrayRead(x, &xa);
        Vec xl; VecGhostGetLocalForm(ctx->x_ghost, &xl);
        PetscScalar* ga; VecGetArray(xl, &ga);
        for (PetscInt i = 0; i < lrows; i++) ga[i] = xa[i];
        VecRestoreArray(xl, &ga);
        VecGhostRestoreLocalForm(ctx->x_ghost, &xl);
        VecRestoreArrayRead(x, &xa);
    }

    // --- Step 2: MPI ghost exchange ---
    VecGhostUpdateBegin(ctx->x_ghost, INSERT_VALUES, SCATTER_FORWARD);
    VecGhostUpdateEnd  (ctx->x_ghost, INSERT_VALUES, SCATTER_FORWARD);

    // --- Step 3 & 4: get device pointers and launch kernel ---
    Vec xl; VecGhostGetLocalForm(ctx->x_ghost, &xl);

    const PetscScalar* d_xlocal;
    VecCUDAGetArrayRead(xl, &d_xlocal);

    PetscScalar* d_y;
    VecCUDAGetArray(y, &d_y);

    launch_matvec(
        d_xlocal, d_y, ctx->d_Rgrid,
        r_start, r_end, ctx->Xl, ctx->Yl,
        ctx->source_local_idx,
        ctx->above_halo_start_global, ctx->above_halo_start_local,
        ctx->below_halo_start_global, ctx->below_halo_start_local);

    VecCUDARestoreArrayRead(xl, &d_xlocal);
    VecCUDARestoreArray(y, &d_y);
    VecGhostRestoreLocalForm(ctx->x_ghost, &xl);
    return 0;
}

// ============================================================
// Diagonal extraction for Jacobi sub-PC. (GPU version)
// ============================================================
static PetscErrorCode levelGetDiagonal(Mat M, Vec diag)
{
    LevelCtx* ctx;
    MatShellGetContext(M, (void**)&ctx);

    PetscScalar* d_diag;
    VecCUDAGetArray(diag, &d_diag);

    launch_diagonal(d_diag, ctx->d_Rgrid,
                    ctx->r_start, ctx->r_end,
                    ctx->Xl, ctx->Yl);

    VecCUDARestoreArray(diag, &d_diag);
    return 0;
}

// ============================================================
// Coarsen Rgrid 2x — CPU version (used for initial build only)
// ============================================================
static std::vector<double> coarsenRgrid(
    const double* Rf, PetscInt Xf, PetscInt Yf,
    PetscInt Xc, PetscInt Yc)
{
    std::vector<double> Rc(Xc * Yc);
    for (PetscInt ic = 0; ic < Xc; ic++) {
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
    }
    return Rc;
}

static void getOwnershipRange(PetscInt n, PetscMPIInt rank, PetscMPIInt nprocs,
                               PetscInt& rs, PetscInt& re)
{
    PetscInt base=n/nprocs, extra=n%nprocs;
    rs = rank*base + std::min((PetscInt)rank, extra);
    re = rs + base + ((PetscInt)rank < extra ? 1 : 0);
}

static void buildGhostVec(LevelCtx& ctx, MPI_Comm comm) {
    std::vector<PetscInt> g;
    // 1. Every rank can ghost the source node (index 0)
    g.push_back(0);

    // 2. Halo Above: Need Yl nodes to satisfy (r - Yl)
    ctx.above_halo_start_global = std::max((PetscInt)1, ctx.r_start - ctx.Yl);
    for (PetscInt i = ctx.above_halo_start_global; i < ctx.r_start; i++) {
        if (i != 0) g.push_back(i);
    }

    // 3. Halo Below: Need Yl nodes to satisfy (r + Yl)
    ctx.below_halo_start_global = ctx.r_end;
    PetscInt below_halo_end = std::min(ctx.n, ctx.r_end + ctx.Yl);
    for (PetscInt i = ctx.below_halo_start_global; i < below_halo_end; i++) {
        g.push_back(i);
    }

    std::sort(g.begin(), g.end());
    g.erase(std::unique(g.begin(), g.end()), g.end());
    
    PetscInt lr = ctx.r_end - ctx.r_start;
    // Set up O(1) mapping offsets
    ctx.source_local_idx = -1;
    ctx.above_halo_start_local = -1;
    ctx.below_halo_start_local = -1;

    for (size_t i = 0; i < g.size(); i++) {
        if (g[i] == 0) ctx.source_local_idx = lr + i;
        if (g[i] == ctx.above_halo_start_global) ctx.above_halo_start_local = lr + i;
        if (g[i] == ctx.below_halo_start_global) ctx.below_halo_start_local = lr + i;
    }

    VecCreateGhost(comm, lr, ctx.n, (PetscInt)g.size(), g.data(), &ctx.x_ghost);
}

// ============================================================
// Intergrid context.
// Both scatter operations now use ISCreateStride/General to 
// prevent global broadcasts.
// ============================================================
static IntergridCtx* buildIntergridCtx(LevelCtx* fine, LevelCtx* coarse,
                                        MPI_Comm comm)
{
    IntergridCtx* ig = new IntergridCtx();
    ig->fine   = fine;
    ig->coarse = coarse;

    const PetscInt Yf = fine->Yl;
    const PetscInt Yc = coarse->Yl;

    // --- FINE SCATTER (for restriction) ---
    // Gather the band of fine rows covering owned coarse rows.
    PetscInt ic_start = -1, ic_end = -1;
    for (PetscInt rc = coarse->r_start; rc < coarse->r_end; rc++) {
        if (rc == 0) continue;
        PetscInt ic = (rc - 1) / Yc;
        if (ic_start < 0) ic_start = ic;
        ic_end = ic;
    }

    std::vector<PetscInt> indices;
    indices.push_back(0);  // source node always needed

    if (ic_start >= 0) {
        PetscInt fi_start = 2 * ic_start;
        PetscInt fi_end   = std::min(2 * ic_end + 1, fine->Xl - 1);
        for (PetscInt fi = fi_start; fi <= fi_end; fi++)
            for (PetscInt fj = 0; fj < Yf; fj++)
                indices.push_back(fi * Yf + fj + 1);
    }

    ig->fine_seq_offset      = (ic_start >= 0) ? (2*ic_start)*Yf + 1 : 0;
    ig->fine_seq_source_slot = 0;
    PetscInt n_seq = (PetscInt)indices.size();

    {
        Vec fine_mpi;
        VecCreateMPI(comm, fine->r_end - fine->r_start, fine->n, &fine_mpi);
        IS from_is, to_is;
        ISCreateGeneral(PETSC_COMM_SELF, n_seq, indices.data(),
                        PETSC_COPY_VALUES, &from_is);
        ISCreateStride(PETSC_COMM_SELF, n_seq, 0, 1, &to_is);
        VecCreateSeq(PETSC_COMM_SELF, n_seq, &ig->fine_seq);
        VecSetType(ig->fine_seq, VECCUDA);
        VecScatterCreate(fine_mpi, from_is, ig->fine_seq, to_is,
                         &ig->fine_scatter);
        ISDestroy(&from_is);
        ISDestroy(&to_is);
        VecDestroy(&fine_mpi);
    }

    // --- COARSE SCATTER (for prolongation) ---
    // Find which coarse ic rows this rank needs for its owned fine rows.
    // Bilinear interpolation means a fine node at ic row needs coarse
    // rows ic/2 AND ic/2+1, so we extend the upper end by 1.
    PetscInt ic_start_prol = -1, ic_end_prol = -1;
    for (PetscInt rf = fine->r_start; rf < fine->r_end; rf++) {
        if (rf == 0) continue;
        PetscInt gf  = rf - 1;
        PetscInt iff = gf / Yf;
        PetscInt ic  = iff / 2;
        if (ic_start_prol < 0) ic_start_prol = ic;
        ic_end_prol = ic;
    }

    // Extend by 1 to cover the ic+1 neighbor in bilinear weights.
    // Clamp to the actual coarse grid boundary.
    if (ic_end_prol >= 0)
        ic_end_prol = std::min(ic_end_prol + 1, coarse->Xl - 1);

    ig->coarse_seq_offset = (ic_start_prol >= 0) ? ic_start_prol * Yc + 1 : 0;
    ig->coarse_seq_ic_end = ic_end_prol;  // store for bounds check in prolongVec

    std::vector<PetscInt> c_indices;
    c_indices.push_back(0);  // source node always needed
    if (ic_start_prol >= 0) {
        for (PetscInt ic = ic_start_prol; ic <= ic_end_prol; ic++)
            for (PetscInt jc = 0; jc < Yc; jc++)
                c_indices.push_back(ic * Yc + jc + 1);
    }

    PetscInt n_seq_c = (PetscInt)c_indices.size();

    {
        Vec coarse_mpi;
        VecCreateMPI(comm, coarse->r_end - coarse->r_start,
                     coarse->n, &coarse_mpi);
        IS from_is_c, to_is_c;
        ISCreateGeneral(PETSC_COMM_SELF, n_seq_c, c_indices.data(),
                        PETSC_COPY_VALUES, &from_is_c);
        ISCreateStride(PETSC_COMM_SELF, n_seq_c, 0, 1, &to_is_c);
        VecCreateSeq(PETSC_COMM_SELF, n_seq_c, &ig->coarse_seq);
        VecSetType(ig->coarse_seq, VECCUDA);
        VecScatterCreate(coarse_mpi, from_is_c, ig->coarse_seq, to_is_c,
                         &ig->coarse_scatter);
        ISDestroy(&from_is_c);
        ISDestroy(&to_is_c);
        VecDestroy(&coarse_mpi);
    }

    return ig;
}

// ============================================================
// Restriction: xc = R * xf  (GPU version)
// ============================================================
static PetscErrorCode restrictVec(Mat Rmat, Vec xf, Vec xc)
{
    IntergridCtx* ig;
    MatShellGetContext(Rmat, (void**)&ig);
    LevelCtx* fine   = ig->fine;
    LevelCtx* coarse = ig->coarse;

    VecScatterBegin(ig->fine_scatter, xf, ig->fine_seq, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd  (ig->fine_scatter, xf, ig->fine_seq, INSERT_VALUES, SCATTER_FORWARD);

    const PetscScalar* d_fseq;
    VecCUDAGetArrayRead(ig->fine_seq, &d_fseq);

    PetscScalar* d_xca;
    VecCUDAGetArray(xc, &d_xca);

    launch_restrict(
        d_fseq, d_xca,
        coarse->r_start, coarse->r_end,
        fine->Xl, fine->Yl, coarse->Yl,
        ig->fine_seq_offset);

    VecCUDARestoreArrayRead(ig->fine_seq, &d_fseq);
    VecCUDARestoreArray(xc, &d_xca);
    return 0;
}

// ============================================================
// Prolongation: xf = P * xc  (GPU version)
// ============================================================
static PetscErrorCode prolongVec(Mat Pmat, Vec xc, Vec xf)
{
    IntergridCtx* ig;
    MatShellGetContext(Pmat, (void**)&ig);
    LevelCtx* fine   = ig->fine;
    LevelCtx* coarse = ig->coarse;

    VecScatterBegin(ig->coarse_scatter, xc, ig->coarse_seq,
                    INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd  (ig->coarse_scatter, xc, ig->coarse_seq,
                    INSERT_VALUES, SCATTER_FORWARD);

    const PetscScalar* d_cseq;
    VecCUDAGetArrayRead(ig->coarse_seq, &d_cseq);

    PetscScalar* d_xfa;
    VecCUDAGetArray(xf, &d_xfa);

    launch_prolong(
        d_cseq, d_xfa,
        fine->r_start, fine->r_end,
        fine->Yl, coarse->Yl, coarse->Xl,
        ig->coarse_seq_offset, ig->coarse_seq_ic_end);

    VecCUDARestoreArrayRead(ig->coarse_seq, &d_cseq);
    VecCUDARestoreArray(xf, &d_xfa);
    return 0;
}

// ============================================================
// MGLevel and MGData
// ============================================================
struct MGLevel { LevelCtx ctx; Mat mat; };

struct MGData {
    std::vector<MGLevel> levels;   // [0]=finest, [nlevels-1]=coarsest
    int nlevels = 0;
};

static void buildLevelMat(MGLevel& lv, MPI_Comm comm)
{
    PetscInt lr = lv.ctx.r_end - lv.ctx.r_start;
    MatCreateShell(comm, lr, lr, lv.ctx.n, lv.ctx.n, &lv.ctx, &lv.mat);
    MatShellSetOperation(lv.mat, MATOP_MULT,
                         (void(*)(void))levelMatvec);
    MatShellSetOperation(lv.mat, MATOP_GET_DIAGONAL,
                         (void(*)(void))levelGetDiagonal);
}

static MGData buildMGData(const double* fine_Rgrid,
                           PetscInt X, PetscInt Y,
                           PetscMPIInt rank, PetscMPIInt nprocs,
                           MPI_Comm comm, int coarse_threshold=32)
{
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

    if (rank == 0)
        printf("  GMG: %d levels, coarsest %dx%d (%lld DOFs)\n",
               mg.nlevels,
               (int)sizes.back().first, (int)sizes.back().second,
               (long long)(sizes.back().first*sizes.back().second+1));

    for (int lv = 0; lv < mg.nlevels; lv++) {
        auto& L  = mg.levels[lv];
        L.ctx.Xl = sizes[lv].first;
        L.ctx.Yl = sizes[lv].second;
        L.ctx.n  = L.ctx.Xl * L.ctx.Yl + 1;
        getOwnershipRange(L.ctx.n, rank, nprocs,
                          L.ctx.r_start, L.ctx.r_end);
        buildGhostVec(L.ctx, comm);
        VecSetType(L.ctx.x_ghost, VECCUDA);
        L.ctx.Rgrid.resize(L.ctx.Xl * L.ctx.Yl, 1.0);
        // Allocate device Rgrid
        cuda_malloc((void**)&L.ctx.d_Rgrid,
                    L.ctx.Xl * L.ctx.Yl * sizeof(double));
        buildLevelMat(L, comm);
    }

    // Initial fill from fine_Rgrid.
    mg.levels[0].ctx.Rgrid.assign(fine_Rgrid, fine_Rgrid + X*Y);
    for (int lv = 1; lv < mg.nlevels; lv++) {
        auto& p = mg.levels[lv-1].ctx;
        auto& c = mg.levels[lv].ctx;
        c.Rgrid = coarsenRgrid(p.Rgrid.data(), p.Xl, p.Yl, c.Xl, c.Yl);
    }
    // Upload all levels to device
    for (int lv = 0; lv < mg.nlevels; lv++) {
        auto& L = mg.levels[lv];
        cuda_memcpy_to_device(L.ctx.d_Rgrid, L.ctx.Rgrid.data(),
                              L.ctx.Xl * L.ctx.Yl * sizeof(double));
    }

    return mg;
}

static void rebuildCoarseLevels(MGData& mg, const double* fine_Rgrid)
{
    // Fine level: upload from CPU (updated by caller each temperature step)
    auto& L0 = mg.levels[0].ctx;
    L0.Rgrid.assign(fine_Rgrid, fine_Rgrid + L0.Xl * L0.Yl);
    cuda_memcpy_to_device(L0.d_Rgrid, L0.Rgrid.data(),
                          L0.Xl * L0.Yl * sizeof(double));

    // Coarser levels: coarsen on GPU, chain level by level
    for (int lv = 1; lv < mg.nlevels; lv++) {
        auto& p = mg.levels[lv-1].ctx;
        auto& c = mg.levels[lv].ctx;
        launch_coarsen(p.d_Rgrid, c.d_Rgrid, p.Xl, p.Yl, c.Xl, c.Yl);
    }
}

// ============================================================
// PCMGSetupData: owns intergrid contexts and MatShells.
// ============================================================
struct PCMGSetupData {
    std::vector<IntergridCtx*> ig_ctxs;
    std::vector<Mat>           R_mats;
    std::vector<Mat>           P_mats;
};

static PetscErrorCode setupPCMG(PC pc, MGData& mg,
                                  PCMGSetupData& setup, MPI_Comm comm)
{
    auto PCMGstart = Clock::now();
    int nlevels = mg.nlevels;

    PCSetType(pc, PCMG);
    PCMGSetLevels(pc, nlevels, NULL);
    PCMGSetType(pc, PC_MG_MULTIPLICATIVE);
    PCMGSetCycleType(pc, PC_MG_CYCLE_W);

    setup.ig_ctxs.resize(nlevels - 1, nullptr);
    setup.R_mats.resize(nlevels - 1, nullptr);
    setup.P_mats.resize(nlevels - 1, nullptr);
    print_elapsed(0, "first", PCMGstart, Clock::now());

    for (int lv = 0; lv < nlevels; lv++) {
        int pcmg_lv = (nlevels-1) - lv;
        PCMGSetOperators(pc, pcmg_lv,
                         mg.levels[lv].mat, mg.levels[lv].mat);
    }
    print_elapsed(0, "second", PCMGstart, Clock::now());

    for (int lv = 0; lv < nlevels-1; lv++) {
        int pcmg_fine = (nlevels-1) - lv; 

        LevelCtx* fine_ctx   = &mg.levels[lv].ctx;
        LevelCtx* coarse_ctx = &mg.levels[lv+1].ctx;

        IntergridCtx* ig = buildIntergridCtx(fine_ctx, coarse_ctx, comm);
        setup.ig_ctxs[lv] = ig;

        PetscInt nf=fine_ctx->n,   lf=fine_ctx->r_end   - fine_ctx->r_start;
        PetscInt nc=coarse_ctx->n, lc=coarse_ctx->r_end  - coarse_ctx->r_start;

        MatCreateShell(comm, lc, lf, nc, nf, ig, &setup.R_mats[lv]);
        MatShellSetOperation(setup.R_mats[lv], MATOP_MULT,
                             (void(*)(void))restrictVec);

        MatCreateShell(comm, lf, lc, nf, nc, ig, &setup.P_mats[lv]);
        MatShellSetOperation(setup.P_mats[lv], MATOP_MULT,
                             (void(*)(void))prolongVec);

        PCMGSetRestriction (pc, pcmg_fine, setup.R_mats[lv]);
        PCMGSetInterpolation(pc, pcmg_fine, setup.P_mats[lv]);
    }
    print_elapsed(0, "third", PCMGstart, Clock::now());

    // Keeping your previous smoother setups but allowing command line to override
    for (int pcmg_lv = 1; pcmg_lv < nlevels; pcmg_lv++) {
        KSP smoother;
        PCMGGetSmoother(pc, pcmg_lv, &smoother);
        KSPSetType(smoother, KSPCHEBYSHEV);
        PC sub_pc; KSPGetPC(smoother, &sub_pc);
        PCSetType(sub_pc, PCJACOBI);
        KSPSetTolerances(smoother, PETSC_DEFAULT, PETSC_DEFAULT,
                         PETSC_DEFAULT, 2);
        KSPChebyshevEstEigSet(smoother, 0.0, 0.1, 0.0, 1.3);
        KSPChebyshevEstEigSetUseNoisy(smoother, PETSC_TRUE);
    }
    print_elapsed(0, "fourth", PCMGstart, Clock::now());

    // Coarsest level
    {
        KSP coarse_ksp;
        PCMGGetCoarseSolve(pc, &coarse_ksp);
        KSPSetType(coarse_ksp, KSPGMRES);
        KSPGMRESSetRestart(coarse_ksp, 200);
        PC cpc; KSPGetPC(coarse_ksp, &cpc);
        PCSetType(cpc, PCJACOBI);
        KSPSetTolerances(coarse_ksp, 1e-12, PETSC_DEFAULT,
                         PETSC_DEFAULT, 200);
    }
    print_elapsed(0, "fifth", PCMGstart, Clock::now());

    return 0;
}

// ============================================================
// MAIN
// ============================================================
int main(int argc, char **argv)
{
    auto t_prog_start = Clock::now();

#ifdef _WIN32
    _putenv_s("MKL_INTERFACE_LAYER", "LP64");
    _putenv_s("MKL_THREADING_LAYER", "INTEL");
    _putenv_s("MKL_NUM_THREADS", "1");
#endif

    PetscInitialize(&argc, &argv, NULL, NULL);

    PetscMPIInt rank, nprocs;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    MPI_Comm_size(PETSC_COMM_WORLD, &nprocs);

    if (rank == 0) printf("=== Initialization (nprocs=%d) ===\n", nprocs);

    PetscInt X = 100, Y = 100;
    PetscOptionsGetInt(NULL, NULL, "-X", &X, NULL);
    PetscOptionsGetInt(NULL, NULL, "-Y", &Y, NULL);

    PetscBool save_pics = PETSC_FALSE;
    PetscOptionsGetBool(NULL, NULL, "-save_pics", &save_pics, NULL);

    PetscInt coarse_threshold = 32;
    PetscOptionsGetInt(NULL, NULL, "-mg_coarse_threshold",
                       &coarse_threshold, NULL);

    const PetscInt n = X * Y + 1;
    if (rank == 0)
        printf("  Grid: %d x %d  (n = %lld unknowns)\n",
               (int)X, (int)Y, (long long)n);

    auto t0 = Clock::now();
    auto Tgrid_up   = std::make_unique<double[]>(X * Y);
    auto Tgrid_down = std::make_unique<double[]>(X * Y);
    auto Rgrid      = std::make_unique<double[]>(X * Y);

    Tgridset(Tgrid_up.get(), X, Y, 345.0, 10.0);
    std::copy(Tgrid_up.get(), Tgrid_up.get() + X*Y, Tgrid_down.get());
    Rgridset(Rgrid.get(), X, Y, 1000.0);
    print_elapsed(rank, "Grid arrays", t0, Clock::now());

    t0 = Clock::now();
    precomputeHeatingMap(Tgrid_up.get(),   X, Y, 20.0);
    precomputeCoolingMap(Tgrid_down.get(), X, Y, 20.0);
    print_elapsed(rank, "Heating/cooling cascades", t0, Clock::now());

    t0 = Clock::now();
    MGData mg = buildMGData(Rgrid.get(), X, Y, rank, nprocs,
                             PETSC_COMM_WORLD, (int)coarse_threshold);
    print_elapsed(rank, "MGData hierarchy built", t0, Clock::now());

    t0 = Clock::now();
    Vec b, x_vec;
    MatCreateVecs(mg.levels[0].mat, &x_vec, &b);
    VecSetType(b, VECCUDA);
    VecSetType(x_vec, VECCUDA);
    VecZeroEntries(b);
    if (mg.levels[0].ctx.r_start == 0) {
        PetscScalar one = 1.0;
        VecSetValue(b, 0, one, INSERT_VALUES);
    }
    VecAssemblyBegin(b); VecAssemblyEnd(b);
    print_elapsed(rank, "Vectors", t0, Clock::now());

    t0 = Clock::now();
    KSP ksp;
    KSPCreate(PETSC_COMM_WORLD, &ksp);
    KSPSetOperators(ksp, mg.levels[0].mat, mg.levels[0].mat);
    KSPSetType(ksp, KSPCG);
    KSPSetTolerances(ksp, 1e-8, PETSC_DEFAULT, PETSC_DEFAULT, 5000);

    PC pc;
    KSPGetPC(ksp, &pc);

    PCMGSetupData mg_setup;
    setupPCMG(pc, mg, mg_setup, PETSC_COMM_WORLD);

    KSPSetFromOptions(ksp);
    print_elapsed(rank, "KSP/PC setup", t0, Clock::now());

    if (rank == 0)
        printf("  Total init: %.1f ms\n\n",
               duration<double,std::milli>(Clock::now() - t_prog_start).count());

    auto reassemble = [&]() {
        rebuildCoarseLevels(mg, Rgrid.get());
    };

    double start_temp = 300.0, end_temp = 375.0;
    int total_steps = (int)(end_temp - start_temp);

    FILE* f1 = nullptr;
    FILE* f2 = nullptr;
    std::vector<int> sched_up, sched_down;
    if (rank == 0) {
        f1 = fopen("results_up.dat", "w");
        std::set<int> h = generateHighlyBiasedTemps(
            start_temp, end_temp, 345.0, 15, 5.0);
        sched_up.assign(h.begin(), h.end());

        f2 = fopen("results_down.dat", "w");
        std::set<int> c = generateHighlyBiasedTemps(
            start_temp, end_temp, 335.0, 15, 5.0);
        sched_down.assign(c.begin(), c.end());
        std::reverse(sched_down.begin(), sched_down.end());

        printf(">>> STARTING HEATING CYCLE <<<\n");
    }

    size_t up_idx = 0;
    for (int step = 0; step <= total_steps; step+=10) {
        auto t_loop_start = Clock::now();
        double temp  = start_temp + (double)step;
        double ins_R = getSemiconductorR(temp);

        for (PetscInt i = 0; i < X * Y; i++)
            Rgrid[i] = (temp >= Tgrid_up[i]) ? 1.0 : ins_R;

        auto t_asm = Clock::now();
        reassemble();
        auto t_slv = Clock::now();
        KSPSolve(ksp, b, x_vec);
        auto t_end = Clock::now();

        PetscScalar R_tot = 0.0;
        if (mg.levels[0].ctx.r_start == 0) {
            PetscInt idx = 0; VecGetValues(x_vec, 1, &idx, &R_tot);
        }
        MPI_Bcast(&R_tot, 1, MPI_DOUBLE, 0, PETSC_COMM_WORLD);

        if (rank == 0) {
            fprintf(f1, "%f %f\n", temp, PetscRealPart(R_tot));
            if (save_pics && up_idx < sched_up.size() &&
                temp >= sched_up[up_idx]) {
                char fn[64]; sprintf(fn, "heat_%d.png", sched_up[up_idx]);
                save_rgrid_png(fn, Rgrid.get(), X, Y); up_idx++;
            }
            PetscInt its; KSPGetIterationNumber(ksp, &its);
            printf("Step %3d (H) | T:%.1f | R:%.4e | "
                   "Asm:%.1fms | Slv:%.1fms | Tot:%.1fms | It:%d\n",
                   step, temp, PetscRealPart(R_tot),
                   duration<double,std::milli>(t_asm-t_loop_start).count(),
                   duration<double,std::milli>(t_end-t_slv).count(),
                   duration<double,std::milli>(t_end-t_loop_start).count(),
                   (int)its);
        }
    }

    if (rank == 0) { fclose(f1); printf("\n>>> STARTING COOLING CYCLE <<<\n"); }
    size_t dn_idx = 0;

    for (int step = total_steps; step >= 0; step-=10) {
        auto t_loop_start = Clock::now();
        double temp  = start_temp + (double)step;
        double ins_R = getSemiconductorR(temp);
        printf(">>> Step %3d | Temp: %.1f | Ins_R: %.4e\n", step, temp, ins_R);
        fflush(stdout);

        for (PetscInt i = 0; i < X * Y; i++)
            Rgrid[i] = (temp < Tgrid_down[i]) ? ins_R : 1.0;

        auto t_asm = Clock::now();
        reassemble();
        auto t_slv = Clock::now();
        KSPSolve(ksp, b, x_vec);
        auto t_end = Clock::now();

        PetscScalar R_tot = 0.0;
        if (mg.levels[0].ctx.r_start == 0) {
            PetscInt idx = 0; VecGetValues(x_vec, 1, &idx, &R_tot);
        }
        MPI_Bcast(&R_tot, 1, MPI_DOUBLE, 0, PETSC_COMM_WORLD);

        if (rank == 0) {
            fprintf(f2, "%f %f\n", temp, PetscRealPart(R_tot));
            if (save_pics && dn_idx < sched_down.size() &&
                temp <= sched_down[dn_idx]) {
                char fn[64]; sprintf(fn, "cool_%d.png", (int)temp);
                save_rgrid_png(fn, Rgrid.get(), X, Y); dn_idx++;
            }
            PetscInt its; KSPGetIterationNumber(ksp, &its);
            printf("Step %3d (C) | T:%.1f | R:%.4e | "
                   "Asm:%.1fms | Slv:%.1fms | Tot:%.1fms | It:%d\n",
                   step, temp, PetscRealPart(R_tot),
                   duration<double,std::milli>(t_asm-t_loop_start).count(),
                   duration<double,std::milli>(t_end-t_slv).count(),
                   duration<double,std::milli>(t_end-t_loop_start).count(),
                   (int)its);
        }
    }

    // ---- Cleanup ----
    if (rank == 0) fclose(f2);

    for (int lv = 0; lv < mg.nlevels - 1; lv++) {
        MatDestroy(&mg_setup.R_mats[lv]);
        MatDestroy(&mg_setup.P_mats[lv]);
        auto* ig = mg_setup.ig_ctxs[lv];
        VecScatterDestroy(&ig->fine_scatter);
        VecScatterDestroy(&ig->coarse_scatter);
        VecDestroy(&ig->fine_seq);
        VecDestroy(&ig->coarse_seq);
        delete ig;
        mg_setup.ig_ctxs[lv] = nullptr;
        mg_setup.R_mats[lv]  = nullptr;
        mg_setup.P_mats[lv]  = nullptr;
    }

    for (auto& lv : mg.levels) {
        MatDestroy(&lv.mat);
        if (lv.ctx.x_ghost) VecDestroy(&lv.ctx.x_ghost);
        if (lv.ctx.d_Rgrid) cuda_free(lv.ctx.d_Rgrid);
    }
    KSPDestroy(&ksp);
    VecDestroy(&b);
    VecDestroy(&x_vec);
    PetscFinalize();
    return 0;
}

//fix cmdline:"C:\Program Files (x86)\IntelSWTools\mpi\2019.7.216\intel64\bin\mpiexec.exe" -n 6 ./resgrid -X 500 -Y 500 -ksp_type fgmres -pc_type mg -pc_mg_cycle_type w -mg_levels_ksp_type cg -mg_levels_ksp_max_it 2 -mg_levels_pc_type jacobi -mg_coarse_ksp_type gmres -mg_coarse_ksp_max_it 500 -ksp_gmres_restart 100 -mg_coarse_ksp_rtol 1e-4 -ksp_rtol 1e-6 -ksp_monitor