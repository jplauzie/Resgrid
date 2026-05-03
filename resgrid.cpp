#include <petscksp.h>
#include <petscmat.h>
#include <petscpc.h>
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
// Ghost vector has at most 3 entries (row above, row below,
// row 0) — sufficient for the 5-point stencil matvec only.
// Restriction/prolongation use separate VecScatterCreateToAll.
// ============================================================
struct LevelCtx {
    std::vector<double> Rgrid;
    PetscInt Xl, Yl, n;
    PetscInt r_start, r_end;
    Vec      x_ghost;
    std::vector<PetscInt> ghost_indices;
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

static inline PetscInt globalToLocal(const LevelCtx* ctx, PetscInt global)
{
    if (global >= ctx->r_start && global < ctx->r_end)
        return global - ctx->r_start;
    PetscInt lr = ctx->r_end - ctx->r_start;
    for (PetscInt i = 0; i < (PetscInt)ctx->ghost_indices.size(); i++)
        if (ctx->ghost_indices[i] == global) return lr + i;
    return -1;
}

// ============================================================
// Matrix-free matvec: y = A*x
// Uses the 3-entry ghost vector — correct for the 5-point stencil.
// ============================================================
static PetscErrorCode levelMatvec(Mat M, Vec x, Vec y)
{
    LevelCtx* ctx;
    MatShellGetContext(M, (void**)&ctx);

    const PetscInt  Xl      = ctx->Xl;
    const PetscInt  Yl      = ctx->Yl;
    const double*   R       = ctx->Rgrid.data();
    const PetscInt  r_start = ctx->r_start;
    const PetscInt  r_end   = ctx->r_end;
    const PetscInt  lrows   = r_end - r_start;

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
    VecGhostUpdateBegin(ctx->x_ghost, INSERT_VALUES, SCATTER_FORWARD);
    VecGhostUpdateEnd  (ctx->x_ghost, INSERT_VALUES, SCATTER_FORWARD);

    Vec xl; VecGhostGetLocalForm(ctx->x_ghost, &xl);
    const PetscScalar* xarr; VecGetArrayRead(xl, &xarr);
    PetscScalar* yarr;       VecGetArray(y, &yarr);

    for (PetscInt r = r_start; r < r_end; r++) {
        PetscInt lr = r - r_start;
        if (r == 0) {
            double diag = 0.0, val = 0.0;
            for (PetscInt j = 0; j < Yl; j++) {
                double gv = 2.0 / R[j];
                val  += -gv * xarr[globalToLocal(ctx, j+1)];
                diag += gv;
            }
            yarr[lr] = diag * xarr[lr] + val;
        } else {
            PetscInt gi = r-1, i = gi/Yl, j = gi%Yl;
            double diag = 0.0, val = 0.0;

            if (i == 0) {
                double gv = 2.0 / R[gi];
                val  += -gv * xarr[globalToLocal(ctx, 0)];
                diag += gv;
            } else {
                double gv = gmean(R[gi], R[(i-1)*Yl+j]);
                val  += -gv * xarr[globalToLocal(ctx, r-Yl)];
                diag += gv;
            }
            if (j > 0) {
                double gv = gmean(R[gi], R[i*Yl+(j-1)]);
                val  += -gv * xarr[globalToLocal(ctx, r-1)];
                diag += gv;
            }
            if (j < Yl-1) {
                double gv = gmean(R[gi], R[i*Yl+(j+1)]);
                val  += -gv * xarr[globalToLocal(ctx, r+1)];
                diag += gv;
            }
            if (i < Xl-1) {
                double gv = gmean(R[gi], R[(i+1)*Yl+j]);
                val  += -gv * xarr[globalToLocal(ctx, r+Yl)];
                diag += gv;
            } else {
                diag += 2.0 / R[gi];
            }
            yarr[lr] = diag * xarr[lr] + val;
        }
    }

    VecRestoreArrayRead(xl, &xarr);
    VecGhostRestoreLocalForm(ctx->x_ghost, &xl);
    VecRestoreArray(y, &yarr);
    return 0;
}

// ============================================================
// Diagonal extraction for Jacobi sub-PC. No communication.
// ============================================================
static PetscErrorCode levelGetDiagonal(Mat M, Vec diag)
{
    LevelCtx* ctx;
    MatShellGetContext(M, (void**)&ctx);
    const PetscInt Xl = ctx->Xl, Yl = ctx->Yl;
    const double*  R  = ctx->Rgrid.data();

    PetscScalar* d; VecGetArray(diag, &d);
    for (PetscInt r = ctx->r_start; r < ctx->r_end; r++) {
        PetscInt lr = r - ctx->r_start;
        if (r == 0) {
            double v = 0.0;
            for (PetscInt j = 0; j < Yl; j++) v += 2.0 / R[j];
            d[lr] = v;
        } else {
            PetscInt gi = r-1, i = gi/Yl, j = gi%Yl;
            double v = 0.0;
            if (i == 0)    v += 2.0 / R[gi];
            else           v += gmean(R[gi], R[(i-1)*Yl+j]);
            if (j > 0)     v += gmean(R[gi], R[i*Yl+(j-1)]);
            if (j < Yl-1)  v += gmean(R[gi], R[i*Yl+(j+1)]);
            if (i < Xl-1)  v += gmean(R[gi], R[(i+1)*Yl+j]);
            else            v += 2.0 / R[gi];
            d[lr] = v;
        }
    }
    VecRestoreArray(diag, &d);
    return 0;
}

// ============================================================
// Coarsen Rgrid 2x in both dims using harmonic mean.
// Handles non-power-of-2 by clamping to last fine index.
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

static void buildGhostVec(LevelCtx& ctx, MPI_Comm comm)
{
    ctx.ghost_indices = buildGhostIndices(ctx.n, ctx.r_start, ctx.r_end);
    PetscInt lr = ctx.r_end - ctx.r_start;
    VecCreateGhost(comm, lr, ctx.n,
                   (PetscInt)ctx.ghost_indices.size(),
                   ctx.ghost_indices.data(), &ctx.x_ghost);
}

// ============================================================
// Intergrid context for restriction and prolongation.
//
// Uses VecScatterCreateToAll for both fine and coarse vectors.
// This is the key fix: the matvec ghost only has 3 entries and
// cannot be reused for restriction which needs arbitrary rows.
// VecScatterCreateToAll has simple O(n) setup with no IS analysis
// and no risk of hanging.
//
// Memory cost: one full copy of fine vec per rank during restrict
// (e.g. 500x500 = 2MB), one full copy of coarse vec during prolong.
// Both are acceptable and much cheaper than the previous IS approach.
// ============================================================
struct IntergridCtx {
    LevelCtx*  fine;
    LevelCtx*  coarse;
    VecScatter fine_scatter;    // replicates full fine vec locally
    Vec        fine_seq;        // sequential fine vec on each rank
    VecScatter coarse_scatter;  // replicates full coarse vec locally
    Vec        coarse_seq;      // sequential coarse vec on each rank
};

static IntergridCtx* buildIntergridCtx(LevelCtx* fine, LevelCtx* coarse,
                                        MPI_Comm comm)
{
    auto* ig   = new IntergridCtx();
    ig->fine   = fine;
    ig->coarse = coarse;

    // Fine scatter: replicate entire fine vector on every rank.
    // Used by restriction. VecScatterCreateToAll is O(n) setup,
    // no IS analysis, no hang risk.
    Vec fine_mpi;
    VecCreateMPI(comm, fine->r_end - fine->r_start, fine->n, &fine_mpi);
    VecScatterCreateToAll(fine_mpi, &ig->fine_scatter, &ig->fine_seq);
    VecDestroy(&fine_mpi);

    // Coarse scatter: replicate entire coarse vector on every rank.
    // Used by prolongation. Coarse is tiny (<=32x32+1=1025 entries).
    Vec coarse_mpi;
    VecCreateMPI(comm, coarse->r_end - coarse->r_start,
                 coarse->n, &coarse_mpi);
    VecScatterCreateToAll(coarse_mpi, &ig->coarse_scatter, &ig->coarse_seq);
    VecDestroy(&coarse_mpi);

    return ig;
}

// ============================================================
// Restriction: xc = R * xf
// Replicates full fine vec, then each rank fills its owned
// coarse rows by averaging the 2x2 fine block.
// Direct array indexing — no ghost lookup, no binary search.
// ============================================================
static PetscErrorCode restrictVec(Mat Rmat, Vec xf, Vec xc)
{
    IntergridCtx* ig;
    MatShellGetContext(Rmat, (void**)&ig);
    LevelCtx* fine   = ig->fine;
    LevelCtx* coarse = ig->coarse;

    // Scatter full fine vector to local sequential copy.
    VecScatterBegin(ig->fine_scatter, xf, ig->fine_seq,
                    INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd  (ig->fine_scatter, xf, ig->fine_seq,
                    INSERT_VALUES, SCATTER_FORWARD);

    const PetscScalar* fseq; VecGetArrayRead(ig->fine_seq, &fseq);
    PetscScalar*       xca;  VecGetArray(xc, &xca);

    const PetscInt Yf = fine->Yl, Yc = coarse->Yl;

    for (PetscInt rc = coarse->r_start; rc < coarse->r_end; rc++) {
        PetscInt lc = rc - coarse->r_start;
        if (rc == 0) {
            // Source node: direct injection.
            // fseq[0] is global index 0 (source node).
            xca[lc] = fseq[0];
        } else {
            PetscInt gc = rc - 1;
            PetscInt ic = gc / Yc, jc = gc % Yc;
            PetscInt if0=2*ic, if1=std::min(2*ic+1, fine->Xl-1);
            PetscInt jf0=2*jc, jf1=std::min(2*jc+1, Yf-1);

            // Full-weighting average over 2x2 fine block.
            // fseq global index of fine grain (i,j) = i*Yf + j + 1
            // (+1 because global index 0 is the source node).
            int cnt = 0; PetscScalar sum = 0.0;
            auto acc = [&](PetscInt ri, PetscInt rj) {
                sum += fseq[ri*Yf + rj + 1]; cnt++;
            };
            acc(if0, jf0);
            if (jf1 != jf0) acc(if0, jf1);
            if (if1 != if0) acc(if1, jf0);
            if (if1 != if0 && jf1 != jf0) acc(if1, jf1);
            xca[lc] = sum / (PetscScalar)cnt;
        }
    }

    VecRestoreArrayRead(ig->fine_seq, &fseq);
    VecRestoreArray(xc, &xca);
    return 0;
}

// ============================================================
// Prolongation: xf = P * xc
// Replicates full coarse vec, then each rank fills its owned
// fine rows by injecting the enclosing coarse node's value.
// ============================================================
static PetscErrorCode prolongVec(Mat Pmat, Vec xc, Vec xf)
{
    IntergridCtx* ig;
    MatShellGetContext(Pmat, (void**)&ig);
    LevelCtx* fine   = ig->fine;
    LevelCtx* coarse = ig->coarse;

    // Replicate full coarse vector locally — cheap (<=1025 entries).
    VecScatterBegin(ig->coarse_scatter, xc, ig->coarse_seq,
                    INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd  (ig->coarse_scatter, xc, ig->coarse_seq,
                    INSERT_VALUES, SCATTER_FORWARD);

    const PetscScalar* cseq; VecGetArrayRead(ig->coarse_seq, &cseq);
    PetscScalar*       xfa;  VecGetArray(xf, &xfa);

    const PetscInt Yf = fine->Yl, Yc = coarse->Yl;

    for (PetscInt rf = fine->r_start; rf < fine->r_end; rf++) {
        PetscInt lf = rf - fine->r_start;
        if (rf == 0) {
            // Source node: direct injection from coarse source.
            xfa[lf] = cseq[0];
        } else {
            PetscInt gf  = rf - 1;
            PetscInt iff = gf / Yf, jf = gf % Yf;
            PetscInt ic  = iff / 2, jc = jf / 2;
            if (ic >= coarse->Xl) ic = coarse->Xl - 1;
            if (jc >= coarse->Yl) jc = coarse->Yl - 1;
            // cseq index: 0=source, grain (ic,jc) = ic*Yc+jc+1
            xfa[lf] = cseq[ic * Yc + jc + 1];
        }
    }

    VecRestoreArrayRead(ig->coarse_seq, &cseq);
    VecRestoreArray(xf, &xfa);
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
        L.ctx.Rgrid.resize(L.ctx.Xl * L.ctx.Yl, 1.0);
        buildLevelMat(L, comm);
    }

    // Initial fill from fine_Rgrid.
    mg.levels[0].ctx.Rgrid.assign(fine_Rgrid, fine_Rgrid + X*Y);
    for (int lv = 1; lv < mg.nlevels; lv++) {
        auto& p = mg.levels[lv-1].ctx;
        auto& c = mg.levels[lv].ctx;
        c.Rgrid = coarsenRgrid(p.Rgrid.data(), p.Xl, p.Yl, c.Xl, c.Yl);
    }

    return mg;
}

static void rebuildCoarseLevels(MGData& mg, const double* fine_Rgrid)
{
    mg.levels[0].ctx.Rgrid.assign(
        fine_Rgrid,
        fine_Rgrid + mg.levels[0].ctx.Xl * mg.levels[0].ctx.Yl);
    for (int lv = 1; lv < mg.nlevels; lv++) {
        auto& p = mg.levels[lv-1].ctx;
        auto& c = mg.levels[lv].ctx;
        c.Rgrid = coarsenRgrid(p.Rgrid.data(), p.Xl, p.Yl, c.Xl, c.Yl);
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

    // PETSc PCMG level convention: 0=coarsest, nlevels-1=finest.
    // Our MGData convention:        0=finest,   nlevels-1=coarsest.
    // Mapping: pcmg_lv = (nlevels-1) - our_lv

    for (int lv = 0; lv < nlevels; lv++) {
        int pcmg_lv = (nlevels-1) - lv;
        PCMGSetOperators(pc, pcmg_lv,
                         mg.levels[lv].mat, mg.levels[lv].mat);
    }
    print_elapsed(0, "second", PCMGstart, Clock::now());

    for (int lv = 0; lv < nlevels-1; lv++) {
        int pcmg_fine = (nlevels-1) - lv;  // PCMG index of the fine level

        LevelCtx* fine_ctx   = &mg.levels[lv].ctx;
        LevelCtx* coarse_ctx = &mg.levels[lv+1].ctx;

        IntergridCtx* ig = buildIntergridCtx(fine_ctx, coarse_ctx, comm);
        setup.ig_ctxs[lv] = ig;

        PetscInt nf=fine_ctx->n,   lf=fine_ctx->r_end   - fine_ctx->r_start;
        PetscInt nc=coarse_ctx->n, lc=coarse_ctx->r_end  - coarse_ctx->r_start;

        // Restriction shell: fine -> coarse  (lc rows, lf cols)
        MatCreateShell(comm, lc, lf, nc, nf, ig, &setup.R_mats[lv]);
        MatShellSetOperation(setup.R_mats[lv], MATOP_MULT,
                             (void(*)(void))restrictVec);

        // Prolongation shell: coarse -> fine  (lf rows, lc cols)
        MatCreateShell(comm, lf, lc, nf, nc, ig, &setup.P_mats[lv]);
        MatShellSetOperation(setup.P_mats[lv], MATOP_MULT,
                             (void(*)(void))prolongVec);

        PCMGSetRestriction (pc, pcmg_fine, setup.R_mats[lv]);
        PCMGSetInterpolation(pc, pcmg_fine, setup.P_mats[lv]);
    }
    print_elapsed(0, "third", PCMGstart, Clock::now());

    // Smoothers on all non-coarsest levels (pcmg levels 1..nlevels-1).
    for (int pcmg_lv = 1; pcmg_lv < nlevels; pcmg_lv++) {
        KSP smoother;
        PCMGGetSmoother(pc, pcmg_lv, &smoother);
        KSPSetType(smoother, KSPCHEBYSHEV);
        PC sub_pc; KSPGetPC(smoother, &sub_pc);
        PCSetType(sub_pc, PCJACOBI);
        KSPSetTolerances(smoother, PETSC_DEFAULT, PETSC_DEFAULT,
                         PETSC_DEFAULT, 2);
        // Inflate upper eigenvalue bound 30% — safe overestimate.
        // Prevents indefinite-preconditioner failure near the transition
        // where the spectrum shifts rapidly. Slight weakening of smoother
        // is acceptable; underestimate would break outer CG.
        KSPChebyshevEstEigSet(smoother, 0.0, 0.1, 0.0, 1.3);
        KSPChebyshevEstEigSetUseNoisy(smoother, PETSC_TRUE);
    }
    print_elapsed(0, "fourth", PCMGstart, Clock::now());

    // Coarsest level (pcmg level 0): iterative solve, no explicit matrix.
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
    for (int step = 0; step <= total_steps; step++) {
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

    for (int step = total_steps; step >= 0; step--) {
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

    // ---- Cleanup ---- (must all happen before PetscFinalize)
    if (rank == 0) fclose(f2);

    // Destroy intergrid contexts and their mats explicitly
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
    }
    KSPDestroy(&ksp);
    VecDestroy(&b);
    VecDestroy(&x_vec);
    PetscFinalize();   // now nothing PETSc-owned survives past this
    return 0;
}