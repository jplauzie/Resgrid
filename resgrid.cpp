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

// ============================================================
// Harmonic mean conductance between two grains.
// Identical to your existing g() — reproduced here for clarity.
// For two resistors R1, R2 in series sharing a bond:
//   g_eff = 2 / (R1 + R2)  [harmonic mean of conductances]
// This is physically correct for the inter-grain conductance.
// ============================================================
static inline double gmean(double R1, double R2)
{
    return 2.0 / (R1 + R2);
}

// ============================================================
// Per-level context for matrix-free matvec.
//
// Each multigrid level owns:
//   - A coarsened Rgrid of size (Xl * Yl), harmonically averaged
//     from the level below.
//   - Ghost vector infrastructure for the boundary communication.
//
// The source node (global index 0 on every level) connects to
// every grain in the top row (grid indices 0..Yl-1 on that level)
// with conductance  2/R[j]  (same boundary condition as fine grid,
// applied to the coarsened resistances).
//
// Ghost strategy: unchanged from your original — at most 3 ghost
// entries per rank (row above, row below, row 0).
// ============================================================
struct LevelCtx {
    std::vector<double> Rgrid;   // coarsened resistances, size Xl*Yl
    PetscInt Xl, Yl;             // grid dimensions on this level
    PetscInt n;                  // total DOFs = Xl*Yl + 1
    PetscInt r_start, r_end;     // local row ownership
    Vec      x_ghost;            // ghost vector reused each matvec
    std::vector<PetscInt> ghost_indices;
};

// ============================================================
// Ghost index construction — identical logic to your original.
// ============================================================
static std::vector<PetscInt> buildGhostIndices(
    PetscInt n, PetscInt r_start, PetscInt r_end)
{
    std::vector<PetscInt> ghosts;
    ghosts.reserve(3);
    if (r_start > 1)  ghosts.push_back(r_start - 1);
    if (r_end   < n)  ghosts.push_back(r_end);
    if (r_start > 0)  ghosts.push_back(0);
    std::sort(ghosts.begin(), ghosts.end());
    ghosts.erase(std::unique(ghosts.begin(), ghosts.end()), ghosts.end());
    return ghosts;
}

static inline PetscInt globalToLocal(const LevelCtx* ctx, PetscInt global)
{
    if (global >= ctx->r_start && global < ctx->r_end)
        return global - ctx->r_start;
    PetscInt local_rows = ctx->r_end - ctx->r_start;
    for (PetscInt i = 0; i < (PetscInt)ctx->ghost_indices.size(); i++)
        if (ctx->ghost_indices[i] == global)
            return local_rows + i;
    return -1;
}

// ============================================================
// Matrix-free matvec: y = A_level * x
// Works for any level — fine or coarse.
// ============================================================
static PetscErrorCode levelMatvec(Mat M, Vec x, Vec y)
{
    LevelCtx* ctx;
    MatShellGetContext(M, (void**)&ctx);

    const PetscInt    Xl      = ctx->Xl;
    const PetscInt    Yl      = ctx->Yl;
    const double*     R       = ctx->Rgrid.data();
    const PetscInt    r_start = ctx->r_start;
    const PetscInt    r_end   = ctx->r_end;
    const PetscInt    local_rows = r_end - r_start;

    // Copy owned values into ghost vector, scatter ghosts.
    {
        const PetscScalar* x_arr;
        VecGetArrayRead(x, &x_arr);
        Vec x_local;
        VecGhostGetLocalForm(ctx->x_ghost, &x_local);
        PetscScalar* ghost_arr;
        VecGetArray(x_local, &ghost_arr);
        for (PetscInt i = 0; i < local_rows; i++) ghost_arr[i] = x_arr[i];
        VecRestoreArray(x_local, &ghost_arr);
        VecGhostRestoreLocalForm(ctx->x_ghost, &x_local);
        VecRestoreArrayRead(x, &x_arr);
    }
    VecGhostUpdateBegin(ctx->x_ghost, INSERT_VALUES, SCATTER_FORWARD);
    VecGhostUpdateEnd  (ctx->x_ghost, INSERT_VALUES, SCATTER_FORWARD);

    Vec x_local;
    VecGhostGetLocalForm(ctx->x_ghost, &x_local);
    const PetscScalar* xarr;
    VecGetArrayRead(x_local, &xarr);
    PetscScalar* yarr;
    VecGetArray(y, &yarr);

    for (PetscInt r = r_start; r < r_end; r++) {
        PetscInt lr = r - r_start;

        if (r == 0) {
            // Source node: connected to all grains in top row.
            double diag = 0.0, val = 0.0;
            for (PetscInt j = 0; j < Yl; j++) {
                double gval = 2.0 / R[j];
                PetscInt lj = globalToLocal(ctx, j + 1);
                val  += -gval * xarr[lj];
                diag += gval;
            }
            yarr[lr] = diag * xarr[lr] + val;
        } else {
            PetscInt grid_idx = r - 1;
            PetscInt i = grid_idx / Yl;
            PetscInt j = grid_idx % Yl;
            double diag = 0.0, val = 0.0;

            // Up
            if (i == 0) {
                double gval = 2.0 / R[grid_idx];
                PetscInt l0 = globalToLocal(ctx, 0);
                val  += -gval * xarr[l0];
                diag += gval;
            } else {
                double gval = gmean(R[grid_idx], R[(i-1)*Yl+j]);
                PetscInt lup = globalToLocal(ctx, r - Yl);
                val  += -gval * xarr[lup];
                diag += gval;
            }

            // Left
            if (j > 0) {
                double gval = gmean(R[grid_idx], R[i*Yl+(j-1)]);
                PetscInt ll = globalToLocal(ctx, r - 1);
                val  += -gval * xarr[ll];
                diag += gval;
            }

            // Right
            if (j < Yl - 1) {
                double gval = gmean(R[grid_idx], R[i*Yl+(j+1)]);
                PetscInt lr2 = globalToLocal(ctx, r + 1);
                val  += -gval * xarr[lr2];
                diag += gval;
            }

            // Down
            if (i < Xl - 1) {
                double gval = gmean(R[grid_idx], R[(i+1)*Yl+j]);
                PetscInt ldown = globalToLocal(ctx, r + Yl);
                val  += -gval * xarr[ldown];
                diag += gval;
            } else {
                // Bottom boundary: grounded (Dirichlet V=0 via
                // image conductance, same as top boundary at source).
                diag += 2.0 / R[grid_idx];
            }

            yarr[lr] = diag * xarr[lr] + val;
        }
    }

    VecRestoreArrayRead(x_local, &xarr);
    VecGhostRestoreLocalForm(ctx->x_ghost, &x_local);
    VecRestoreArray(y, &yarr);
    return 0;
}

// ============================================================
// Matrix-free diagonal extraction (for Jacobi sub-PC inside
// Chebyshev smoother — no communication needed).
// ============================================================
static PetscErrorCode levelGetDiagonal(Mat M, Vec diag)
{
    LevelCtx* ctx;
    MatShellGetContext(M, (void**)&ctx);

    const PetscInt  Xl = ctx->Xl;
    const PetscInt  Yl = ctx->Yl;
    const double*   R  = ctx->Rgrid.data();

    PetscScalar* d;
    VecGetArray(diag, &d);

    for (PetscInt r = ctx->r_start; r < ctx->r_end; r++) {
        PetscInt lr = r - ctx->r_start;

        if (r == 0) {
            double val = 0.0;
            for (PetscInt j = 0; j < Yl; j++) val += 2.0 / R[j];
            d[lr] = val;
        } else {
            PetscInt grid_idx = r - 1;
            PetscInt i = grid_idx / Yl;
            PetscInt j = grid_idx % Yl;
            double val = 0.0;

            if (i == 0)    val += 2.0 / R[grid_idx];
            else           val += gmean(R[grid_idx], R[(i-1)*Yl+j]);

            if (j > 0)     val += gmean(R[grid_idx], R[i*Yl+(j-1)]);
            if (j < Yl-1)  val += gmean(R[grid_idx], R[i*Yl+(j+1)]);

            if (i < Xl-1)  val += gmean(R[grid_idx], R[(i+1)*Yl+j]);
            else            val += 2.0 / R[grid_idx];

            d[lr] = val;
        }
    }
    VecRestoreArray(diag, &d);
    return 0;
}

// ============================================================
// Coarsen Rgrid by 2x in both dimensions using harmonic averaging.
//
// Each coarse grain (ic, jc) represents a 2x2 block of fine grains.
// We use the harmonic mean of the 4 fine resistances because
// parallel/series combinations of resistors average harmonically.
//
// For the source node connection (top boundary), we average the
// top two fine grains in each 2x2 block harmonically.
//
// NOTE: If Xf or Yf is odd, the last coarse row/col covers only
// one fine grain (no averaging needed). This handles non-power-of-2
// grids gracefully.
// ============================================================
static std::vector<double> coarsenRgrid(
    const double* Rf, PetscInt Xf, PetscInt Yf,
    PetscInt Xc, PetscInt Yc)
{
    std::vector<double> Rc(Xc * Yc, 0.0);

    for (PetscInt ic = 0; ic < Xc; ic++) {
        for (PetscInt jc = 0; jc < Yc; jc++) {
            // Fine-grid indices for this 2x2 block
            PetscInt if0 = 2*ic,     if1 = std::min(2*ic+1, Xf-1);
            PetscInt jf0 = 2*jc,     jf1 = std::min(2*jc+1, Yf-1);

            double r00 = Rf[if0*Yf + jf0];
            double r01 = Rf[if0*Yf + jf1];
            double r10 = Rf[if1*Yf + jf0];
            double r11 = Rf[if1*Yf + jf1];

            // Harmonic mean of up to 4 values.
            // Count distinct grains (handles odd dimensions).
            int count = 0;
            double sum_inv = 0.0;
            auto acc = [&](double r){ sum_inv += 1.0/r; count++; };
            acc(r00);
            if (jf1 != jf0) acc(r01);
            if (if1 != if0) acc(r10);
            if (if1 != if0 && jf1 != jf0) acc(r11);

            // Harmonic mean: 1/R_eff = (1/n) * sum(1/Ri)
            // => R_eff = n / sum(1/Ri)
            Rc[ic*Yc + jc] = (double)count / sum_inv;
        }
    }
    return Rc;
}

// ============================================================
// Full multigrid level data.
// ============================================================
struct MGLevel {
    LevelCtx ctx;
    Mat      mat;       // MATSHELL referencing ctx
};

// ============================================================
// Build and initialise all multigrid levels.
//
// Level 0 = finest.  Level nlevels-1 = coarsest.
// Coarsest grid is at most 32x32 (enforced by stopping coarsening
// when min(X,Y) <= coarse_threshold, default 32).
//
// The fine-level Rgrid pointer is set to the live simulation array
// so it is always up-to-date without any copying.
// Coarser levels hold their own std::vector<double> that must be
// rebuilt via rebuildCoarseLevels() whenever Rgrid changes.
// ============================================================
struct MGData {
    std::vector<MGLevel> levels;  // [0]=fine .. [nlevels-1]=coarse
    int nlevels;
};

static PetscErrorCode buildLevelMat(MGLevel& lv, MPI_Comm comm)
{
    PetscInt n         = lv.ctx.n;
    PetscInt local_rows = lv.ctx.r_end - lv.ctx.r_start;

    MatCreateShell(comm, local_rows, local_rows, n, n,
                   &lv.ctx, &lv.mat);
    MatShellSetOperation(lv.mat, MATOP_MULT,
                         (void(*)(void))levelMatvec);
    MatShellSetOperation(lv.mat, MATOP_GET_DIAGONAL,
                         (void(*)(void))levelGetDiagonal);
    return 0;
}

// Initialise ghost vector for a level given its ownership range.
static PetscErrorCode buildGhostVec(LevelCtx& ctx, MPI_Comm comm)
{
    ctx.ghost_indices = buildGhostIndices(ctx.n, ctx.r_start, ctx.r_end);
    PetscInt local_rows = ctx.r_end - ctx.r_start;
    VecCreateGhost(comm,
                   local_rows, ctx.n,
                   (PetscInt)ctx.ghost_indices.size(),
                   ctx.ghost_indices.data(),
                   &ctx.x_ghost);
    return 0;
}

// Determine row ownership range for a given total size n,
// matching PETSc's default block distribution.
static void getOwnershipRange(PetscInt n, PetscMPIInt rank,
                               PetscMPIInt nprocs,
                               PetscInt& r_start, PetscInt& r_end)
{
    PetscInt base  = n / nprocs;
    PetscInt extra = n % nprocs;
    r_start = rank * base + std::min((PetscInt)rank, extra);
    r_end   = r_start + base + ((PetscInt)rank < extra ? 1 : 0);
}

static MGData buildMGData(
    const double* fine_Rgrid,
    PetscInt X, PetscInt Y,
    PetscMPIInt rank, PetscMPIInt nprocs,
    MPI_Comm comm,
    int coarse_threshold = 32)
{
    MGData mg;
    mg.nlevels = 0;

    // Collect grid sizes for each level by repeatedly halving.
    std::vector<std::pair<PetscInt,PetscInt>> grid_sizes;
    PetscInt cx = X, cy = Y;
    while (true) {
        grid_sizes.push_back({cx, cy});
        if (cx <= coarse_threshold || cy <= coarse_threshold) break;
        cx = (cx + 1) / 2;
        cy = (cy + 1) / 2;
    }
    mg.nlevels = (int)grid_sizes.size();
    mg.levels.resize(mg.nlevels);

    if (rank == 0)
        printf("  GMG: %d levels, coarsest grid %dx%d (%d DOFs)\n",
               mg.nlevels,
               (int)grid_sizes.back().first,
               (int)grid_sizes.back().second,
               (int)(grid_sizes.back().first * grid_sizes.back().second + 1));

    for (int lv = 0; lv < mg.nlevels; lv++) {
        auto& L    = mg.levels[lv];
        L.ctx.Xl   = grid_sizes[lv].first;
        L.ctx.Yl   = grid_sizes[lv].second;
        L.ctx.n    = L.ctx.Xl * L.ctx.Yl + 1;
        getOwnershipRange(L.ctx.n, rank, nprocs,
                          L.ctx.r_start, L.ctx.r_end);
        buildGhostVec(L.ctx, comm);

        if (lv == 0) {
            // Fine level: point directly at the live simulation array.
            // We cast away const here; the matvec only reads it.
            L.ctx.Rgrid.assign(fine_Rgrid, fine_Rgrid + X*Y);
        }
        // Coarse levels: vectors are filled by rebuildCoarseLevels().

        buildLevelMat(L, comm);
    }
    return mg;
}

// Call this every time Rgrid changes (each temperature step).
// Rebuilds the coarsened resistance arrays from level 0 upward.
// Level 0 always mirrors the live fine Rgrid pointer.
static void rebuildCoarseLevels(MGData& mg, const double* fine_Rgrid)
{
    // Update fine level (level 0)
    mg.levels[0].ctx.Rgrid.assign(
        fine_Rgrid,
        fine_Rgrid + mg.levels[0].ctx.Xl * mg.levels[0].ctx.Yl);

    for (int lv = 1; lv < mg.nlevels; lv++) {
        // parent and child are aliases to the ctx member already
        auto& parent = mg.levels[lv-1].ctx; 
        auto& child  = mg.levels[lv].ctx;

        // Corrected: Removed .ctx from child accesses
        child.Rgrid = coarsenRgrid(
            parent.Rgrid.data(), parent.Xl, parent.Yl,
            child.Xl, child.Yl);
    }
}

// ============================================================
// Restriction operator: full-weighting on the grid DOFs,
// direct injection for the source node (index 0).
//
// For a 2D grid node (ic, jc) on the coarse level, the
// full-weighting stencil averages from 4 fine-grid neighbours
// (with equal weight 1/4 for simplicity; bilinear weighting
// can be added later without changing correctness).
//
// Called by PCMG via PCMGSetRestriction().
// ============================================================
static PetscErrorCode restrictVec(Mat R, Vec xf, Vec xc)
{
    // We store the level pair as a 2-element array in the shell context.
    // [0] = fine LevelCtx*, [1] = coarse LevelCtx*
    void* ctx_raw;
    MatShellGetContext(R, &ctx_raw);
    LevelCtx** pair = (LevelCtx**)ctx_raw;
    LevelCtx* fine   = pair[0];
    LevelCtx* coarse = pair[1];

    const PetscInt Xf = fine->Xl,   Yf = fine->Yl;
    const PetscInt Xc = coarse->Xl, Yc = coarse->Yl;

    // We need the full fine vector on each rank for restriction.
    // For a structured grid this is unavoidable — but we only
    // scatter the boundary rows we actually need, reusing the
    // ghost mechanism of the fine level.
    {
        const PetscScalar* xf_arr;
        VecGetArrayRead(xf, &xf_arr);
        Vec xf_local;
        VecGhostGetLocalForm(fine->x_ghost, &xf_local);
        PetscScalar* ghost_arr;
        VecGetArray(xf_local, &ghost_arr);
        PetscInt flocal = fine->r_end - fine->r_start;
        for (PetscInt i = 0; i < flocal; i++) ghost_arr[i] = xf_arr[i];
        VecRestoreArray(xf_local, &ghost_arr);
        VecGhostRestoreLocalForm(fine->x_ghost, &xf_local);
        VecRestoreArrayRead(xf, &xf_arr);
    }
    VecGhostUpdateBegin(fine->x_ghost, INSERT_VALUES, SCATTER_FORWARD);
    VecGhostUpdateEnd  (fine->x_ghost, INSERT_VALUES, SCATTER_FORWARD);

    Vec xf_local;
    VecGhostGetLocalForm(fine->x_ghost, &xf_local);
    const PetscScalar* xfarr;
    VecGetArrayRead(xf_local, &xfarr);
    PetscScalar* xcarr;
    VecGetArray(xc, &xcarr);

    auto fget = [&](PetscInt rf) -> PetscScalar {
        PetscInt lf = globalToLocal(fine, rf);
        if (lf < 0) return 0.0;  // not available — contributes 0
        return xfarr[lf];
    };

    for (PetscInt rc = coarse->r_start; rc < coarse->r_end; rc++) {
        PetscInt lc = rc - coarse->r_start;

        if (rc == 0) {
            // Source node: inject directly.
            xcarr[lc] = fget(0);
        } else {
            PetscInt grid_idx_c = rc - 1;
            PetscInt ic = grid_idx_c / Yc;
            PetscInt jc = grid_idx_c % Yc;

            // Corresponding fine-grid nodes (2x2 block).
            PetscInt if0 = 2*ic,              if1 = std::min(2*ic+1, Xf-1);
            PetscInt jf0 = 2*jc,              jf1 = std::min(2*jc+1, Yf-1);

            // Full-weighting: average fine values in the 2x2 block.
            int cnt = 0;
            PetscScalar sum = 0.0;
            auto acc = [&](PetscInt rfi, PetscInt rfj) {
                PetscInt rfidx = rfi*Yf + rfj + 1; // +1 for source offset
                sum += fget(rfidx); cnt++;
            };
            acc(if0, jf0);
            if (jf1 != jf0) acc(if0, jf1);
            if (if1 != if0) acc(if1, jf0);
            if (if1 != if0 && jf1 != jf0) acc(if1, jf1);
            xcarr[lc] = sum / (PetscScalar)cnt;
        }
    }

    VecRestoreArrayRead(xf_local, &xfarr);
    VecGhostRestoreLocalForm(fine->x_ghost, &xf_local);
    VecRestoreArray(xc, &xcarr);
    return 0;
}

// ============================================================
// Prolongation (interpolation) operator: bilinear on grid DOFs,
// direct injection for source node.
// P = R^T (adjoint of restriction), so prolongation just injects
// each coarse value to the overlapping fine nodes.
// ============================================================
static PetscErrorCode prolongVec(Mat P, Vec xc, Vec xf)
{
    void* ctx_raw;
    MatShellGetContext(P, &ctx_raw);
    LevelCtx** pair = (LevelCtx**)ctx_raw;
    LevelCtx* fine   = pair[0];
    LevelCtx* coarse = pair[1];

    const PetscInt Xf = fine->Xl,   Yf = fine->Yl;
    const PetscInt Xc = coarse->Xl, Yc = coarse->Yl;

    // Need full coarse vector; replicate it similarly.
    {
        const PetscScalar* xc_arr;
        VecGetArrayRead(xc, &xc_arr);
        Vec xc_local;
        VecGhostGetLocalForm(coarse->x_ghost, &xc_local);
        PetscScalar* ghost_arr;
        VecGetArray(xc_local, &ghost_arr);
        PetscInt clocal = coarse->r_end - coarse->r_start;
        for (PetscInt i = 0; i < clocal; i++) ghost_arr[i] = xc_arr[i];
        VecRestoreArray(xc_local, &ghost_arr);
        VecGhostRestoreLocalForm(coarse->x_ghost, &xc_local);
        VecRestoreArrayRead(xc, &xc_arr);
    }
    VecGhostUpdateBegin(coarse->x_ghost, INSERT_VALUES, SCATTER_FORWARD);
    VecGhostUpdateEnd  (coarse->x_ghost, INSERT_VALUES, SCATTER_FORWARD);

    Vec xc_local;
    VecGhostGetLocalForm(coarse->x_ghost, &xc_local);
    const PetscScalar* xcarr;
    VecGetArrayRead(xc_local, &xcarr);
    PetscScalar* xfarr;
    VecGetArray(xf, &xfarr);

    auto cget = [&](PetscInt rc) -> PetscScalar {
        PetscInt lc = globalToLocal(coarse, rc);
        if (lc < 0) return 0.0;
        return xcarr[lc];
    };

    VecZeroEntries(xf);

    for (PetscInt rf = fine->r_start; rf < fine->r_end; rf++) {
        PetscInt lf = rf - fine->r_start;

        if (rf == 0) {
            xfarr[lf] = cget(0);
        } else {
            PetscInt grid_idx_f = rf - 1;
            PetscInt if_ = grid_idx_f / Yf;
            PetscInt jf  = grid_idx_f % Yf;

            // Each fine node maps to the coarse node of its 2x2 block.
            PetscInt ic = if_ / 2;
            PetscInt jc = jf  / 2;
            if (ic >= Xc) ic = Xc - 1;
            if (jc >= Yc) jc = Yc - 1;
            PetscInt rc = ic * Yc + jc + 1;
            xfarr[lf] = cget(rc);
        }
    }

    VecRestoreArrayRead(xc_local, &xcarr);
    VecGhostRestoreLocalForm(coarse->x_ghost, &xc_local);
    VecRestoreArray(xf, &xfarr);
    return 0;
}

// ============================================================
// Pair context for restriction/prolongation shells.
// Stored as a flat array of two pointers to avoid heap allocation.
// One pair per inter-level interface.
// ============================================================
struct LevelPair {
    LevelCtx* fine;
    LevelCtx* coarse;
    LevelCtx* arr[2];  // arr[0]=fine, arr[1]=coarse, passed to MatShell
};

// ============================================================
// Build and configure the PCMG preconditioner from MGData.
//
// Uses Chebyshev+Jacobi as the smoother on all levels — this
// combination is:
//   - Fully matrix-free (Jacobi only needs the diagonal)
//   - A fixed linear operator (Chebyshev with fixed degree)
//   - SPD-preserving, so outer CG remains valid
//
// Eigenvalue estimation uses CG (-mg_levels_esteig_ksp_type cg)
// which is more accurate for SPD systems than GMRES.
// Upper bound is inflated by 30% as a conservative safety margin
// for the high-contrast regime near the VO2 transition.
//
// Coarsest level: FGMRES with no preconditioner, tight tolerance.
// The coarsest grid is <=32x32 = <=1025 DOFs, so this is cheap.
// ============================================================
struct PCMGSetupData {
    std::vector<LevelPair> pairs;  // size nlevels-1
    std::vector<Mat>       R_mats; // restriction shells
    std::vector<Mat>       P_mats; // prolongation shells
};

static PetscErrorCode setupPCMG(PC pc, MGData& mg,
                                 PCMGSetupData& setup_data,
                                 MPI_Comm comm)
{
    int nlevels = mg.nlevels;

    PCSetType(pc, PCMG);
    PCMGSetLevels(pc, nlevels, NULL);
    PCMGSetType(pc, PC_MG_MULTIPLICATIVE);
    PCMGSetCycleType(pc, PC_MG_CYCLE_W);

    setup_data.pairs.resize(nlevels - 1);
    setup_data.R_mats.resize(nlevels - 1);
    setup_data.P_mats.resize(nlevels - 1);

    // --- CRITICAL FIX HERE ---
    // Mapping: PETSc Level 0 must be your coarsest level (mg.levels[nlevels-1])
    for (int lv = 0; lv < nlevels; lv++) {
        int pcmg_lv = (nlevels - 1) - lv; 
        PCMGSetOperators(pc, pcmg_lv, mg.levels[lv].mat, mg.levels[lv].mat);
    }

    for (int lv = 0; lv < nlevels - 1; lv++) {
        auto& pair     = setup_data.pairs[lv];
        pair.fine      = &mg.levels[lv].ctx;
        pair.coarse    = &mg.levels[lv+1].ctx;
        pair.arr[0]    = pair.fine;
        pair.arr[1]    = pair.coarse;

        PetscInt nf = pair.fine->n;
        PetscInt nc = pair.coarse->n;
        PetscInt lf_rows = pair.fine->r_end   - pair.fine->r_start;
        PetscInt lc_rows = pair.coarse->r_end  - pair.coarse->r_start;

        // Restriction: maps Fine -> Coarse
        MatCreateShell(comm, lc_rows, lf_rows, nc, nf,
                       pair.arr, &setup_data.R_mats[lv]);
        MatShellSetOperation(setup_data.R_mats[lv], MATOP_MULT,
                             (void(*)(void))restrictVec);

        // Prolongation: maps Coarse -> Fine
        MatCreateShell(comm, lf_rows, lc_rows, nf, nc,
                       pair.arr, &setup_data.P_mats[lv]);
        MatShellSetOperation(setup_data.P_mats[lv], MATOP_MULT,
                             (void(*)(void))prolongVec);

        // PETSc expects Restriction/Interpolation for Level L to map between L and L-1
        int pcmg_fine_idx = (nlevels - 1) - lv;
        PCMGSetRestriction(pc, pcmg_fine_idx, setup_data.R_mats[lv]);
        PCMGSetInterpolation(pc, pcmg_fine_idx, setup_data.P_mats[lv]);
    }

    // Configure smoothers (on all levels except the coarsest, which is PETSc level 0)
    for (int pcmg_lv = 1; pcmg_lv < nlevels; pcmg_lv++) {
        KSP smoother;
        PCMGGetSmoother(pc, pcmg_lv, &smoother);
        KSPSetType(smoother, KSPCHEBYSHEV);
        
        PC sub_pc;
        KSPGetPC(smoother, &sub_pc);
        PCSetType(sub_pc, PCJACOBI);
        KSPSetTolerances(smoother, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT, 2);

        KSPChebyshevEstEigSetUseNoisy(smoother, PETSC_TRUE);
        KSPChebyshevEstEigSet(smoother, 0.0, 0.1, 0.0, 1.3);
    }

    
    // Coarsest level (PETSc level 0)
    KSP coarse_ksp;
    PCMGGetCoarseSolve(pc, &coarse_ksp);

    // Use GMRES, which handles Shell matrices perfectly
    KSPSetType(coarse_ksp, KSPGMRES);
    KSPGMRESSetRestart(coarse_ksp, 200); // Allow it to "remember" up to 200 vectors

    PC coarse_pc;
    KSPGetPC(coarse_ksp, &coarse_pc);
    PCSetType(coarse_pc, PCJACOBI); // Jacobi is fine for the small coarse grid

    // Solve to a very tight tolerance (effectively direct)
    KSPSetTolerances(coarse_ksp, 1e-12, PETSC_DEFAULT, PETSC_DEFAULT, 200);

    return 0;
}

// ============================================================
// MAIN
// ============================================================
int main(int argc, char **argv)
{
    auto t_prog_start = Clock::now();

    // MKL threading settings (Windows-specific; harmless on Linux).
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

    // ---- Command-line options ----
    PetscInt X = 100, Y = 100;
    PetscOptionsGetInt(NULL, NULL, "-X", &X, NULL);
    PetscOptionsGetInt(NULL, NULL, "-Y", &Y, NULL);

    PetscBool save_pics = PETSC_FALSE;
    PetscOptionsGetBool(NULL, NULL, "-save_pics", &save_pics, NULL);

    // Coarsest grid threshold: stop coarsening when min(X,Y) <= this.
    // Default 32 gives a 32x32+1 = 1025 DOF coarse system.
    // Reduce to 16 if you want more levels (cheaper coarse solve,
    // more cycling overhead); increase to 64 if coarse solve is slow.
    PetscInt coarse_threshold = 32;
    PetscOptionsGetInt(NULL, NULL, "-mg_coarse_threshold",
                       &coarse_threshold, NULL);

    const PetscInt n = X * Y + 1;
    if (rank == 0)
        printf("  Grid: %d x %d  (n = %lld unknowns)\n",
               (int)X, (int)Y, (long long)n);

    // ---- Grid arrays ----
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

    // ---- Build multigrid hierarchy ----
    t0 = Clock::now();
    MGData mg = buildMGData(Rgrid.get(), X, Y, rank, nprocs,
                             PETSC_COMM_WORLD, (int)coarse_threshold);
    print_elapsed(rank, "MGData hierarchy built", t0, Clock::now());

    // ---- Vectors ----
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

    // ---- KSP + PCMG ----
    t0 = Clock::now();
    KSP ksp;
    KSPCreate(PETSC_COMM_WORLD, &ksp);
    KSPSetOperators(ksp, mg.levels[0].mat, mg.levels[0].mat);

    // Outer solver: CG is valid because the GMG preconditioner with
    // Chebyshev+Jacobi smoothers is a fixed SPD operator.
    // Use FGMRES if you switch smoothers to inner-CG in the future.
    KSPSetType(ksp, KSPCG);

    // Convergence: relative residual tolerance.
    // Tighten if you need more accurate V(x) for resistance extraction.
    KSPSetTolerances(ksp, 1e-8, PETSC_DEFAULT, PETSC_DEFAULT, 5000);

    PC pc;
    KSPGetPC(ksp, &pc);

    PCMGSetupData mg_setup;
    setupPCMG(pc, mg, mg_setup, PETSC_COMM_WORLD);

    // Allow command-line overrides for tuning without recompiling.
    // e.g.:  -ksp_type fgmres  to switch outer solver
    //        -pc_mg_cycle_type v  to use V-cycles
    //        -mg_levels_ksp_max_it 4  to increase smoothing steps
    KSPSetFromOptions(ksp);

    print_elapsed(rank, "KSP/PC setup", t0, Clock::now());

    if (rank == 0)
        printf("  Total init: %.1f ms\n\n",
               duration<double,std::milli>(Clock::now() - t_prog_start).count());

    // ---- Reassembly lambda ----
    // For matrix-free mode: just rebuild the coarsened Rgrids.
    // The fine-level MatShell already reads from Rgrid.get() directly.
    auto reassemble = [&]() {
        rebuildCoarseLevels(mg, Rgrid.get());
    };

    // ---- Temperature sweep setup ----
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

    // ---- HEATING LOOP ----
    size_t up_idx = 0;
    for (int step = 0; step <= total_steps; step++) {
        auto t_loop_start = Clock::now();
        double temp   = start_temp + (double)step;
        double ins_R  = getSemiconductorR(temp);

        for (PetscInt i = 0; i < X * Y; i++)
            Rgrid[i] = (temp >= Tgrid_up[i]) ? 1.0 : ins_R;

        auto t_asm = Clock::now();
        reassemble();
        auto t_slv = Clock::now();
        KSPSolve(ksp, b, x_vec);
        auto t_end = Clock::now();

        PetscScalar R_tot = 0.0;
        if (mg.levels[0].ctx.r_start == 0) {
            PetscInt idx = 0;
            VecGetValues(x_vec, 1, &idx, &R_tot);
        }
        MPI_Bcast(&R_tot, 1, MPI_DOUBLE, 0, PETSC_COMM_WORLD);

        if (rank == 0) {
            fprintf(f1, "%f %f\n", temp, PetscRealPart(R_tot));
            if (save_pics && up_idx < sched_up.size() &&
                temp >= sched_up[up_idx]) {
                char fn[64];
                sprintf(fn, "heat_%d.png", sched_up[up_idx]);
                save_rgrid_png(fn, Rgrid.get(), X, Y);
                up_idx++;
            }
            PetscInt its;
            KSPGetIterationNumber(ksp, &its);
            printf("Step %3d (H) | T:%.1f | R:%.4e | "
                   "Asm:%.1fms | Slv:%.1fms | Tot:%.1fms | It:%d\n",
                   step, temp, PetscRealPart(R_tot),
                   duration<double,std::milli>(t_asm - t_loop_start).count(),
                   duration<double,std::milli>(t_end - t_slv).count(),
                   duration<double,std::milli>(t_end - t_loop_start).count(),
                   (int)its);
        }
    }

    if (rank == 0) { fclose(f1); printf("\n>>> STARTING COOLING CYCLE <<<\n"); }
    size_t dn_idx = 0;

    // ---- COOLING LOOP ----
    for (int step = total_steps; step >= 0; step--) {
        auto t_loop_start = Clock::now();
        double temp  = start_temp + (double)step;
        double ins_R = getSemiconductorR(temp);

        for (PetscInt i = 0; i < X * Y; i++)
            Rgrid[i] = (temp < Tgrid_down[i]) ? ins_R : 1.0;

        auto t_asm = Clock::now();
        reassemble();
        auto t_slv = Clock::now();
        KSPSolve(ksp, b, x_vec);
        auto t_end = Clock::now();

        PetscScalar R_tot = 0.0;
        if (mg.levels[0].ctx.r_start == 0) {
            PetscInt idx = 0;
            VecGetValues(x_vec, 1, &idx, &R_tot);
        }
        MPI_Bcast(&R_tot, 1, MPI_DOUBLE, 0, PETSC_COMM_WORLD);

        if (rank == 0) {
            fprintf(f2, "%f %f\n", temp, PetscRealPart(R_tot));
            if (save_pics && dn_idx < sched_down.size() &&
                temp <= sched_down[dn_idx]) {
                char fn[64];
                sprintf(fn, "cool_%d.png", (int)temp);
                save_rgrid_png(fn, Rgrid.get(), X, Y);
                dn_idx++;
            }
            PetscInt its;
            KSPGetIterationNumber(ksp, &its);
            printf("Step %3d (C) | T:%.1f | R:%.4e | "
                   "Asm:%.1fms | Slv:%.1fms | Tot:%.1fms | It:%d\n",
                   step, temp, PetscRealPart(R_tot),
                   duration<double,std::milli>(t_asm - t_loop_start).count(),
                   duration<double,std::milli>(t_end - t_slv).count(),
                   duration<double,std::milli>(t_end - t_loop_start).count(),
                   (int)its);
        }
    }

    // ---- Cleanup ----
    if (rank == 0) fclose(f2);

    for (auto& R_mat : mg_setup.R_mats) MatDestroy(&R_mat);
    for (auto& P_mat : mg_setup.P_mats) MatDestroy(&P_mat);
    for (auto& lv : mg.levels) {
        MatDestroy(&lv.mat);
        if (lv.ctx.x_ghost) VecDestroy(&lv.ctx.x_ghost);
    }

    KSPDestroy(&ksp);
    VecDestroy(&b);
    VecDestroy(&x_vec);
    PetscFinalize();
    return 0;
}