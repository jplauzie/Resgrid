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

<<<<<<< Updated upstream
    printf("Precomputing Heating & Cooling Cascades (J=15.0)...\n");
=======
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
    gx.comm = comm; gx.tag = 42;
}

static void ghostExchangeExecute(LevelCtx& ctx) {
    PetscMPIInt nprocs; MPI_Comm_size(ctx.ghost_ex.comm, &nprocs);
    if (nprocs == 1) return;
    GhostExchange& gx = ctx.ghost_ex;
    for (auto& sc : gx.self_copies) cuda_memcpy_device_to_device(ctx.d_xlocal + sc.dst, ctx.d_xlocal + sc.src, sizeof(double));
    if (gx.reqs.empty()) return;
    for (auto& seg : gx.sends) launch_gather(seg.d_buf, ctx.d_xlocal, seg.d_indices, (int)seg.count, g_stream);
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
        // Fast path: bypass intermediate copy buffer entirely. Write straight to d_ya.
        launch_matvec(d_xa, d_ya, ctx->d_R, ctx->r_start, ctx->r_end, ctx->Xl, ctx->Yl,
                      ctx->source_local_idx, ctx->above_halo_start_global, ctx->above_halo_start_local,
                      ctx->below_halo_start_global, ctx->below_halo_start_local, g_stream);
    } else {
        cuda_memcpy_device_to_device(ctx->d_xlocal, d_xa, (size_t)lrows * sizeof(double));
        ghostExchangeExecute(*ctx);
        launch_matvec(ctx->d_xlocal, ctx->d_y, ctx->d_R, ctx->r_start, ctx->r_end, ctx->Xl, ctx->Yl,
                      ctx->source_local_idx, ctx->above_halo_start_global, ctx->above_halo_start_local,
                      ctx->below_halo_start_global, ctx->below_halo_start_local, g_stream);
        cuda_memcpy_device_to_device(d_ya, ctx->d_y, (size_t)lrows * sizeof(double));
    }
    VecCUDARestoreArrayRead(x, &d_xa);
    VecCUDARestoreArrayWrite(y, &d_ya);
    PetscLogFlops(7.0 * (ctx->r_end - ctx->r_start));
        
    return 0;
}

static PetscErrorCode levelGetDiagonal(Mat M, Vec diag) {
    LevelCtx* ctx; MatShellGetContext(M, (void**)&ctx);
    PetscScalar* d;
    VecCUDAGetArrayWrite(diag, &d);
    launch_diagonal(d, ctx->d_R, ctx->r_start, ctx->r_end, ctx->Xl, ctx->Yl, g_stream);
    VecCUDARestoreArrayWrite(diag, &d);
    PetscLogFlops(5.0 * (ctx->r_end - ctx->r_start));
    return 0;
}

static PetscErrorCode restrictVec(Mat Rmat, Vec xf, Vec xc) {
    IntergridCtx* ig; MatShellGetContext(Rmat, (void**)&ig);
    PetscMPIInt size; MPI_Comm_size(PETSC_COMM_WORLD, &size);

    const PetscScalar* d_xf; VecCUDAGetArrayRead(xf, &d_xf);
    PetscScalar* d_xc;       VecCUDAGetArrayWrite(xc, &d_xc);

    if (size == 1) {
        launch_restrict(d_xf, d_xc, ig->coarse->r_start, ig->coarse->r_end,
                        ig->fine->Xl, ig->fine->Yl, ig->coarse->Yl, ig->fine_seq_offset, g_stream);
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
                       ig->coarse_seq_offset, ig->coarse_seq_ic_end, g_stream);
    } else {
        // [Fallback MPI Scatter Logic]
    }

    VecCUDARestoreArrayRead(xc, &d_xc);
    VecCUDARestoreArrayWrite(xf, &d_xf);
    return 0;
}

// ============================================================
// Explicit CPU Matrix Assembly (for coarse grids)
// ============================================================
static inline double cpu_gmean(double R1, double R2) {
    return 2.0 / (R1 + R2);
}

static void update_coarse_matrix_cpu(LevelCtx& ctx, Mat A) {
    PetscInt Xl = ctx.Xl;
    PetscInt Yl = ctx.Yl;
    PetscInt N = ctx.n;  // Xl*Yl + 1
    
    // Zero out the matrix before re-filling it
    MatZeroEntries(A);
    
    // Max nonzeros per row: Yl+1 for source row, 5 for interior grid rows
    std::vector<PetscInt> cols(Yl + 1);  // Large enough for worst case
    std::vector<PetscScalar> vals(Yl + 1);
    
    for (PetscInt r = 0; r < N; r++) {
        PetscInt ncols = 0;
        double diag = 0.0;
        
        if (r == 0) {
            // Source row: connects to all nodes in first row of grid (indices 1..Yl)
            for (PetscInt j = 0; j < Yl; j++) {
                double gv = 2.0 / ctx.h_R[j];
                cols[ncols] = 1 + j;
                vals[ncols++] = -gv;
                diag += gv;
            }
            cols[ncols] = 0;
            vals[ncols++] = diag;
        } else {
            PetscInt gi = r - 1;
            PetscInt i = gi / Yl;
            PetscInt j = gi % Yl;
            
            // Up connection
            if (i == 0) {
                double gv = 2.0 / ctx.h_R[gi];
                cols[ncols] = 0;
                vals[ncols++] = -gv;
                diag += gv;
            } else {
                double gv = cpu_gmean(ctx.h_R[gi], ctx.h_R[(i-1)*Yl + j]);
                cols[ncols] = r - Yl;
                vals[ncols++] = -gv;
                diag += gv;
            }
            // Left connection
            if (j > 0) {
                double gv = cpu_gmean(ctx.h_R[gi], ctx.h_R[i*Yl + (j-1)]);
                cols[ncols] = r - 1;
                vals[ncols++] = -gv;
                diag += gv;
            }
            // Right connection
            if (j < Yl - 1) {
                double gv = cpu_gmean(ctx.h_R[gi], ctx.h_R[i*Yl + (j+1)]);
                cols[ncols] = r + 1;
                vals[ncols++] = -gv;
                diag += gv;
            }
            // Down connection
            if (i < Xl - 1) {
                double gv = cpu_gmean(ctx.h_R[gi], ctx.h_R[(i+1)*Yl + j]);
                cols[ncols] = r + Yl;
                vals[ncols++] = -gv;
                diag += gv;
            } else {
                diag += 2.0 / ctx.h_R[gi];
            }
            
            cols[ncols] = r;
            vals[ncols++] = diag;
        }
        
        MatSetValues(A, 1, &r, ncols, cols.data(), vals.data(), INSERT_VALUES);
    }
    
    MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);
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
    // buildGhostExchange(ctx, g, comm);  // This line was incomplete - commented out for now
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
    
    // Grids 50x50 or smaller - use explicit CPU matrix for direct solve
    if (std::max(lv.ctx.Xl, lv.ctx.Yl) <= 50) {
        lv.ctx.use_explicit_cpu_mat = true;
        // Allocate for R values only (Xl * Yl)
        lv.ctx.h_R = new double[lv.ctx.Xl * lv.ctx.Yl];
        
        MatCreate(comm, &lv.mat);
        MatSetSizes(lv.mat, PETSC_DECIDE, PETSC_DECIDE, lv.ctx.n, lv.ctx.n);
        MatSetType(lv.mat, MATAIJ); // Standard CPU sparse matrix
        
        // Calculate proper preallocation for each row
        // Row 0 (source) connects to Yl nodes + itself -> Yl+1 non-zeros
        // Interior rows connect to up to 4 neighbors + itself -> 5 non-zeros
        // Boundary rows have fewer connections
        PetscInt d_nz = 5;  // Default for interior points
        PetscInt d_nz_row0 = lv.ctx.Yl + 1;  // Source row has many connections
        
        // For sequential case
        MatSeqAIJSetPreallocation(lv.mat, d_nz_row0, NULL);
        // But we need per-row preallocation for the source row
        // Alternative: just disable the error and let it allocate dynamically
        MatSetOption(lv.mat, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE);
        
        // For MPI case (though nprocs=1 here)
        MatMPIAIJSetPreallocation(lv.mat, d_nz, NULL, d_nz, NULL);
    } else {
        lv.ctx.use_explicit_cpu_mat = false;
        MatCreateShell(comm, lr, lr, lv.ctx.n, lv.ctx.n, &lv.ctx, &lv.mat);
        MatSetVecType(lv.mat, VECCUDA);
        MatShellSetOperation(lv.mat, MATOP_MULT, (void(*)(void))levelMatvec);
        MatShellSetOperation(lv.mat, MATOP_GET_DIAGONAL, (void(*)(void))levelGetDiagonal);
    }
}

static MGData buildMGData(const double* fine_Rgrid, PetscInt X, PetscInt Y, PetscMPIInt rank, PetscMPIInt nprocs, MPI_Comm comm, int coarsen_threshold=32, int lu_threshold=50) {
    MGData mg;
    std::vector<std::pair<PetscInt,PetscInt>> sizes;
    PetscInt cx=X, cy=Y;
    
    // 1. Coarsen until min dimension <= coarsen_threshold (e.g., 32x32 or smaller)
    while (true) {
        sizes.push_back({cx,cy});
        if (std::min(cx,cy) <= coarsen_threshold) break;
        cx = (cx + 1) / 2;
        cy = (cy + 1) / 2;
    }
    mg.nlevels = (int)sizes.size();
    mg.levels.resize(mg.nlevels);

    if (rank==0) {
        printf("  GMG: %d levels, coarsest %dx%d\n", mg.nlevels, (int)sizes.back().first, (int)sizes.back().second);
        for (int lv=0; lv<mg.nlevels; lv++) {
            printf("    Level %d: %dx%d (%lld DOFs) - ", 
                lv, (int)sizes[lv].first, (int)sizes[lv].second, 
                (long long)(sizes[lv].first * sizes[lv].second + 1));
            if (std::max(sizes[lv].first, sizes[lv].second) <= lu_threshold) {
                printf("CPU Matrix (LU)\n");
            } else {
                printf("GPU Matrix-Free (iterative)\n");
            }
        }
    }

    // 2. Initialize each level
    for (int lv=0; lv<mg.nlevels; lv++) {
        auto& L  = mg.levels[lv];
        L.ctx.Xl = sizes[lv].first; 
        L.ctx.Yl = sizes[lv].second; 
        L.ctx.n  = L.ctx.Xl * L.ctx.Yl + 1;
        
        getOwnershipRange(L.ctx.n, rank, nprocs, L.ctx.r_start, L.ctx.r_end);
        
        // Use explicit CPU matrix for grids <= lu_threshold (50)
        // But ensure we never go below 2x2 for CPU matrices (1x1 is problematic)
        bool use_cpu = (std::max(L.ctx.Xl, L.ctx.Yl) <= lu_threshold) && (L.ctx.Xl * L.ctx.Yl >= 4);
        if (use_cpu) {
            L.ctx.use_explicit_cpu_mat = true;
            L.ctx.h_R = new double[L.ctx.Xl * L.ctx.Yl];
        } else {
            L.ctx.use_explicit_cpu_mat = false;
        }

        buildGhostVec(L.ctx, comm);
        buildLevelMat(L, comm);
    }

    // 3. Populate Rgrid and upload
    mg.levels[0].ctx.Rgrid.assign(fine_Rgrid, fine_Rgrid + X*Y);
    for (int lv=1; lv<mg.nlevels; lv++) {
        auto& p = mg.levels[lv-1].ctx;
        auto& c = mg.levels[lv].ctx;
        c.Rgrid = coarsenRgrid(p.Rgrid.data(), p.Xl, p.Yl, c.Xl, c.Yl);
    }

    for (int lv=0; lv<mg.nlevels; lv++) {
        auto& ctx = mg.levels[lv].ctx;
        cuda_memcpy_to_device(ctx.d_R, ctx.Rgrid.data(), (size_t)(ctx.Xl*ctx.Yl)*sizeof(double));
        if (ctx.use_explicit_cpu_mat) {
            memcpy(ctx.h_R, ctx.Rgrid.data(), (size_t)(ctx.Xl * ctx.Yl) * sizeof(double));
        }
    }
    
    return mg;
}

static void rebuildCoarseLevelsGPU(MGData& mg) {
    for (int lv=1; lv<mg.nlevels; lv++) {
        auto& p = mg.levels[lv-1].ctx;
        auto& c = mg.levels[lv].ctx;
        launch_coarsen(p.d_R, c.d_R, p.Xl, p.Yl, c.Xl, c.Yl, g_stream);
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

    for (int lv=0; lv<nlevels; lv++) {
        PCMGSetOperators(pc, (nlevels-1)-lv, mg.levels[lv].mat, mg.levels[lv].mat);
    }

    for (int lv=0; lv<nlevels-1; lv++) {
        int pcmg_fine = (nlevels-1)-lv;
        LevelCtx* fine_ctx   = &mg.levels[lv].ctx;
        LevelCtx* coarse_ctx = &mg.levels[lv+1].ctx;
        IntergridCtx* ig = buildIntergridCtx(fine_ctx, coarse_ctx, comm);
        setup.ig_ctxs[lv] = ig;

        PetscInt nf=fine_ctx->n, lf=fine_ctx->r_end - fine_ctx->r_start;
        PetscInt nc=coarse_ctx->n, lc=coarse_ctx->r_end - coarse_ctx->r_start;

        MatCreateShell(comm, lc, lf, nc, nf, ig, &setup.R_mats[lv]);
        MatSetVecType(setup.R_mats[lv], VECCUDA);
        MatShellSetOperation(setup.R_mats[lv], MATOP_MULT, (void(*)(void))restrictVec);

        MatCreateShell(comm, lf, lc, nf, nc, ig, &setup.P_mats[lv]);
        MatSetVecType(setup.P_mats[lv], VECCUDA);
        MatShellSetOperation(setup.P_mats[lv], MATOP_MULT, (void(*)(void))prolongVec);

        PCMGSetRestriction (pc, pcmg_fine, setup.R_mats[lv]);
        PCMGSetInterpolation(pc, pcmg_fine, setup.P_mats[lv]);
    }
    
    // Configure each level based on matrix type
    for (int lv = 0; lv < nlevels; lv++) {
        int pcmg_level = (nlevels - 1) - lv;
        KSP smoother;
        PCMGGetSmoother(pc, pcmg_level, &smoother);
        
        if (mg.levels[lv].ctx.use_explicit_cpu_mat) {
            // CPU matrices (grids ≤50x50) - use direct LU solve
            PC smoother_pc;
            KSPGetPC(smoother, &smoother_pc);
            PCSetType(smoother_pc, PCLU);
            KSPSetType(smoother, KSPPREONLY);
        } else {
            // Shell matrices (grids >50x50, matrix-free GPU) - use Chebyshev/Jacobi
            PC smoother_pc;
            KSPGetPC(smoother, &smoother_pc);
            PCSetType(smoother_pc, PCJACOBI);
            KSPSetType(smoother, KSPCHEBYSHEV);
            
            // Set maximum iterations - use KSPSetTolerances for max iterations
            KSPSetTolerances(smoother, 1e-2, PETSC_DEFAULT, PETSC_DEFAULT, 10);
            
            // The rest of the Chebyshev settings should be done via command line
            // since they are specific to the KSP type
        }
    }
    
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
>>>>>>> Stashed changes
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