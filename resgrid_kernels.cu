#include "resgrid_kernels.h"
#include <cuda_runtime.h>
#include <stdio.h>

// ============================================================
// Device helper: harmonic mean conductance
// ============================================================
__device__ static inline double d_gmean(double R1, double R2)
{
    return 2.0 / (R1 + R2);
}

// ============================================================
// Kernel: matrix-free matvec  y = A * x
//
// Each thread handles one global row r in [r_start, r_end).
// xlocal is the ghost-extended local vector (already scatter-updated).
// R is the device copy of Rgrid.
// ============================================================
__global__ void matvec_kernel(
    const double* __restrict__ xlocal,   // ghost-extended, length = lrows + nghosts
    double*       __restrict__ y,         // output, length = lrows
    const double* __restrict__ R,         // Rgrid on device, length = Xl*Yl
    PetscInt r_start, PetscInt r_end,
    PetscInt Xl, PetscInt Yl,
    // ghost mapping offsets (mirrors LevelCtx)
    PetscInt source_local_idx,
    PetscInt above_halo_start_global,
    PetscInt above_halo_start_local,
    PetscInt below_halo_start_global,
    PetscInt below_halo_start_local)
{
    PetscInt r = r_start + blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= r_end) return;

    PetscInt lr = r - r_start;

    // Inline globalToLocal
    auto g2l = [&](PetscInt global) -> PetscInt {
        if (global >= r_start && global < r_end)
            return global - r_start;
        if (global == 0)
            return source_local_idx;
        if (global < r_start)
            return above_halo_start_local + (global - above_halo_start_global);
        return below_halo_start_local + (global - below_halo_start_global);
    };

    double diag = 0.0, val = 0.0;

    if (r == 0) {
        // Source node: connected to all nodes in first row (j = 0..Yl-1)
        for (PetscInt j = 0; j < Yl; j++) {
            double gv = 2.0 / R[j];
            val  += -gv * xlocal[g2l(j + 1)];
            diag += gv;
        }
    } else {
        PetscInt gi = r - 1;
        PetscInt i  = gi / Yl;
        PetscInt j  = gi % Yl;

        // Up neighbor (row i-1, or source node if i==0)
        if (i == 0) {
            double gv = 2.0 / R[gi];
            val  += -gv * xlocal[g2l(0)];
            diag += gv;
        } else {
            double gv = d_gmean(R[gi], R[(i-1)*Yl + j]);
            val  += -gv * xlocal[g2l(r - Yl)];
            diag += gv;
        }

        // Left neighbor
        if (j > 0) {
            double gv = d_gmean(R[gi], R[i*Yl + (j-1)]);
            val  += -gv * xlocal[g2l(r - 1)];
            diag += gv;
        }

        // Right neighbor
        if (j < Yl - 1) {
            double gv = d_gmean(R[gi], R[i*Yl + (j+1)]);
            val  += -gv * xlocal[g2l(r + 1)];
            diag += gv;
        }

        // Down neighbor (row i+1, or boundary)
        if (i < Xl - 1) {
            double gv = d_gmean(R[gi], R[(i+1)*Yl + j]);
            val  += -gv * xlocal[g2l(r + Yl)];
            diag += gv;
        } else {
            diag += 2.0 / R[gi];
        }
    }

    y[lr] = diag * xlocal[lr] + val;
}

// ============================================================
// Kernel: diagonal extraction  d[r] = A[r,r]
// ============================================================
__global__ void diagonal_kernel(
    double* __restrict__ d,
    const double* __restrict__ R,
    PetscInt r_start, PetscInt r_end,
    PetscInt Xl, PetscInt Yl)
{
    PetscInt r = r_start + blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= r_end) return;
    PetscInt lr = r - r_start;

    double v = 0.0;
    if (r == 0) {
        for (PetscInt j = 0; j < Yl; j++) v += 2.0 / R[j];
    } else {
        PetscInt gi = r - 1, i = gi / Yl, j = gi % Yl;
        if (i == 0)    v += 2.0 / R[gi];
        else           v += d_gmean(R[gi], R[(i-1)*Yl + j]);
        if (j > 0)     v += d_gmean(R[gi], R[i*Yl + (j-1)]);
        if (j < Yl-1)  v += d_gmean(R[gi], R[i*Yl + (j+1)]);
        if (i < Xl-1)  v += d_gmean(R[gi], R[(i+1)*Yl + j]);
        else            v += 2.0 / R[gi];
    }
    d[lr] = v;
}

// ============================================================
// Kernel: restriction  xc = R_op * xf
// Each thread handles one coarse row rc in [coarse_r_start, coarse_r_end).
// fseq is the gathered fine local sequence (on device).
// ============================================================
__global__ void restrict_kernel(
    const double* __restrict__ fseq,
    double*       __restrict__ xca,
    PetscInt coarse_r_start, PetscInt coarse_r_end,
    PetscInt fine_Xl, PetscInt fine_Yl,
    PetscInt coarse_Yl,
    PetscInt fine_seq_offset)
{
    PetscInt rc = coarse_r_start + blockIdx.x * blockDim.x + threadIdx.x;
    if (rc >= coarse_r_end) return;
    PetscInt lc = rc - coarse_r_start;

    if (rc == 0) {
        xca[lc] = fseq[0];
        return;
    }

    PetscInt gc = rc - 1;
    PetscInt ic = gc / coarse_Yl, jc = gc % coarse_Yl;
    PetscInt if0 = 2 * ic;
    PetscInt if1 = (2*ic+1 < fine_Xl) ? 2*ic+1 : fine_Xl-1;
    PetscInt jf0 = 2 * jc;
    PetscInt jf1 = (2*jc+1 < fine_Yl) ? 2*jc+1 : fine_Yl-1;

    auto fget = [&](PetscInt ri, PetscInt rj) -> double {
        PetscInt gi = ri * fine_Yl + rj + 1;
        return fseq[gi - fine_seq_offset + 1];
    };

    int cnt = 0;
    double sum = 0.0;
    sum += fget(if0, jf0); cnt++;
    if (jf1 != jf0) { sum += fget(if0, jf1); cnt++; }
    if (if1 != if0) { sum += fget(if1, jf0); cnt++; }
    if (if1 != if0 && jf1 != jf0) { sum += fget(if1, jf1); cnt++; }

    xca[lc] = sum / (double)cnt;
}

// ============================================================
// Kernel: prolongation  xf = P_op * xc
// Each thread handles one fine row rf in [fine_r_start, fine_r_end).
// cseq is the gathered coarse local sequence (on device).
// ============================================================
__global__ void prolong_kernel(
    const double* __restrict__ cseq,
    double*       __restrict__ xfa,
    PetscInt fine_r_start, PetscInt fine_r_end,
    PetscInt fine_Yl, PetscInt coarse_Yl, PetscInt coarse_Xl,
    PetscInt coarse_seq_offset,
    PetscInt coarse_seq_ic_end)
{
    PetscInt rf = fine_r_start + blockIdx.x * blockDim.x + threadIdx.x;
    if (rf >= fine_r_end) return;
    PetscInt lf = rf - fine_r_start;

    if (rf == 0) {
        xfa[lf] = cseq[0];
        return;
    }

    PetscInt gf  = rf - 1;
    PetscInt iff = gf / fine_Yl;
    PetscInt jf  = gf % fine_Yl;

    PetscInt ic = iff / 2;
    PetscInt jc = jf  / 2;

    double wi0 = (iff % 2 == 0) ? 1.0 : 0.5;
    double wi1 = (iff % 2 == 0) ? 0.0 : 0.5;
    double wj0 = (jf  % 2 == 0) ? 1.0 : 0.5;
    double wj1 = (jf  % 2 == 0) ? 0.0 : 0.5;

    PetscInt ic1 = (ic + 1 < coarse_Xl) ? ic + 1 : coarse_Xl - 1;
    PetscInt jc1 = (jc + 1 < coarse_Yl) ? jc + 1 : coarse_Yl - 1;

    auto cget = [&](PetscInt gi) -> double {
        if (gi == 0) return cseq[0];
        return cseq[1 + (gi - coarse_seq_offset)];
    };

    PetscInt g00 = ic  * coarse_Yl + jc  + 1;
    PetscInt g01 = ic  * coarse_Yl + jc1 + 1;
    PetscInt g10 = ic1 * coarse_Yl + jc  + 1;
    PetscInt g11 = ic1 * coarse_Yl + jc1 + 1;

    xfa[lf] = wi0*wj0 * cget(g00)
            + wi0*wj1 * cget(g01)
            + wi1*wj0 * cget(g10)
            + wi1*wj1 * cget(g11);
}

// ============================================================
// Kernel: coarsen Rgrid 2x
// Each thread handles one coarse cell (ic, jc).
// ============================================================
__global__ void coarsen_kernel(
    const double* __restrict__ Rf,
    double*       __restrict__ Rc,
    PetscInt Xf, PetscInt Yf,
    PetscInt Xc, PetscInt Yc)
{
    PetscInt ic = blockIdx.x * blockDim.x + threadIdx.x;
    PetscInt jc = blockIdx.y * blockDim.y + threadIdx.y;
    if (ic >= Xc || jc >= Yc) return;

    PetscInt if0 = 2*ic, if1 = (2*ic+1 < Xf) ? 2*ic+1 : Xf-1;
    PetscInt jf0 = 2*jc, jf1 = (2*jc+1 < Yf) ? 2*jc+1 : Yf-1;

    int cnt = 0; double si = 0.0;
    si += 1.0 / Rf[if0*Yf + jf0]; cnt++;
    if (jf1 != jf0) { si += 1.0 / Rf[if0*Yf + jf1]; cnt++; }
    if (if1 != if0) { si += 1.0 / Rf[if1*Yf + jf0]; cnt++; }
    if (if1 != if0 && jf1 != jf0) { si += 1.0 / Rf[if1*Yf + jf1]; cnt++; }

    Rc[ic*Yc + jc] = (double)cnt / si;
}

// ============================================================
// Host-callable launcher functions
// ============================================================

void launch_matvec(
    const double* xlocal, double* y, const double* d_R,
    PetscInt r_start, PetscInt r_end,
    PetscInt Xl, PetscInt Yl,
    PetscInt source_local_idx,
    PetscInt above_halo_start_global, PetscInt above_halo_start_local,
    PetscInt below_halo_start_global, PetscInt below_halo_start_local)
{
    PetscInt lrows = r_end - r_start;
    int threads = 256;
    int blocks  = (lrows + threads - 1) / threads;
    matvec_kernel<<<blocks, threads>>>(
        xlocal, y, d_R,
        r_start, r_end, Xl, Yl,
        source_local_idx,
        above_halo_start_global, above_halo_start_local,
        below_halo_start_global, below_halo_start_local);
    cudaDeviceSynchronize();
}

void launch_diagonal(
    double* d, const double* d_R,
    PetscInt r_start, PetscInt r_end,
    PetscInt Xl, PetscInt Yl)
{
    PetscInt lrows = r_end - r_start;
    int threads = 256;
    int blocks  = (lrows + threads - 1) / threads;
    diagonal_kernel<<<blocks, threads>>>(d, d_R, r_start, r_end, Xl, Yl);
    cudaDeviceSynchronize();
}

void launch_restrict(
    const double* fseq, double* xca,
    PetscInt coarse_r_start, PetscInt coarse_r_end,
    PetscInt fine_Xl, PetscInt fine_Yl, PetscInt coarse_Yl,
    PetscInt fine_seq_offset)
{
    PetscInt lrows = coarse_r_end - coarse_r_start;
    int threads = 256;
    int blocks  = (lrows + threads - 1) / threads;
    restrict_kernel<<<blocks, threads>>>(
        fseq, xca,
        coarse_r_start, coarse_r_end,
        fine_Xl, fine_Yl, coarse_Yl,
        fine_seq_offset);
    cudaDeviceSynchronize();
}

void launch_prolong(
    const double* cseq, double* xfa,
    PetscInt fine_r_start, PetscInt fine_r_end,
    PetscInt fine_Yl, PetscInt coarse_Yl, PetscInt coarse_Xl,
    PetscInt coarse_seq_offset, PetscInt coarse_seq_ic_end)
{
    PetscInt lrows = fine_r_end - fine_r_start;
    int threads = 256;
    int blocks  = (lrows + threads - 1) / threads;
    prolong_kernel<<<blocks, threads>>>(
        cseq, xfa,
        fine_r_start, fine_r_end,
        fine_Yl, coarse_Yl, coarse_Xl,
        coarse_seq_offset, coarse_seq_ic_end);
    cudaDeviceSynchronize();
}

void launch_coarsen(
    const double* d_Rf, double* d_Rc,
    PetscInt Xf, PetscInt Yf, PetscInt Xc, PetscInt Yc)
{
    dim3 threads(16, 16);
    dim3 blocks((Xc + 15) / 16, (Yc + 15) / 16);
    coarsen_kernel<<<blocks, threads>>>(d_Rf, d_Rc, Xf, Yf, Xc, Yc);
    cudaDeviceSynchronize();
}

void cuda_malloc(void** ptr, size_t size)
{
    cudaError_t err = cudaMalloc(ptr, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
    }
}

void cuda_free(void* ptr)
{
    if (ptr) cudaFree(ptr);
}

void cuda_memcpy_to_device(void* dst, const void* src, size_t size)
{
    cudaError_t err = cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy H2D failed: %s\n", cudaGetErrorString(err));
    }
}
