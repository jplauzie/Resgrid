#include "resgrid_kernels.h"
#include <cuda_runtime.h>
#include <stdio.h>

__device__ static inline double d_gmean(double R1, double R2) {
    return 2.0 / (R1 + R2);
}

// === Matvec Kernel ===
__global__ void matvec_kernel(
    const double* __restrict__ xlocal, double* __restrict__ y, const double* __restrict__ R,
    PetscInt r_start, PetscInt r_end, PetscInt Xl, PetscInt Yl,
    PetscInt source_local_idx,
    PetscInt above_halo_start_global, PetscInt above_halo_start_local,
    PetscInt below_halo_start_global, PetscInt below_halo_start_local)
{
    PetscInt r = r_start + (PetscInt)(blockIdx.x * blockDim.x + threadIdx.x);
    if (r >= r_end) return;

    PetscInt lr = r - r_start;

    auto g2l = [&](PetscInt global) -> PetscInt {
        if (global >= r_start && global < r_end) return global - r_start;
        if (global == 0) return source_local_idx;
        if (global < r_start) return above_halo_start_local + (global - above_halo_start_global);
        return below_halo_start_local + (global - below_halo_start_global);
    };

    double diag = 0.0, val = 0.0;

    if (r == 0) {
        for (PetscInt j = 0; j < Yl; j++) {
            double gv = 2.0 / __ldg(&R[j]);
            val  += -gv * xlocal[g2l(j + 1)];
            diag += gv;
        }
    } else {
        PetscInt gi = r - 1, i  = gi / Yl, j  = gi % Yl;
        if (i == 0) {
            double gv = 2.0 / __ldg(&R[gi]);
            val  += -gv * xlocal[g2l(0)];
            diag += gv;
        } else {
            double gv = d_gmean(__ldg(&R[gi]), __ldg(&R[(i-1)*Yl + j]));
            val  += -gv * xlocal[g2l(r - Yl)];
            diag += gv;
        }
        if (j > 0) {
            double gv = d_gmean(__ldg(&R[gi]), __ldg(&R[i*Yl + (j-1)]));
            val  += -gv * xlocal[g2l(r - 1)];
            diag += gv;
        }
        if (j < Yl - 1) {
            double gv = d_gmean(__ldg(&R[gi]), __ldg(&R[i*Yl + (j+1)]));
            val  += -gv * xlocal[g2l(r + 1)];
            diag += gv;
        }
        if (i < Xl - 1) {
            double gv = d_gmean(__ldg(&R[gi]), __ldg(&R[(i+1)*Yl + j]));
            val  += -gv * xlocal[g2l(r + Yl)];
            diag += gv;
        } else {
            diag += 2.0 / __ldg(&R[gi]);
        }
    }
    y[lr] = diag * xlocal[lr] + val;
}

// === Diagonal Kernel ===
__global__ void diagonal_kernel(
    double* __restrict__ d, const double* __restrict__ R,
    PetscInt r_start, PetscInt r_end, PetscInt Xl, PetscInt Yl)
{
    PetscInt r = r_start + (PetscInt)(blockIdx.x * blockDim.x + threadIdx.x);
    if (r >= r_end) return;
    PetscInt lr = r - r_start;
    double v = 0.0;
    if (r == 0) {
        for (PetscInt j = 0; j < Yl; j++) v += 2.0 / __ldg(&R[j]);
    } else {
        PetscInt gi = r - 1, i = gi / Yl, j = gi % Yl;
        if (i == 0)    v += 2.0 / __ldg(&R[gi]);
        else           v += d_gmean(__ldg(&R[gi]), __ldg(&R[(i-1)*Yl + j]));
        if (j > 0)     v += d_gmean(__ldg(&R[gi]), __ldg(&R[i*Yl + (j-1)]));
        if (j < Yl-1)  v += d_gmean(__ldg(&R[gi]), __ldg(&R[i*Yl + (j+1)]));
        if (i < Xl-1)  v += d_gmean(__ldg(&R[gi]), __ldg(&R[(i+1)*Yl + j]));
        else           v += 2.0 / __ldg(&R[gi]);
    }
    d[lr] = v;
}

// === Restrict Kernel ===
__global__ void restrict_kernel(
    const double* __restrict__ fseq, double* __restrict__ xca,
    PetscInt coarse_r_start, PetscInt coarse_r_end,
    PetscInt fine_Xl, PetscInt fine_Yl, PetscInt coarse_Yl, PetscInt fine_seq_offset)
{
    PetscInt rc = coarse_r_start + (PetscInt)(blockIdx.x * blockDim.x + threadIdx.x);
    if (rc >= coarse_r_end) return;
    PetscInt lc = rc - coarse_r_start;

    if (rc == 0) { xca[lc] = fseq[0]; return; }
    PetscInt gc = rc - 1, ic = gc / coarse_Yl, jc = gc % coarse_Yl;
    PetscInt if0 = 2 * ic, if1 = (2*ic+1 < fine_Xl) ? 2*ic+1 : fine_Xl-1;
    PetscInt jf0 = 2 * jc, jf1 = (2*jc+1 < fine_Yl) ? 2*jc+1 : fine_Yl-1;

    auto fget = [&](PetscInt ri, PetscInt rj) -> double {
        PetscInt gi = ri * fine_Yl + rj + 1;
        if (gi == 0) return fseq[0];
        return fseq[gi - fine_seq_offset + 1];
    };

    int cnt = 0; double sum = 0.0;
    sum += fget(if0, jf0); cnt++;
    if (jf1 != jf0) { sum += fget(if0, jf1); cnt++; }
    if (if1 != if0) { sum += fget(if1, jf0); cnt++; }
    if (if1 != if0 && jf1 != jf0) { sum += fget(if1, jf1); cnt++; }
    xca[lc] = sum / (double)cnt;
}

// === Prolong Kernel ===
__global__ void prolong_kernel(
    const double* __restrict__ cseq, double* __restrict__ xfa,
    PetscInt fine_r_start, PetscInt fine_r_end,
    PetscInt fine_Yl, PetscInt coarse_Yl, PetscInt coarse_Xl,
    PetscInt coarse_seq_offset, PetscInt coarse_seq_ic_end)
{
    PetscInt rf = fine_r_start + (PetscInt)(blockIdx.x * blockDim.x + threadIdx.x);
    if (rf >= fine_r_end) return;
    PetscInt lf = rf - fine_r_start;

    if (rf == 0) { xfa[lf] = cseq[0]; return; }
    PetscInt gf = rf - 1, iff = gf / fine_Yl, jf = gf % fine_Yl;
    PetscInt ic = iff / 2, jc = jf  / 2;

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

    PetscInt g00 = ic*coarse_Yl + jc + 1, g01 = ic*coarse_Yl + jc1 + 1;
    PetscInt g10 = ic1*coarse_Yl + jc + 1, g11 = ic1*coarse_Yl + jc1 + 1;
    xfa[lf] = wi0*wj0*cget(g00) + wi0*wj1*cget(g01) + wi1*wj0*cget(g10) + wi1*wj1*cget(g11);
}

// === Coarsen Kernel ===
// === Coarsen Kernel with Arithmetic Mean ===
__global__ void coarsen_kernel(
    const double* __restrict__ Rf, double* __restrict__ Rc,
    PetscInt Xf, PetscInt Yf, PetscInt Xc, PetscInt Yc)
{
    PetscInt ic = (PetscInt)(blockIdx.x * blockDim.x + threadIdx.x);
    PetscInt jc = (PetscInt)(blockIdx.y * blockDim.y + threadIdx.y);
    if (ic >= Xc || jc >= Yc) return;

    PetscInt if0 = 2 * ic;
    PetscInt if1 = (2 * ic + 1 < Xf) ? 2 * ic + 1 : Xf - 1;
    PetscInt jf0 = 2 * jc;
    PetscInt jf1 = (2 * jc + 1 < Yf) ? 2 * jc + 1 : Yf - 1;

    // Arithmetic mean: (R1 + R2 + ... + Rn) / n
    double sum = 0.0;
    int cnt = 0;

    sum += __ldg(&Rf[if0 * Yf + jf0]);
    cnt++;

    if (jf1 != jf0) {
        sum += __ldg(&Rf[if0 * Yf + jf1]);
        cnt++;
    }

    if (if1 != if0) {
        sum += __ldg(&Rf[if1 * Yf + jf0]);
        cnt++;
    }

    if (if1 != if0 && jf1 != jf0) {
        sum += __ldg(&Rf[if1 * Yf + jf1]);
        cnt++;
    }

    Rc[ic * Yc + jc] = sum / (double)cnt;
}
// === NEW: Update Rgrid Kernel ===
__global__ void update_rgrid_kernel(
    double* __restrict__ d_R, const double* __restrict__ d_Tthresh, 
    double temp, double ins_R, PetscInt N, bool is_heating)
{
    PetscInt i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        if (is_heating) d_R[i] = (temp >= d_Tthresh[i]) ? 1.0 : ins_R;
        else            d_R[i] = (temp <  d_Tthresh[i]) ? ins_R : 1.0;
    }
}

// === Gather Kernel ===
__global__ void gather_kernel(
    double* __restrict__ d_dst, const double* __restrict__ d_src,
    const int* __restrict__ d_indices, int count)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) d_dst[i] = d_src[d_indices[i]];
}

// ============================================================
// Launch Wrappers
// ============================================================
extern "C" {

void launch_matvec(
    const double* xlocal_dev, double* y_dev, const double* d_R,
    PetscInt r_start, PetscInt r_end, PetscInt Xl, PetscInt Yl,
    PetscInt source_local_idx, PetscInt above_halo_start_global, 
    PetscInt above_halo_start_local, PetscInt below_halo_start_global, 
    PetscInt below_halo_start_local, cudaStream_t stream) // Added stream
{
    PetscInt lrows = r_end - r_start;
    int threads = 256;
    int blocks  = ((int)lrows + threads - 1) / threads;
    matvec_kernel<<<blocks, threads, 0, stream>>>( // Use stream
        xlocal_dev, y_dev, d_R, r_start, r_end, Xl, Yl,
        source_local_idx, above_halo_start_global, above_halo_start_local,
        below_halo_start_global, below_halo_start_local);
}

void launch_diagonal(
    double* d_dev, const double* d_R,
    PetscInt r_start, PetscInt r_end, PetscInt Xl, PetscInt Yl, 
    cudaStream_t stream) // Added stream
{
    PetscInt lrows = r_end - r_start;
    int threads = 256;
    int blocks  = ((int)lrows + threads - 1) / threads;
    diagonal_kernel<<<blocks, threads, 0, stream>>>(d_dev, d_R, r_start, r_end, Xl, Yl);
}

void launch_restrict(
    const double* fseq_dev, double* xca_dev,
    PetscInt coarse_r_start, PetscInt coarse_r_end,
    PetscInt fine_Xl, PetscInt fine_Yl, PetscInt coarse_Yl, 
    PetscInt fine_seq_offset, cudaStream_t stream) // Added stream
{
    PetscInt lrows = coarse_r_end - coarse_r_start;
    int threads = 256;
    int blocks  = ((int)lrows + threads - 1) / threads;
    restrict_kernel<<<blocks, threads, 0, stream>>>(
        fseq_dev, xca_dev, coarse_r_start, coarse_r_end,
        fine_Xl, fine_Yl, coarse_Yl, fine_seq_offset);
}

void launch_prolong(
    const double* cseq_dev, double* xfa_dev,
    PetscInt fine_r_start, PetscInt fine_r_end,
    PetscInt fine_Yl, PetscInt coarse_Yl, PetscInt coarse_Xl,
    PetscInt coarse_seq_offset, PetscInt coarse_seq_ic_end, 
    cudaStream_t stream) // Added stream
{
    PetscInt lrows = fine_r_end - fine_r_start;
    int threads = 256;
    int blocks  = ((int)lrows + threads - 1) / threads;
    prolong_kernel<<<blocks, threads, 0, stream>>>(
        cseq_dev, xfa_dev, fine_r_start, fine_r_end,
        fine_Yl, coarse_Yl, coarse_Xl, coarse_seq_offset, coarse_seq_ic_end);
}

void launch_coarsen(
    const double* d_Rf, double* d_Rc,
    PetscInt Xf, PetscInt Yf, PetscInt Xc, PetscInt Yc, 
    cudaStream_t stream) // Added stream
{
    dim3 threads(16, 16);
    dim3 blocks(((int)Xc + 15) / 16, ((int)Yc + 15) / 16);
    coarsen_kernel<<<blocks, threads, 0, stream>>>(d_Rf, d_Rc, Xf, Yf, Xc, Yc);
}

void launch_update_rgrid(
    double* d_R, const double* d_Tthresh, 
    double temp, double ins_R, 
    PetscInt N, bool is_heating, cudaStream_t stream) // Added stream
{
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    update_rgrid_kernel<<<blocks, threads, 0, stream>>>(d_R, d_Tthresh, temp, ins_R, N, is_heating);
}

void launch_gather(double* d_dst, const double* d_src, const int* d_indices, int count, cudaStream_t stream) // Added stream
{
    int threads = 256;
    int blocks  = (count + threads - 1) / threads;
    if (blocks > 0) gather_kernel<<<blocks, threads, 0, stream>>>(d_dst, d_src, d_indices, count);
}

// === Memory Helpers ===
void cuda_malloc(void** ptr, size_t size) { cudaMalloc(ptr, size); }
void cuda_free(void* ptr) { if (ptr) cudaFree(ptr); }
void cuda_memcpy_to_device(void* dst, const void* src, size_t size) { cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice); }
void cuda_memcpy_to_host(void* dst, const void* src, size_t size) { cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost); }
void cuda_memcpy_device_to_device(void* dst, const void* src, size_t size) { cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice); }
void cuda_memcpy_device_to_device_async(void* dst, const void* src, size_t size, cudaStream_t stream) {
    cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToDevice, stream);
}
}





// === Explicit Matrix COO Assembly Kernel ===
__global__ void fill_coo_values_kernel(
    double* __restrict__ coo_v, const int* __restrict__ offsets, const double* __restrict__ R, 
    PetscInt Xl, PetscInt Yl, PetscInt N)
{
    PetscInt r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= N) return;

    int idx = offsets[r];
    double diag = 0.0;

    if (r == 0) {
        for (PetscInt j = 0; j < Yl; j++) {
            double gv = 2.0 / __ldg(&R[j]);
            coo_v[idx + 1 + j] = -gv; // off-diagonal connections to row 0
            diag += gv;
        }
        coo_v[idx] = diag; // Diagonal is placed at the start of the row's block
    } else {
        PetscInt gi = r - 1, i = gi / Yl, j = gi % Yl;
        int offset_idx = 1;

        if (i == 0) {
            double gv = 2.0 / __ldg(&R[gi]);
            coo_v[idx + offset_idx++] = -gv;
            diag += gv;
        } else {
            double gv = d_gmean(__ldg(&R[gi]), __ldg(&R[(i-1)*Yl + j]));
            coo_v[idx + offset_idx++] = -gv;
            diag += gv;
        }
        if (j > 0) {
            double gv = d_gmean(__ldg(&R[gi]), __ldg(&R[i*Yl + (j-1)]));
            coo_v[idx + offset_idx++] = -gv;
            diag += gv;
        }
        if (j < Yl - 1) {
            double gv = d_gmean(__ldg(&R[gi]), __ldg(&R[i*Yl + (j+1)]));
            coo_v[idx + offset_idx++] = -gv;
            diag += gv;
        }
        if (i < Xl - 1) {
            double gv = d_gmean(__ldg(&R[gi]), __ldg(&R[(i+1)*Yl + j]));
            coo_v[idx + offset_idx++] = -gv;
            diag += gv;
        } else {
            diag += 2.0 / __ldg(&R[gi]); // Bottom boundary reflection
        }
        coo_v[idx] = diag; // Diagonal is always at the 0th offset index
    }
}

void launch_fill_coo_values(
    double* coo_v_dev, const int* offsets_dev, const double* d_R,
    PetscInt Xl, PetscInt Yl, PetscInt N,
    cudaStream_t stream)
{
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    fill_coo_values_kernel<<<blocks, threads, 0, stream>>>(coo_v_dev, offsets_dev, d_R, Xl, Yl, N);
}