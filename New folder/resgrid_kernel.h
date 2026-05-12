#pragma once
#include <petscconf.h>
#include <petscsys.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================
// MatVec and Diagonal
// ============================================================
void launch_matvec(
    const double* xlocal_dev, double* y_dev, const double* d_R,
    PetscInt r_start, PetscInt r_end,
    PetscInt Xl, PetscInt Yl,
    PetscInt source_local_idx,
    PetscInt above_halo_start_global, PetscInt above_halo_start_local,
    PetscInt below_halo_start_global, PetscInt below_halo_start_local);

void launch_diagonal(
    double* d_dev, const double* d_R,
    PetscInt r_start, PetscInt r_end,
    PetscInt Xl, PetscInt Yl);

// ============================================================
// Intergrid Operations
// ============================================================
void launch_restrict(
    const double* fseq_dev, double* xca_dev,
    PetscInt coarse_r_start, PetscInt coarse_r_end,
    PetscInt fine_Xl, PetscInt fine_Yl, PetscInt coarse_Yl,
    PetscInt fine_seq_offset);

void launch_prolong(
    const double* cseq_dev, double* xfa_dev,
    PetscInt fine_r_start, PetscInt fine_r_end,
    PetscInt fine_Yl, PetscInt coarse_Yl, PetscInt coarse_Xl,
    PetscInt coarse_seq_offset, PetscInt coarse_seq_ic_end);

void launch_coarsen(
    const double* d_Rf, double* d_Rc,
    PetscInt Xf, PetscInt Yf, PetscInt Xc, PetscInt Yc);

// ============================================================
// Grid State Machine (NEW)
// ============================================================
void launch_update_rgrid(
    double* d_R, const double* d_Tthresh, 
    double temp, double ins_R, 
    PetscInt N, bool is_heating);

void launch_gather(double* d_dst, const double* d_src,
                   const int* d_indices, int count);

// ============================================================
// CUDA memory helpers
// ============================================================
void cuda_malloc(void** ptr, size_t size);
void cuda_free(void* ptr);
void cuda_memcpy_to_device(void* dst, const void* src, size_t size);
void cuda_memcpy_to_host(void* dst, const void* src, size_t size);
void cuda_memcpy_device_to_device(void* dst, const void* src, size_t size);

#ifdef __cplusplus
}
#endif