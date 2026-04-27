#include "cholmod.h"
#include "amd.h"
#include "colamd.h"
#include "mkl_cblas.h"
#include "mkl_lapack.h"
#include "mkl.h"

#include <stdio.h>
#include <iostream>
#include <chrono>
#include <omp.h>
#include <string>
#include <vector>

#include "Header.h"

using std::cout;
using std::endl;

FILE* ff;

static void my_handler(int status, const char* file, int line,
    const char* message)
{
    printf("cholmod error: file: %s line: %d status: %d: %s\n",
        file, line, status, message);
    if (status < 0)
    {
        if (ff != NULL) fclose(ff);
        exit(1);
    }
}

int main(void)
{
    mkl_set_dynamic(0);
    mkl_set_num_threads(12);
    mkl_domain_set_num_threads(MKL_DOMAIN_ALL, 12);
    omp_set_num_threads(12);

    cholmod_sparse* A;
    cholmod_dense*  x = NULL, *b = NULL, *r = NULL;
    cholmod_factor* L;

    double one[2] = { 1, 0 }, m1[2] = { -1, 0 };

    cholmod_common c;
    cholmod_start(&c);
    c.error_handler = my_handler;
    c.nthreads_max  = 12;
    c.useGPU        = 0;

    int dtype = CHOLMOD_DOUBLE;

    const int X = 1000;
    const int Y = 1000;

    double* Tgrid = new double[X * Y]();
    double* Rgrid = new double[X * Y]();

    Tgridset(Tgrid, X, Y);
    Rgridset(Rgrid, X, Y, 1000.0);

    // -----------------------------------------------------------------------
    // nnz and index arrays
    // -----------------------------------------------------------------------
    const int nnz2 = compute_nnz2(X, Y);   // 1 + X*(3Y-1)

    double* Gvals    = new double[nnz2]();
    int*    Rowlind  = new int[nnz2]();
    int*    Colcoords = new int[nnz2]();

    constructRowlind (Rowlind,    nnz2, X, Y);
    constructColind  (Colcoords,  nnz2, X, Y);

    fillGvals          (Gvals, Rgrid, X, Y);
    fillGvalsSourceDiag(Gvals,        X, Y);

    // -----------------------------------------------------------------------
    // Build initial sparse matrix via triplet -> CSC
    // stype=-1 on the triplet: CHOLMOD keeps only lower-triangle entries and
    // sets A->stype=-1 on the result, so Cholesky works directly on A.
    // -----------------------------------------------------------------------
    const size_t n = (size_t)(X * Y + 1);

    cholmod_triplet* T =
        cholmod_allocate_triplet(n, n, (size_t)nnz2,
                                 -1 /*lower tri*/, CHOLMOD_REAL, &c);

    int*    ti = (int*)   T->i;
    int*    tj = (int*)   T->j;
    double* tx = (double*)T->x;

    for (int k = 0; k < nnz2; k++)
    {
        ti[k] = Rowlind[k];
        tj[k] = Colcoords[k];
        tx[k] = Gvals[k];
    }
    T->nnz = (size_t)nnz2;

    A = cholmod_triplet_to_sparse(T, (size_t)nnz2, &c);
    cholmod_free_triplet(&T, &c);

    // -----------------------------------------------------------------------
    // Build static mapping: triplet k -> index into A->x (CSC slot).
    // Valid for the entire run because the sparsity pattern never changes.
    // -----------------------------------------------------------------------
    int* Ap = (int*)A->p;  // column pointers, length ncol+1
    int* Ai = (int*)A->i;  // row indices

    std::vector<int> triplet_to_csc(nnz2, -1);
    for (int k = 0; k < nnz2; k++)
    {
        int row = Rowlind[k];
        int col = Colcoords[k];
        if (row < col) continue;  // upper triangle: discarded by CHOLMOD

        // Binary search for row within CSC column col
        int lo = Ap[col], hi = Ap[col + 1] - 1;
        while (lo <= hi)
        {
            int mid = lo + (hi - lo) / 2;
            if      (Ai[mid] < row) lo = mid + 1;
            else if (Ai[mid] > row) hi = mid - 1;
            else { triplet_to_csc[k] = mid; break; }
        }
    }

    // Sanity check
    for (int k = 0; k < nnz2; k++)
        if (Rowlind[k] >= Colcoords[k] && triplet_to_csc[k] < 0)
            printf("WARNING: unmapped lower-tri entry k=%d (%d,%d)\n",
                   k, Rowlind[k], Colcoords[k]);

    // -----------------------------------------------------------------------
    // RHS: b[0]=1 sets source node voltage; all other entries 0.
    // -----------------------------------------------------------------------
    b = cholmod_zeros(A->nrow, 1, A->xtype + dtype, &c);
    ((double*)b->x)[0] = 1.0;

    // -----------------------------------------------------------------------
    // Symbolic analysis (done ONCE — pattern is fixed across all temperatures)
    // -----------------------------------------------------------------------
    c.supernodal         = CHOLMOD_SUPERNODAL;
    c.supernodal_switch  = 0;
    c.nmethods           = 1;
    c.method[0].ordering = CHOLMOD_AMD;
    c.dbound             = 0.0;

    L = cholmod_analyze(A, &c);

    // -----------------------------------------------------------------------
    // Output file
    // -----------------------------------------------------------------------
    FILE* f = fopen("results.dat", "w");
    if (!f) { printf("ERROR: could not open results.dat\n"); exit(1); }
    fprintf(f, "# temp x0 rcond loop_time_microsec\n");

    const double tempstart = 200.0;
    const double tempfinal = 395.0;
    const double tempstep  = 1.0;
    const int    nsteps    = (int)((tempfinal - tempstart) / tempstep);

    for (int step = 0; step <= nsteps; step++)
    {
        auto loop_begin = std::chrono::steady_clock::now();

        // Integer step avoids floating-point accumulation across 120 iterations
        const double temp = tempstart + step * tempstep;
        printf("T = %.2f\n", temp);

        // ------------------------------------------------------------------
        // Rebuild RHS
        // ------------------------------------------------------------------
        cholmod_free_dense(&b, &c);
        b = cholmod_zeros(A->nrow, 1, A->xtype + dtype, &c);
        ((double*)b->x)[0] = 1.0;

        // ------------------------------------------------------------------
        // Update resistances (one-way ratchet: insulator -> metal at Tc)
        // ------------------------------------------------------------------
        for (int i = 0; i < X * Y; i++)
            if (Tgrid[i] <= temp)
                Rgrid[i] = 1.0;

        // ------------------------------------------------------------------
        // Recompute conductance values
        // ------------------------------------------------------------------
        fillGvals          (Gvals, Rgrid, X, Y);
        fillGvalsSourceDiag(Gvals,        X, Y);

        // ------------------------------------------------------------------
        // Scatter Gvals into A->x via the precomputed mapping.
        // Zero A->x first (no duplicate pairs in our layout, but defensive).
        // ------------------------------------------------------------------
        double* Ax = (double*)A->x;
        for (size_t p = 0; p < A->nzmax; p++) Ax[p] = 0.0;

        for (int k = 0; k < nnz2; k++)
        {
            int p = triplet_to_csc[k];
            if (p >= 0)
                Ax[p] += Gvals[k];
        }

        // ------------------------------------------------------------------
        // Numerical factorisation and solve
        // ------------------------------------------------------------------
        double beta[2] = { 0.0, 0.0 };
        cholmod_factorize_p(A, beta, NULL, 0, L, &c);

        if (x) cholmod_free_dense(&x, &c);
        x = cholmod_solve(CHOLMOD_A, L, b, &c);

        // Optional residual check (remove for max performance)
        r = cholmod_copy_dense(b, &c);
        cholmod_sdmult(A, 0, m1, one, x, r, &c);
        cholmod_free_dense(&r, &c);

        // ------------------------------------------------------------------
        // Timing + logging
        // ------------------------------------------------------------------
        auto loop_end = std::chrono::steady_clock::now();
        long long loop_time =
            std::chrono::duration_cast<std::chrono::microseconds>(
                loop_end - loop_begin).count();

        double x0    = ((double*)x->x)[0];
        double rcond = cholmod_rcond(L, &c);

        fprintf(f, "%f %f %e %lld\n", temp, x0, rcond, loop_time);
        printf("loop time = %lld microseconds\n", loop_time);
    }

    // -----------------------------------------------------------------------
    // Cleanup
    // -----------------------------------------------------------------------
    fclose(f);

    delete[] Tgrid;
    delete[] Rgrid;
    delete[] Gvals;
    delete[] Rowlind;
    delete[] Colcoords;

    cholmod_free_dense (&b, &c);
    cholmod_free_dense (&x, &c);
    cholmod_free_factor(&L, &c);
    cholmod_free_sparse(&A, &c);
    cholmod_finish(&c);

    return 0;
}
