#include "cholmod.h"
#include <stdio.h>
#include <iostream>
#include <vector>
#include <stdint.h>
#include <memory>
#include <algorithm>
#include <chrono>
#include <stdlib.h>
#include "Header.h"

using namespace std::chrono;

int main(void) {
    // --- MKL DIAGNOSTIC MODE ---
    const char* threads = "4"; 
	// This tells MKL to be extra careful with memory buffers
	_putenv_s("MKL_PARDISO_ALIGNMENT", "64"); 
	// This forces MKL to use a more conservative threading layer
	_putenv_s("MKL_INTERFACE_LAYER", "ILP64");
	_putenv_s("MKL_THREADING_LAYER", "INTEL");
    _putenv_s("MKL_NUM_THREADS", "1");
    //_putenv_s("OMP_NUM_THREADS", "1");
    
    // TURN ON MKL VERBOSE OUTPUT
    //_putenv_s("MKL_VERBOSE", "1");

    printf("--- SuiteSparse Diagnostic: MKL_VERBOSE enabled (%s Threads) ---\n", threads);
	

    cholmod_common c;
    cholmod_l_start(&c);
    c.supernodal = CHOLMOD_SUPERNODAL; 
    c.supernodal_switch = 40.0;
    c.final_ll = 1;
    c.nmethods = 1;
    c.method[0].ordering = CHOLMOD_AMD;
	c.nthreads_max=1;
	printf("cholmodthreads: %ld\n", c.nthreads_max);
	printf("\n");

    const SuiteSparse_long X = 1000, Y = 1000;
    const SuiteSparse_long n = X * Y + 1;
    const SuiteSparse_long nnz2 = compute_nnz2(X, Y);

    auto Tgrid = std::make_unique<double[]>(X * Y);
    auto Rgrid = std::make_unique<double[]>(X * Y);
    auto Gvals = std::make_unique<double[]>(nnz2);
    auto Rowlind = std::make_unique<SuiteSparse_long[]>(nnz2);
    auto Colcoords = std::make_unique<SuiteSparse_long[]>(nnz2);

    Tgridset(Tgrid.get(), X, Y);
    Rgridset(Rgrid.get(), X, Y, 1000.0);
    constructRowlind(Rowlind.get(), X, Y);
    constructColind(Colcoords.get(), X, Y);
    fillGvals(Gvals.get(), Rgrid.get(), X, Y);
    fillGvalsSourceDiag(Gvals.get(), Y);

    cholmod_triplet* T = cholmod_l_allocate_triplet(n, n, nnz2, -1, CHOLMOD_REAL, &c);
    for (SuiteSparse_long k = 0; k < nnz2; k++) {
        ((SuiteSparse_long*)T->i)[k] = Rowlind[k];
        ((SuiteSparse_long*)T->j)[k] = Colcoords[k];
        ((double*)T->x)[k] = Gvals[k];
    }
    T->nnz = nnz2;
    cholmod_sparse* A = cholmod_l_triplet_to_sparse(T, nnz2, &c);
    A->stype = -1;
    cholmod_l_free_triplet(&T, &c);

    SuiteSparse_long* Ap = (SuiteSparse_long*)A->p;
    SuiteSparse_long* Ai = (SuiteSparse_long*)A->i;
    auto triplet_to_csc = std::make_unique<SuiteSparse_long[]>(nnz2);
    for (SuiteSparse_long k = 0; k < nnz2; k++) {
        triplet_to_csc[k] = -1;
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
    cholmod_dense *x = NULL, *b = NULL;

    printf("\n>>> ENTERING SIMULATION LOOP. MKL VERBOSE OUTPUT WILL BEGIN NOW. <<<\n\n");
    fflush(stdout);

	FILE* f = fopen("results.dat", "w");
	if (!f) { printf("ERROR: could not open results.dat\n"); exit(1); }
		fprintf(f, "# temp x0 rcond loop_time_microsec\n");
    // Just one step needed to catch the crash
    for (int step = 0; step <= 75; step++) {
		auto step_start = high_resolution_clock::now();
        double temp = 300.0 + (double)step;
		printf("temp: %.2f\n", temp);

		
		

        for (SuiteSparse_long i = 0; i < X * Y; i++) if (temp >= Tgrid[i]) Rgrid[i] = 1.0;
        fillGvals(Gvals.get(), Rgrid.get(), X, Y);
        fillGvalsSourceDiag(Gvals.get(), Y);

        double* Ax = (double*)A->x;
        std::fill(Ax, Ax + A->nzmax, 0.0); 
        for (SuiteSparse_long k = 0; k < nnz2; k++) if (triplet_to_csc[k] >= 0) Ax[triplet_to_csc[k]] += Gvals[k];

        printf("--- Calling cholmod_l_factorize ---\n"); fflush(stdout);
        cholmod_l_factorize(A, L, &c);
        
        if (b) cholmod_l_free_dense(&b, &c);
        b = cholmod_l_zeros(n, 1, CHOLMOD_REAL, &c);
        ((double*)b->x)[0] = 1.0; 

        if (x) cholmod_l_free_dense(&x, &c);
        
        printf("--- Calling cholmod_l_solve ---\n"); fflush(stdout);
        x = cholmod_l_solve(CHOLMOD_A, L, b, &c);
		
		auto step_end = high_resolution_clock::now();
		fprintf(f, "%f %f\n", temp,((double*)x->x)[0] );
        double duration_ms = duration<double, std::milli>(step_end - step_start).count();
		if (x && x->x) {
            printf("Step %d: %.2f ms | V[0]: %.4f\n", step, duration_ms, ((double*)x->x)[0]);
        }
        fflush(stdout);
    }
	fclose(f);

    cholmod_l_free_factor(&L, &c);
    cholmod_l_free_sparse(&A, &c);
    cholmod_l_finish(&c);
    return 0;
}