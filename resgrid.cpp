// ------------------------------------------------
// Code from Tim Davis' CHOLMOD User Guide
// ------------------------------------------------

#include "cholmod.h"
#include "amd.h"
#include "colamd.h"
#include "mkl_cblas.h"
#include "mkl_lapack.h"
#include "mkl.h"


#include <stdio.h>
#include <iostream>
#include "Header.h"
#include <chrono>
#include <omp.h>
#include <string>




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

static void check_version(const char* package, int ver[3],
	int major, int minor, int patch)
{
	printf("%s version %d.%d.%d\n", package, ver[0], ver[1], ver[2]);
#ifndef TEST_COVERAGE
	if (ver[0] != major || ver[1] != minor || ver[2] != patch)
	{
		printf("header version differs (%d,%d,%d) from library\n",
			major, minor, patch);
		my_handler(CHOLMOD_INVALID, __FILE__, __LINE__,
			"version mismatch");
	}
#endif
}


int main(void)
{

	mkl_set_dynamic(0);
	mkl_set_num_threads(12);
	mkl_domain_set_num_threads(MKL_DOMAIN_ALL, 12);

	omp_set_num_threads(12);
	cholmod_sparse* A;
	cholmod_dense* x, * b, * r;
	cholmod_factor* L;
	double one[2] = { 1,0 }, m1[2] = { -1,0 }; /* basic scalars */
	cholmod_common c;
	cholmod_start(&c); /* start CHOLMOD */
	cout << "gpu: " << c.useGPU << endl;
	c.nthreads_max = 12;
	cout << "nthreads: " << c.nthreads_max << endl;



	int dtype = CHOLMOD_DOUBLE;
	c.print = 5;
	//double metis_memory = 2;
	c.useGPU = 0;

	double Rmetal = 1;
	double Rins = 1000;
	const int X = 2000; //number of columns, number sites in X direction
	const int Y = 2000; //number of rows, number of sites in Y direction

	double* Tgrid = new double[X * Y]();
	Tgridset(Tgrid, X, Y);
	double* Rgrid = new double[X * Y]();
	Rgridset(Rgrid, X, Y, Rins);

	int nnz2 = (1 + 2 + 3 * (Y - 1)) + ((Y + 1) + 2 + 3 * (Y - 1)) + (((Y - 1) * 3) + 2) * (X - 2) - (Y + 1);//(1+(Y-2)*3+2)+((X-1)*((Y - 1) * 3+2))-(Y-2);
	cout << "nnz: " << nnz2 << endl;

	double* Gvals = new double[nnz2]();
	int* Rowlind = new int[nnz2]();
	int* Colcoords = new int[nnz2]();

	constructRowlind(Rowlind, nnz2, X, Y);
	constructColind(Colcoords, nnz2, X, Y);

	fillgvalsfirstrow(Gvals, Rgrid, X, Y);
	std::chrono::steady_clock::time_point begin1 = std::chrono::steady_clock::now();
	fillgvalsmid(Gvals, Rgrid, X, Y, nnz2);
	std::chrono::steady_clock::time_point end1 = std::chrono::steady_clock::now();
	std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::microseconds>(end1 - begin1).count() << "[micros]" << std::endl;
	fillgvalslastrow(Gvals, Rgrid, X, Y, nnz2);
	fillgvalsfirselement(Gvals, Rgrid, X, Y);

	if ((X * Y + 1) < 100) {
		printrowsfloat(Gvals, nnz2);
	}
	cout << endl << endl << "end of Gvals" << endl;

	//printrows(Colcoords,nnz2);
	cout << endl << endl << "end of cols" << endl;

	//printrows(Rowlind, nnz2);
	cout << endl << endl << "end of rows" << endl;


	float temp = 300;

	Tgridupdater(Tgrid, X, Y, temp);

	bool skipfornow = 1;
	if (skipfornow != 0) {

	int nnz = nnz2;
	size_t nnzsizet = nnz2;
	size_t sizer = (X * Y + 1);

	cholmod_triplet* T = cholmod_allocate_triplet(sizer, sizer, nnzsizet, 1, CHOLMOD_REAL, &c);
	std::cout << "T nnz check:" << T->nnz << std::endl;

	int* triplet_i = (int*)(T->i);
	int* triplet_j = (int*)(T->j);
	double* triplet_x = (double*)(T->x);
	for (int ne = 0; ne < nnz2; ne++)
	{
		triplet_i[T->nnz] = Rowlind[ne];
		triplet_j[T->nnz] = Colcoords[ne];
		triplet_x[T->nnz] = Gvals[ne];
		T->nnz++;
	}

	A = cholmod_triplet_to_sparse(T, nnzsizet, &c);
	if ((X * Y + 1) < 100) {
		cholmod_print_sparse(A, "A", &c); /* print the matrix */
	}

	A->stype = -1; // lower triangular; required for supernodal cholmod_solve

	std::cout << "check val" << ((double*)A->x)[0] << std::endl;

	if (A == NULL || A->stype == 0) /* A must be symmetric */
	{
		cholmod_free_sparse(&A, &c);
		cholmod_finish(&c);
		return (0);
	}

	// Build a one-time mapping from Gvals index -> A->x index.
	// After cholmod_triplet_to_sparse the values in A->x are in CSC
	// (column-sorted) order, which is NOT the same order as Gvals /
	// Rowlind / Colcoords.  We walk the CSC structure once here to
	// record, for each Gvals entry k, which slot in A->x holds that
	// (row, col) pair.  In the temperature loop we then scatter values
	// with: A->x[gvals_to_Ax[k]] = Gvals[k]  -- no triplet rebuild,
	// no free/realloc of A, and L's symbolic factor stays valid.
	int* Ap = (int*)(A->p);   // column pointers (size ncol+1)
	int* Ai = (int*)(A->i);   // row indices      (size nnz)
	int* gvals_to_Ax = new int[nnz2];
	for (int k = 0; k < nnz2; k++)
	{
		int row = Rowlind[k];
		int col = Colcoords[k];
		bool found = false;
		for (int p = Ap[col]; p < Ap[col + 1]; p++)
		{
			if (Ai[p] == row)
			{
				gvals_to_Ax[k] = p;
				found = true;
				break;
			}
		}
		if (!found)
		{
			printf("ERROR: mapping not found for Gvals[%d] (row=%d col=%d)\n", k, row, col);
			exit(1);
		}
	}
	std::cout << "Gvals->A->x mapping built." << std::endl;

	b = cholmod_zeros(A->nrow, 1, A->xtype + dtype, &c);
	((double*)b->x)[0] = 1;

	std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

	c.supernodal = CHOLMOD_SUPERNODAL;
	c.supernodal_switch = 0; // always use supernodal regardless of flops/nnz ratio

	c.nmethods = 1;
	c.method[0].ordering = CHOLMOD_AMD;

	// Lower the pivot acceptance threshold. The default dbound causes
	// CHOLMOD to abandon supernodal Cholesky and fall back to simplicial
	// LDL when it encounters small pivots (which occur here because the
	// resistance ratio Rins/Rmetal = 1000 makes the matrix ill-conditioned
	// as cells switch state). Setting dbound to 0 tells CHOLMOD to accept
	// any positive pivot and stay in supernodal mode.
	c.dbound = 0.0;
	L = cholmod_analyze(A, &c); /* analyze */
	std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
	std::cout << "Analyze A time = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[micros]" << std::endl;


	std::cout << "here1 = " << A->stype << endl;
	if ((X * Y + 1) < 100) {
		cholmod_print_factor(L, "L", &c);
	}

	std::cout << "here2 = " << A->stype << endl;
	

	float tempstart = 305;
	float tempfinal = 365;
	float tempstep = 30;

	// Ywork/Ework retained for potential future use but not needed by cholmod_solve.
	cholmod_dense* Ywork = NULL, * Ework = NULL;
	cholmod_dense* x = NULL;

	// bset/Xset were used by cholmod_solve2 but are not needed by cholmod_solve.
	cholmod_sparse* bset = NULL;

	for (float temp = tempstart; temp <= tempfinal; temp = temp + tempstep) {
		std::cout << "T:" << temp << std::endl;

		// Rebuild b (it is freed at the end of each iteration below)
		b = cholmod_zeros(A->nrow, 1, A->xtype + dtype, &c);
		((double*)b->x)[0] = 1;

		std::chrono::steady_clock::time_point begin1 = std::chrono::steady_clock::now();
		for (int i = 0; i < X * Y; i++) {
			if (Tgrid[i] <= temp) {
				Rgrid[i] = Rmetal;
			}
		}

		fillgvalsfirstrow(Gvals, Rgrid, X, Y);
		fillgvalsmid(Gvals, Rgrid, X, Y, nnz2);
		fillgvalslastrow(Gvals, Rgrid, X, Y, nnz2);
		fillgvalsfirselement(Gvals, Rgrid, X, Y);

		// FIX 3: Scatter updated values directly into A->x using the
		//         pre-built mapping.  A is never freed so L's symbolic
		//         factor remains valid across iterations.
		double* Ax = (double*)(A->x);
		for (int k = 0; k < nnz2; k++)
			Ax[gvals_to_Ax[k]] = Gvals[k];

		std::chrono::steady_clock::time_point end1 = std::chrono::steady_clock::now();
		std::cout << "Update G time = " << std::chrono::duration_cast<std::chrono::microseconds>(end1 - begin1).count() << "[micros]" << std::endl;

		std::chrono::steady_clock::time_point begin2 = std::chrono::steady_clock::now();

		// If CHOLMOD fell back to simplicial on the previous iteration,
		// re-analyze to get a fresh supernodal symbolic factor.
		if (!L->is_super)
		{
			std::cout << "NOTE: re-analyzing (L was simplicial)" << std::endl;
			cholmod_free_factor(&L, &c);
			L = cholmod_analyze(A, &c);
		}

		// Print L state before factorizing so we can see what
		// cholmod_factorize_p is working with each iteration.
		std::cout << "pre-factor: L->is_super=" << L->is_super
		          << " L->is_ll=" << L->is_ll
		          << " L->minor=" << L->minor << std::endl;

		double beta[2] = { 0, 0 };
		// final_ll=1 keeps L in LL' form after factorization.
		// Without this CHOLMOD converts to LDL' on completion, and
		// on the next call sees a non-LL' supernodal factor which
		// triggers a simplicial restart.
		c.final_ll = 1;
		cholmod_factorize_p(A, beta, NULL, 0, L, &c);

		std::chrono::steady_clock::time_point end2 = std::chrono::steady_clock::now();
		std::cout << "Refactor time = " << std::chrono::duration_cast<std::chrono::microseconds>(end2 - begin2).count() << "[micros]" << std::endl;
		std::cout << "post-factor: L->is_super=" << L->is_super
		          << " L->is_ll=" << L->is_ll
		          << " L->minor=" << L->minor << std::endl;

		std::chrono::steady_clock::time_point begin4 = std::chrono::steady_clock::now();

		// Use cholmod_solve instead of cholmod_solve2. cholmod_solve2
		// converts L from supernodal to simplicial internally and does
		// not restore it, causing every subsequent cholmod_factorize_p
		// to run in slow simplicial mode. cholmod_solve does not modify
		// L's storage type, so the supernodal structure is preserved for
		// the next factorization. The workspace (Ywork/Ework) is managed
		// separately and freed after the loop.
		if (x != NULL) cholmod_free_dense(&x, &c);
		x = cholmod_solve(CHOLMOD_A, L, b, &c);

		std::chrono::steady_clock::time_point end4 = std::chrono::steady_clock::now();
		std::cout << "fastSolve time = " << std::chrono::duration_cast<std::chrono::microseconds>(end4 - begin4).count() << "[micros]" << std::endl;

		std::cout << "out1:" << " " << ((double*)x->x)[0] << std::endl;

		// Copy b into r BEFORE freeing b, then free b
		r = cholmod_copy_dense(b, &c);          /* r = b */
		cholmod_free_dense(&b, &c);             /* safe to free b now */

		// Residual check: both r and x are valid here
		cholmod_sdmult(A, 0, m1, one, x, r, &c); /* r = r - Ax */
		printf("norm(b-Ax) %8.1e\n", cholmod_norm_dense(r, 0, &c)); /* print norm(r) */
		std::cout << "rcond:" << cholmod_rcond(L, &c) << std::endl;

		// Free r after it has been used
		cholmod_free_dense(&r, &c);

		if ((X * Y + 1) < 100) {
			cholmod_print_factor(L, "L", &c);
			cholmod_print_dense(x, "x", &c);
			cholmod_print_common("c", &c);
		}
	}

	// Free persistent solve workspace after the loop
	cholmod_free_dense(&Ywork, &c);
	cholmod_free_dense(&Ework, &c);
	cholmod_free_dense(&x, &c);
	delete[] gvals_to_Ax;

	} // end skipfornow

	std::cout << "MKL max threads: " << mkl_get_max_threads() << std::endl;
	std::cout << "MKL BLAS domain: " << mkl_domain_get_max_threads(MKL_DOMAIN_BLAS) << std::endl;

	cholmod_free_factor(&L, &c); /* free matrices */
	cholmod_free_sparse(&A, &c);
	cholmod_finish(&c); /* finish CHOLMOD */
	std::cout << sizeof(int) << std::endl;
	std::cout << c.blas_ok << std::endl;

	return (0);
}