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

int main(void)
{
    mkl_set_dynamic(0);
    mkl_set_num_threads(12);
    mkl_domain_set_num_threads(MKL_DOMAIN_ALL, 12);

    omp_set_num_threads(12);

    cholmod_sparse* A;
    cholmod_dense* x = NULL, * b, * r;
    cholmod_factor* L;

    double one[2] = { 1,0 }, m1[2] = { -1,0 };
    cholmod_common c;
    cholmod_start(&c);

    c.nthreads_max = 12;
    c.print = 3;

    int dtype = CHOLMOD_DOUBLE;

    double Rmetal = 1;
    double Rins = 1000;
    const int X = 2000;
    const int Y = 2000;

    double* Tgrid = new double[X * Y]();
    Tgridset(Tgrid, X, Y);

    double* Rgrid = new double[X * Y]();
    Rgridset(Rgrid, X, Y, Rins);

    int nnz2 = (1 + 2 + 3 * (Y - 1)) + ((Y + 1) + 2 + 3 * (Y - 1)) + (((Y - 1) * 3) + 2) * (X - 2) - (Y + 1);
    cout << "nnz: " << nnz2 << endl;

    double* Gvals = new double[nnz2]();
    int* Rowlind = new int[nnz2]();
    int* Colcoords = new int[nnz2]();

    constructRowlind(Rowlind, nnz2, X, Y);
    constructColind(Colcoords, nnz2, X, Y);

    fillgvalsfirstrow(Gvals, Rgrid, X, Y);
    auto begin_fill = std::chrono::steady_clock::now();
    fillgvalsmid(Gvals, Rgrid, X, Y, nnz2);
    auto end_fill = std::chrono::steady_clock::now();
    cout << "Time difference fill mid = "
         << std::chrono::duration_cast<std::chrono::microseconds>(end_fill - begin_fill).count()
         << "[micros]" << endl;

    fillgvalslastrow(Gvals, Rgrid, X, Y, nnz2);
    fillgvalsfirselement(Gvals, Rgrid, X, Y);

    size_t n = (X * Y + 1);

    cholmod_triplet* T = cholmod_allocate_triplet(n, n, nnz2, 1, CHOLMOD_REAL, &c);

    int* Ti = (int*)T->i;
    int* Tj = (int*)T->j;
    double* Tx = (double*)T->x;

    for (int k = 0; k < nnz2; k++)
    {
        Ti[T->nnz] = Rowlind[k];
        Tj[T->nnz] = Colcoords[k];
        Tx[T->nnz] = Gvals[k];
        T->nnz++;
    }

    A = cholmod_triplet_to_sparse(T, nnz2, &c);
    A->stype = -1;

    cout << "Initial A val check: " << ((double*)A->x)[0] << endl;

    // Mapping
    int* Ap = (int*)A->p;
    int* Ai = (int*)A->i;
    int* gvals_to_Ax = new int[nnz2];

    for (int k = 0; k < nnz2; k++)
    {
        int row = Rowlind[k];
        int col = Colcoords[k];

        for (int p = Ap[col]; p < Ap[col + 1]; p++)
        {
            if (Ai[p] == row)
            {
                gvals_to_Ax[k] = p;
                break;
            }
        }
    }
    cout << "Mapping built" << endl;

    // RHS
    b = cholmod_zeros(A->nrow, 1, A->xtype + dtype, &c);
    ((double*)b->x)[0] = 1;

    // Sparse RHS pattern
    cholmod_sparse* Bset = cholmod_allocate_sparse(A->nrow, 1, 1,
        0, 1, 0, CHOLMOD_PATTERN + dtype, &c);

    int* Bsetp = (int*)Bset->p;
    int* Bseti = (int*)Bset->i;

    Bsetp[0] = 0;
    Bsetp[1] = 1;
    Bseti[0] = 0;

    cholmod_dense* Ywork = NULL, * Ework = NULL;
    cholmod_sparse* Xset = NULL;

    // Settings
    c.supernodal = CHOLMOD_SUPERNODAL;
    c.supernodal_switch = 0;
    c.final_ll = 1;
    c.dbound = 0.0;

    auto begin_analyze = std::chrono::steady_clock::now();
    L = cholmod_analyze(A, &c);
    auto end_analyze = std::chrono::steady_clock::now();
    cout << "Analyze time = "
         << std::chrono::duration_cast<std::chrono::microseconds>(end_analyze - begin_analyze).count()
         << "[micros]" << endl;

    float tempstart = 305;
    float tempfinal = 365;
    float tempstep = 30;

    for (float temp = tempstart; temp <= tempfinal; temp += tempstep)
    {
        cout << "\nT: " << temp << endl;

        auto begin_update = std::chrono::steady_clock::now();

        for (int i = 0; i < X * Y; i++)
            if (Tgrid[i] <= temp) Rgrid[i] = Rmetal;

        fillgvalsfirstrow(Gvals, Rgrid, X, Y);
        fillgvalsmid(Gvals, Rgrid, X, Y, nnz2);
        fillgvalslastrow(Gvals, Rgrid, X, Y, nnz2);
        fillgvalsfirselement(Gvals, Rgrid, X, Y);

        double* Ax = (double*)A->x;
        for (int k = 0; k < nnz2; k++)
            Ax[gvals_to_Ax[k]] = Gvals[k];

        auto end_update = std::chrono::steady_clock::now();
        cout << "Update G time = "
             << std::chrono::duration_cast<std::chrono::microseconds>(end_update - begin_update).count()
             << "[micros]" << endl;

        cout << "pre-factor: L->is_super=" << L->is_super
             << " L->is_ll=" << L->is_ll
             << " L->minor=" << L->minor << endl;

        auto begin_fact = std::chrono::steady_clock::now();
        double beta[2] = { 0,0 };
        cholmod_factorize_p(A, beta, NULL, 0, L, &c);
        auto end_fact = std::chrono::steady_clock::now();

        cout << "Refactor time = "
             << std::chrono::duration_cast<std::chrono::microseconds>(end_fact - begin_fact).count()
             << "[micros]" << endl;

        cout << "post-factor: L->is_super=" << L->is_super
             << " L->is_ll=" << L->is_ll
             << " L->minor=" << L->minor << endl;

        // Copy + solve2
        cholmod_factor* Ltmp = cholmod_copy_factor(L, &c);

		
		cout << "post-copy (L original): "
			<< "is_super=" << L->is_super
			<< " is_ll=" << L->is_ll
			<< " minor=" << L->minor << endl;

		cout << "post-copy (Ltmp copy): "
			<< "is_super=" << Ltmp->is_super
			<< " is_ll=" << Ltmp->is_ll
			<< " minor=" << Ltmp->minor << endl;

        auto begin_solve = std::chrono::steady_clock::now();
        cholmod_solve2(CHOLMOD_A, Ltmp, b, Bset,
            &x, &Xset, &Ywork, &Ework, &c);
        auto end_solve = std::chrono::steady_clock::now();

        cout << "solve2 time = "
             << std::chrono::duration_cast<std::chrono::microseconds>(end_solve - begin_solve).count()
             << "[micros]" << endl;

        cout << "out1: " << ((double*)x->x)[0] << endl;

        r = cholmod_copy_dense(b, &c);
        cholmod_sdmult(A, 0, m1, one, x, r, &c);
        printf("norm(b-Ax) %8.1e\n", cholmod_norm_dense(r, 0, &c));
        cout << "rcond: " << cholmod_rcond(L, &c) << endl;

        cholmod_free_dense(&r, &c);
        cholmod_free_factor(&Ltmp, &c);
    }

    cholmod_free_dense(&Ywork, &c);
    cholmod_free_dense(&Ework, &c);
    cholmod_free_sparse(&Bset, &c);
    cholmod_free_sparse(&Xset, &c);
    cholmod_free_dense(&x, &c);

    delete[] gvals_to_Ax;

    cholmod_free_factor(&L, &c);
    cholmod_free_sparse(&A, &c);
    cholmod_finish(&c);

    return 0;
}