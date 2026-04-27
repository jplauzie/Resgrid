#ifndef HEADER_H
#define HEADER_H

#include <stdint.h>
#include <random>

typedef int64_t SuiteSparse_long;

static inline double g(double a, double b) { return 2.0 / (a + b); }
static inline SuiteSparse_long node_idx(SuiteSparse_long i, SuiteSparse_long j, SuiteSparse_long Y) { 
    return 1 + i * Y + j; 
}

// Fixed: 1 entry for source diag, plus 3 entries for every single node in the grid.
static inline SuiteSparse_long compute_nnz2(SuiteSparse_long X, SuiteSparse_long Y) {
    return 1 + (X * Y * 3);
}

void Tgridset(double Tgrid[], SuiteSparse_long X, SuiteSparse_long Y) {
    static std::mt19937 gen(41);
    static std::normal_distribution<double> dist(345.0, 6.0);
    for (SuiteSparse_long i = 0; i < X * Y; i++) Tgrid[i] = dist(gen);
}

void Rgridset(double Rgrid[], SuiteSparse_long X, SuiteSparse_long Y, double Rins) {
    for (SuiteSparse_long i = 0; i < X * Y; i++) Rgrid[i] = Rins;
}

void constructRowlind(SuiteSparse_long Rowlind[], SuiteSparse_long X, SuiteSparse_long Y) {
    SuiteSparse_long k = 0;
    Rowlind[k++] = 0; // Source diag
    for (SuiteSparse_long i = 0; i < X; i++) {
        for (SuiteSparse_long j = 0; j < Y; j++) {
            SuiteSparse_long cur = node_idx(i, j, Y);
            Rowlind[k++] = cur; // Entry 1: Connection Above (or Source)
            Rowlind[k++] = cur; // Entry 2: Connection Left (or zero)
            Rowlind[k++] = cur; // Entry 3: Diag
        }
    }
}

void constructColind(SuiteSparse_long Colind[], SuiteSparse_long X, SuiteSparse_long Y) {
    SuiteSparse_long k = 0;
    Colind[k++] = 0; // Source diag
    for (SuiteSparse_long i = 0; i < X; i++) {
        for (SuiteSparse_long j = 0; j < Y; j++) {
            SuiteSparse_long cur = node_idx(i, j, Y);
            // Entry 1: Above or Source
            if (i == 0) Colind[k++] = 0; 
            else Colind[k++] = node_idx(i - 1, j, Y);
            
            // Entry 2: Left or Dummy
            if (j > 0) Colind[k++] = node_idx(i, j - 1, Y);
            else Colind[k++] = cur; // Set to diag (will be added to diag)
            
            // Entry 3: Diag
            Colind[k++] = cur;
        }
    }
}

void fillGvals(double Gvals[], const double Rgrid[], SuiteSparse_long X, SuiteSparse_long Y) {
    auto G = [&](SuiteSparse_long ia, SuiteSparse_long ja, SuiteSparse_long ib, SuiteSparse_long jb) {
        return g(Rgrid[ia * Y + ja], Rgrid[ib * Y + jb]);
    };

    SuiteSparse_long k = 1; 
    for (SuiteSparse_long i = 0; i < X; i++) {
        for (SuiteSparse_long j = 0; j < Y; j++) {
            double diag = 0.0;
            
            // Connection 1: Up or Source
            if (i == 0) {
                double s = 2.0 / Rgrid[i * Y + j];
                Gvals[k++] = -s; diag += s;
            } else {
                double a = G(i, j, i - 1, j);
                Gvals[k++] = -a; diag += a;
            }

            // Connection 2: Left or Dummy
            if (j > 0) {
                double l = G(i, j, i, j - 1);
                Gvals[k++] = -l; diag += l;
            } else {
                Gvals[k++] = 0.0; // Dummy
            }

            // Connection 3: Diag (accumulate right/down)
            if (i < X - 1) diag += G(i, j, i + 1, j);
            else diag += 2.0 / Rgrid[i * Y + j]; // Sink
            
            if (j < Y - 1) diag += G(i, j, i, j + 1);
            
            Gvals[k++] = diag;
        }
    }
}

void fillGvalsSourceDiag(double Gvals[], SuiteSparse_long Y) {
    double sum = 0.0;
    // The source node (0) connects to Gvals[1], [4], [7]... for the first row
    for (SuiteSparse_long j = 0; j < Y; j++) {
        sum += Gvals[1 + (j * 3)];
    }
    Gvals[0] = -sum;
}

#endif