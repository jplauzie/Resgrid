#ifndef HEADER_H
#define HEADER_H

#include <stdint.h>
#include <random>



typedef int64_t SuiteSparse_long;

static inline double g(double a, double b) { return 2.0 / (a + b); }
static inline SuiteSparse_long node_idx(SuiteSparse_long i, SuiteSparse_long j, SuiteSparse_long Y) { 
    return 1 + i * Y + j; 
}

// 1 source diag + 3 entries per grid node + Y sink off-diagonals + 1 sink diag.
// The sink off-diagonals connect each last-row node to the explicit sink node,
// replacing the implicit 2/R drain-to-ground that was folded into their diagonal.
static inline SuiteSparse_long compute_nnz2(SuiteSparse_long X, SuiteSparse_long Y) {
    return 2 + (X * Y * 3) + Y;
}

// sink_idx: the explicit sink node, one past the last grid node
static inline SuiteSparse_long sink_idx(SuiteSparse_long X, SuiteSparse_long Y) {
    return X * Y + 1;
}

void Tgridset(double Tgrid[], SuiteSparse_long X, SuiteSparse_long Y) {
    static std::mt19937 gen(41);
    static std::normal_distribution<double> dist(345.0, 10.0);
    for (SuiteSparse_long i = 0; i < X * Y; i++) Tgrid[i] = dist(gen);
}

void Rgridset(double Rgrid[], SuiteSparse_long X, SuiteSparse_long Y, double Rins) {
    for (SuiteSparse_long i = 0; i < X * Y; i++) Rgrid[i] = Rins;
}

void constructRowlind(SuiteSparse_long Rowlind[], SuiteSparse_long X, SuiteSparse_long Y) {
    SuiteSparse_long k = 0;
    SuiteSparse_long sink = sink_idx(X, Y);

    Rowlind[k++] = 0; // Source diagonal

    for (SuiteSparse_long i = 0; i < X; i++) {
        for (SuiteSparse_long j = 0; j < Y; j++) {
            SuiteSparse_long cur = node_idx(i, j, Y);
            Rowlind[k++] = cur; // Entry 1: Connection above (or source)
            Rowlind[k++] = cur; // Entry 2: Connection left (or dummy)
            Rowlind[k++] = cur; // Entry 3: Diagonal
        }
    }

    // Sink off-diagonals: sink row, last-row-node col (lower triangle: row > col)
    for (SuiteSparse_long j = 0; j < Y; j++)
        Rowlind[k++] = sink; // row = sink (larger index)

    Rowlind[k++] = sink; // Sink diagonal
}

void constructColind(SuiteSparse_long Colind[], SuiteSparse_long X, SuiteSparse_long Y) {
    SuiteSparse_long k = 0;
    SuiteSparse_long sink = sink_idx(X, Y);

    Colind[k++] = 0; // Source diagonal

    for (SuiteSparse_long i = 0; i < X; i++) {
        for (SuiteSparse_long j = 0; j < Y; j++) {
            SuiteSparse_long cur = node_idx(i, j, Y);

            // Entry 1: Above or Source
            if (i == 0) Colind[k++] = 0;
            else        Colind[k++] = node_idx(i - 1, j, Y);

            // Entry 2: Left or Dummy
            if (j > 0) Colind[k++] = node_idx(i, j - 1, Y);
            else       Colind[k++] = cur; // dummy: same as diag, contributes 0

            // Entry 3: Diagonal
            Colind[k++] = cur;
        }
    }

    // Sink off-diagonals: col = last-row node (smaller index, lower triangle)
    for (SuiteSparse_long j = 0; j < Y; j++)
        Colind[k++] = node_idx(X - 1, j, Y);

    Colind[k++] = sink; // Sink diagonal
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

            // Connection 3: Diagonal (accumulate right/down neighbours)
            if (i < X - 1) diag += G(i, j, i + 1, j);
            // NOTE: last row (i == X-1) no longer adds implicit 2/R drain to ground.
            // That connection is now an explicit off-diagonal entry to the sink node.

            if (j < Y - 1) diag += G(i, j, i, j + 1);

            Gvals[k++] = diag;
        }
    }

    // Sink off-diagonals: -conductance from each last-row node to sink
    double sink_diag = 0.0;
    for (SuiteSparse_long j = 0; j < Y; j++) {
        double s = 2.0 / Rgrid[(X - 1) * Y + j];
        Gvals[k++] = -s;
        sink_diag += s;
    }

    // Sink diagonal: positive sum of all sink conductances.
    // The sink is pinned to 0V in main by replacing this row with a unit row,
    // so this value will be overwritten there — but we fill it correctly here
    // for consistency (and so fillGvals is correct standalone).
    Gvals[k++] = sink_diag;
}

void fillGvalsSourceDiag(double Gvals[], SuiteSparse_long Y) {
    double sum = 0.0;
    // Source node connects to Gvals[1], [4], [7]... for the first row.
    // These are the Entry 1 values (i==0), which are -s (negative).
    for (SuiteSparse_long j = 0; j < Y; j++)
        sum += Gvals[1 + (j * 3)];
    Gvals[0] = -sum; // negate negatives → positive source diagonal
}

// ---------------------------------------------------------------------------
// updateAx: recomputes all conductance values from Rgrid and writes them
// directly into the CSC value array Ax, bypassing Gvals entirely.
//
// Sign convention:
//   - Off-diagonal entries: -conductance (negative)
//   - Diagonal entries:      +sum of all conductances at that node (positive)
//   - Source diagonal:       +sum of all first-row source conductances (positive)
//   - Sink diagonal:         filled correctly here, then overwritten with 1.0
//                            in main to pin sink voltage to zero
//
// Duplicate handling: the j==0 dummy left-connection shares a CSC slot with
// the diagonal. We skip writing it (value is 0.0) and just advance k.
// The std::fill ensures that slot starts at zero before the diagonal += runs.
// ---------------------------------------------------------------------------
static void updateAx(
    double* Ax,
    SuiteSparse_long nzmax,
    const double* Rgrid,
    const SuiteSparse_long* triplet_to_csc,
    SuiteSparse_long X,
    SuiteSparse_long Y)
{
    std::fill(Ax, Ax + nzmax, 0.0);

    SuiteSparse_long k = 1; // k=0 is source diagonal, handled separately below

    for (SuiteSparse_long i = 0; i < X; i++) {
        for (SuiteSparse_long j = 0; j < Y; j++) {
            double diag = 0.0;

            // Entry 1: up or source
            if (i == 0) {
                double s = 2.0 / Rgrid[i * Y + j];
                Ax[triplet_to_csc[k++]] = -s;
                diag += s;
            } else {
                double a = g(Rgrid[i * Y + j], Rgrid[(i - 1) * Y + j]);
                Ax[triplet_to_csc[k++]] = -a;
                diag += a;
            }

            // Entry 2: left or dummy (j==0 dummy: skip write, advance k)
            if (j > 0) {
                double l = g(Rgrid[i * Y + j], Rgrid[i * Y + (j - 1)]);
                Ax[triplet_to_csc[k++]] = -l;
                diag += l;
            } else {
                k++;
            }

            // Entry 3: diagonal — accumulate right and down neighbours
            if (i < X - 1) diag += g(Rgrid[i * Y + j], Rgrid[(i + 1) * Y + j]);
            // Last row: no implicit drain — explicit sink entries handle this below

            if (j < Y - 1) diag += g(Rgrid[i * Y + j], Rgrid[i * Y + (j + 1)]);

            Ax[triplet_to_csc[k++]] += diag;
        }
    }

    // Source diagonal
    double source_diag = 0.0;
    for (SuiteSparse_long j = 0; j < Y; j++)
        source_diag += 2.0 / Rgrid[j];
    Ax[triplet_to_csc[0]] = source_diag;

    // Sink off-diagonals and sink diagonal
    double sink_diag = 0.0;
    for (SuiteSparse_long j = 0; j < Y; j++) {
        double s = 2.0 / Rgrid[(X - 1) * Y + j];
        Ax[triplet_to_csc[k++]] = -s;
        sink_diag += s;
    }
    // Sink diagonal written here; main will overwrite with 1.0 to pin to 0V.
    Ax[triplet_to_csc[k++]] = sink_diag;
}

#endif