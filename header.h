#ifndef HEADER_H
#define HEADER_H

#include <stdint.h>
#include <random>
#include <queue>
#include <vector>
#include <set>
#include <cmath>
#include <algorithm>

typedef int64_t SuiteSparse_long;

static inline double g(double a, double b) { return 2.0 / (a + b); }

// Node indexing: 0 is Source. Grid nodes are 1 to X*Y.
static inline SuiteSparse_long node_idx(SuiteSparse_long i, SuiteSparse_long j, SuiteSparse_long Y) { 
    return 1 + i * Y + j; 
}

// 1 source diag + 3 lower-triangular entries per grid node.
// Explicit sink is removed (handled implicitly as ground).
static inline SuiteSparse_long compute_nnz2(SuiteSparse_long X, SuiteSparse_long Y) {
    return 1 + (X * Y * 3);
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
    Rowlind[k++] = 0; // Source diagonal

    for (SuiteSparse_long i = 0; i < X; i++) {
        for (SuiteSparse_long j = 0; j < Y; j++) {
            SuiteSparse_long cur = node_idx(i, j, Y);
            Rowlind[k++] = cur; // Entry 1: Connection above (or source)
            Rowlind[k++] = cur; // Entry 2: Connection left (or dummy)
            Rowlind[k++] = cur; // Entry 3: Diagonal
        }
    }
}

void constructColind(SuiteSparse_long Colind[], SuiteSparse_long X, SuiteSparse_long Y) {
    SuiteSparse_long k = 0;
    Colind[k++] = 0; // Source diagonal

    for (SuiteSparse_long i = 0; i < X; i++) {
        for (SuiteSparse_long j = 0; j < Y; j++) {
            SuiteSparse_long cur = node_idx(i, j, Y);

            // Entry 1: Above or Source
            if (i == 0) Colind[k++] = 0;
            else        Colind[k++] = node_idx(i - 1, j, Y);

            // Entry 2: Left or Dummy
            if (j > 0) Colind[k++] = node_idx(i, j - 1, Y);
            else       Colind[k++] = cur; // dummy

            // Entry 3: Diagonal
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

            // Connection 3: Diagonal
            if (i < X - 1) {
                diag += G(i, j, i + 1, j);
            } else {
                // IMPLICIT GROUND: The connection to the sink is added to the 
                // node's diagonal, which naturally bleeds current out of the system.
                diag += 2.0 / Rgrid[(X - 1) * Y + j]; 
            }

            if (j < Y - 1) diag += G(i, j, i, j + 1);

            Gvals[k++] = diag;
        }
    }
}

void fillGvalsSourceDiag(double Gvals[], SuiteSparse_long Y) {
    double sum = 0.0;
    for (SuiteSparse_long j = 0; j < Y; j++)
        sum += Gvals[1 + (j * 3)];
    Gvals[0] = -sum;
}

static void updateAx(
    double* Ax,
    SuiteSparse_long nzmax,
    const double* Rgrid,
    const SuiteSparse_long* triplet_to_csc,
    SuiteSparse_long X,
    SuiteSparse_long Y)
{
    std::fill(Ax, Ax + nzmax, 0.0);

    SuiteSparse_long k = 1;

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

            // Entry 2: left or dummy
            if (j > 0) {
                double l = g(Rgrid[i * Y + j], Rgrid[i * Y + (j - 1)]);
                Ax[triplet_to_csc[k++]] = -l;
                diag += l;
            } else {
                k++;
            }

            // Entry 3: diagonal
            if (i < X - 1) {
                diag += g(Rgrid[i * Y + j], Rgrid[(i + 1) * Y + j]);
            } else {
                // IMPLICIT GROUND
                diag += 2.0 / Rgrid[(X - 1) * Y + j];
            }

            if (j < Y - 1) diag += g(Rgrid[i * Y + j], Rgrid[i * Y + (j + 1)]);

            Ax[triplet_to_csc[k++]] += diag;
        }
    }

    // Source diagonal
    double source_diag = 0.0;
    for (SuiteSparse_long j = 0; j < Y; j++)
        source_diag += 2.0 / Rgrid[j];
    Ax[triplet_to_csc[0]] = source_diag;
}


// --- PHYSICS PRECOMPUTATION AND SCHEDULING ---

struct Grain {
    SuiteSparse_long idx;
    double threshold;
    bool operator>(const Grain& other) const { return threshold > other.threshold; }
};

void precomputeTriggerTemps(double Tgrid[], SuiteSparse_long X, SuiteSparse_long Y, double reduction_per_neighbor) {
    const double step_reduction = reduction_per_neighbor / 4.0;
    std::vector<double> effective_T(X * Y);
    std::vector<bool> swapped(X * Y, false);
    
    std::priority_queue<Grain, std::vector<Grain>, std::greater<Grain>> pq;

    for (SuiteSparse_long i = 0; i < X * Y; i++) {
        effective_T[i] = Tgrid[i];
        pq.push({i, effective_T[i]});
    }

    double current_max_trigger = 0.0;

    while (!pq.empty()) {
        Grain top = pq.top();
        pq.pop();

        if (swapped[top.idx]) continue;
        swapped[top.idx] = true;

        current_max_trigger = std::max(current_max_trigger, top.threshold);
        Tgrid[top.idx] = current_max_trigger;

        SuiteSparse_long r = top.idx / Y;
        SuiteSparse_long c = top.idx % Y;
        SuiteSparse_long neighbors[4] = {
            (r > 0) ? (r - 1) * Y + c : -1,
            (r < X - 1) ? (r + 1) * Y + c : -1,
            (c > 0) ? r * Y + (c - 1) : -1,
            (c < Y - 1) ? r * Y + (c + 1) : -1
        };

        for (int n = 0; n < 4; n++) {
            SuiteSparse_long n_idx = neighbors[n];
            if (n_idx != -1 && !swapped[n_idx]) {
                effective_T[n_idx] -= step_reduction;
                pq.push({n_idx, effective_T[n_idx]});
            }
        }
    }
}

std::set<int> generateHighlyBiasedTemps(double min_t, double max_t, double bias_point, int num_images, double tightness = 3.0) {
    std::set<int> temps;
    double target = std::max(min_t, std::min(max_t, bias_point));

    for (int i = 0; i < num_images; i++) {
        double x = -1.0 + 2.0 * i / (double)(num_images - 1);
        double biased = (x > 0 ? 1.0 : -1.0) * std::pow(std::abs(x), tightness);
        
        double t = (biased < 0) ? target + biased * (target - min_t) 
                                : target + biased * (max_t - target);
        
        int val = static_cast<int>(std::round(t));
        
        while (temps.count(val) && val < max_t) { val++; }
        if (val <= max_t) temps.insert(val);
    }
    return temps;
}

#endif