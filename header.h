#ifndef HEADER_H
#define HEADER_H

#include <petscsys.h>
#include <random>
#include <queue>
#include <vector>
#include <set>
#include <cmath>
#include <algorithm>

// PetscInt replaces SuiteSparse_long throughout
static inline double g(double a, double b) { return 2.0 / (a + b); }

static inline PetscInt node_idx(PetscInt i, PetscInt j, PetscInt Y) {
    return 1 + i * Y + j;
}

static inline PetscInt compute_nnz2(PetscInt X, PetscInt Y) {
    return 1 + (X * Y * 3);
}

void Tgridset(double Tgrid[], PetscInt X, PetscInt Y, double mean, double std_dev) {
    static std::mt19937 gen(41);
    std::normal_distribution<double> dist(mean, std_dev);
    for (PetscInt i = 0; i < X * Y; i++) Tgrid[i] = dist(gen);
}

void Rgridset(double Rgrid[], PetscInt X, PetscInt Y, double Rins) {
    for (PetscInt i = 0; i < X * Y; i++) Rgrid[i] = Rins;
}

void constructRowlind(PetscInt Rowlind[], PetscInt X, PetscInt Y) {
    PetscInt k = 0;
    Rowlind[k++] = 0;
    for (PetscInt i = 0; i < X; i++) {
        for (PetscInt j = 0; j < Y; j++) {
            PetscInt cur = node_idx(i, j, Y);
            Rowlind[k++] = cur; Rowlind[k++] = cur; Rowlind[k++] = cur;
        }
    }
}

void constructColind(PetscInt Colind[], PetscInt X, PetscInt Y) {
    PetscInt k = 0;
    Colind[k++] = 0;
    for (PetscInt i = 0; i < X; i++) {
        for (PetscInt j = 0; j < Y; j++) {
            PetscInt cur = node_idx(i, j, Y);
            if (i == 0) Colind[k++] = 0;
            else        Colind[k++] = node_idx(i - 1, j, Y);
            if (j > 0) Colind[k++] = node_idx(i, j - 1, Y);
            else       Colind[k++] = cur;
            Colind[k++] = cur;
        }
    }
}

// Computes the lower-triangle nonzero values for the conductance matrix.
// Values are indexed identically to Rowlind/Colind from constructRowlind/constructColind.
// The caller (main.cpp) symmetrizes when inserting into the PETSc Mat.
static void computeAxValues(double* values, const double* Rgrid, PetscInt X, PetscInt Y) {
    PetscInt k = 1;

    // Source node diagonal (node 0): sum of conductances to first row
    double source_diag = 0.0;
    for (PetscInt j = 0; j < Y; j++) source_diag += 2.0 / Rgrid[j];
    values[0] = source_diag;

    for (PetscInt i = 0; i < X; i++) {
        for (PetscInt j = 0; j < Y; j++) {
            double diag = 0.0;

            // Off-diagonal: connection upward (to source node 0 if i==0)
            if (i == 0) {
                double s = 2.0 / Rgrid[i * Y + j];
                values[k++] = -s;
                diag += s;
            } else {
                double a = g(Rgrid[i * Y + j], Rgrid[(i - 1) * Y + j]);
                values[k++] = -a;
                diag += a;
            }

            // Off-diagonal: connection left
            if (j > 0) {
                double l = g(Rgrid[i * Y + j], Rgrid[i * Y + (j - 1)]);
                values[k++] = -l;
                diag += l;
            } else {
                values[k++] = 0.0;  // placeholder (diagonal duplicate in original)
            }

            // Diagonal: accumulate contributions from all 4 neighbors
            if (i < X - 1) diag += g(Rgrid[i * Y + j], Rgrid[(i + 1) * Y + j]);
            else            diag += 2.0 / Rgrid[(X - 1) * Y + j];  // bottom boundary
            if (j < Y - 1) diag += g(Rgrid[i * Y + j], Rgrid[i * Y + (j + 1)]);

            values[k++] = diag;
        }
    }
}

// --- TWO-MAP CASCADE LOGIC (unchanged) ---

struct Grain {
    PetscInt idx;
    double threshold;
    bool operator>(const Grain& other) const { return threshold > other.threshold; }
    bool operator<(const Grain& other) const { return threshold < other.threshold; }
};

void precomputeHeatingMap(double Tgrid[], PetscInt X, PetscInt Y, double J_total) {
    const double step_J = J_total / 4.0;
    std::vector<double> eff_T(X * Y);
    std::vector<bool> swapped(X * Y, false);
    std::priority_queue<Grain, std::vector<Grain>, std::greater<Grain>> pq;
    for (PetscInt i = 0; i < X * Y; i++) {
        eff_T[i] = Tgrid[i];
        pq.push({i, eff_T[i]});
    }
    double current_max = 0.0;
    while (!pq.empty()) {
        Grain top = pq.top(); pq.pop();
        if (swapped[top.idx]) continue;
        swapped[top.idx] = true;
        current_max = std::max(current_max, top.threshold);
        Tgrid[top.idx] = current_max;
        PetscInt r = top.idx / Y, c = top.idx % Y;
        PetscInt neighbors[4] = { (r>0)?(r-1)*Y+c:-1, (r<X-1)?(r+1)*Y+c:-1, (c>0)?r*Y+(c-1):-1, (c<Y-1)?r*Y+(c+1):-1 };
        for (int n = 0; n < 4; n++) {
            if (neighbors[n] != -1 && !swapped[neighbors[n]]) {
                eff_T[neighbors[n]] -= step_J;
                pq.push({neighbors[n], eff_T[neighbors[n]]});
            }
        }
    }
}

void precomputeCoolingMap(double Tgrid[], PetscInt X, PetscInt Y, double J_total) {
    const double step_J = J_total / 4.0;
    std::vector<double> eff_T(X * Y);
    std::vector<bool> reverted(X * Y, false);
    std::priority_queue<Grain, std::vector<Grain>, std::less<Grain>> pq;
    for (PetscInt i = 0; i < X * Y; i++) {
        eff_T[i] = Tgrid[i] - J_total;
        pq.push({i, eff_T[i]});
    }
    double current_min = 1e9;
    while (!pq.empty()) {
        Grain top = pq.top(); pq.pop();
        if (reverted[top.idx]) continue;
        reverted[top.idx] = true;
        if (current_min == 1e9) current_min = top.threshold;
        else current_min = std::min(current_min, top.threshold);
        Tgrid[top.idx] = current_min;
        PetscInt r = top.idx / Y, c = top.idx % Y;
        PetscInt neighbors[4] = { (r>0)?(r-1)*Y+c:-1, (r<X-1)?(r+1)*Y+c:-1, (c>0)?r*Y+(c-1):-1, (c<Y-1)?r*Y+(c+1):-1 };
        for (int n = 0; n < 4; n++) {
            if (neighbors[n] != -1 && !reverted[neighbors[n]]) {
                eff_T[neighbors[n]] += step_J;
                pq.push({neighbors[n], eff_T[neighbors[n]]});
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
        double t = (biased < 0) ? target + biased * (target - min_t) : target + biased * (max_t - target);
        int val = static_cast<int>(std::round(t));
        while (temps.count(val) && val < max_t) { val++; }
        if (val <= max_t) temps.insert(val);
    }
    return temps;
}

double getSemiconductorR(double T) {
    const double R_ref = 1000.0;
    const double T_ref = 300.0;
    const double alpha = 3500.0;
    return R_ref * std::exp(alpha * (1.0 / T - 1.0 / T_ref));
}



#endif