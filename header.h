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

static inline SuiteSparse_long node_idx(SuiteSparse_long i, SuiteSparse_long j, SuiteSparse_long Y) { 
    return 1 + i * Y + j; 
}

static inline SuiteSparse_long compute_nnz2(SuiteSparse_long X, SuiteSparse_long Y) {
    return 1 + (X * Y * 3);
}

void Tgridset(double Tgrid[], SuiteSparse_long X, SuiteSparse_long Y, double mean, double std_dev) {
    static std::mt19937 gen(41);
    std::normal_distribution<double> dist(mean, std_dev);
    for (SuiteSparse_long i = 0; i < X * Y; i++) Tgrid[i] = dist(gen);
}

void Rgridset(double Rgrid[], SuiteSparse_long X, SuiteSparse_long Y, double Rins) {
    for (SuiteSparse_long i = 0; i < X * Y; i++) Rgrid[i] = Rins;
}

void constructRowlind(SuiteSparse_long Rowlind[], SuiteSparse_long X, SuiteSparse_long Y) {
    SuiteSparse_long k = 0;
    Rowlind[k++] = 0; 
    for (SuiteSparse_long i = 0; i < X; i++) {
        for (SuiteSparse_long j = 0; j < Y; j++) {
            SuiteSparse_long cur = node_idx(i, j, Y);
            Rowlind[k++] = cur; Rowlind[k++] = cur; Rowlind[k++] = cur;
        }
    }
}

void constructColind(SuiteSparse_long Colind[], SuiteSparse_long X, SuiteSparse_long Y) {
    SuiteSparse_long k = 0;
    Colind[k++] = 0; 
    for (SuiteSparse_long i = 0; i < X; i++) {
        for (SuiteSparse_long j = 0; j < Y; j++) {
            SuiteSparse_long cur = node_idx(i, j, Y);
            if (i == 0) Colind[k++] = 0;
            else        Colind[k++] = node_idx(i - 1, j, Y);
            if (j > 0) Colind[k++] = node_idx(i, j - 1, Y);
            else       Colind[k++] = cur;
            Colind[k++] = cur;
        }
    }
}

static void updateAx(double* Ax, SuiteSparse_long nzmax, const double* Rgrid, const SuiteSparse_long* triplet_to_csc, SuiteSparse_long X, SuiteSparse_long Y) {
    std::fill(Ax, Ax + nzmax, 0.0);
    SuiteSparse_long k = 1;
    for (SuiteSparse_long i = 0; i < X; i++) {
        for (SuiteSparse_long j = 0; j < Y; j++) {
            double diag = 0.0;
            if (i == 0) {
                double s = 2.0 / Rgrid[i * Y + j];
                Ax[triplet_to_csc[k++]] = -s; diag += s;
            } else {
                double a = g(Rgrid[i * Y + j], Rgrid[(i - 1) * Y + j]);
                Ax[triplet_to_csc[k++]] = -a; diag += a;
            }
            if (j > 0) {
                double l = g(Rgrid[i * Y + j], Rgrid[i * Y + (j - 1)]);
                Ax[triplet_to_csc[k++]] = -l; diag += l;
            } else k++;

            if (i < X - 1) diag += g(Rgrid[i * Y + j], Rgrid[(i + 1) * Y + j]);
            else           diag += 2.0 / Rgrid[(X - 1) * Y + j]; 
            if (j < Y - 1) diag += g(Rgrid[i * Y + j], Rgrid[i * Y + (j + 1)]);
            Ax[triplet_to_csc[k++]] += diag;
        }
    }
    double source_diag = 0.0;
    for (SuiteSparse_long j = 0; j < Y; j++) source_diag += 2.0 / Rgrid[j];
    Ax[triplet_to_csc[0]] = source_diag;
}

// --- TWO-MAP CASCADE LOGIC ---

struct Grain {
    SuiteSparse_long idx;
    double threshold;
    bool operator>(const Grain& other) const { return threshold > other.threshold; }
    bool operator<(const Grain& other) const { return threshold < other.threshold; }
};

void precomputeHeatingMap(double Tgrid[], SuiteSparse_long X, SuiteSparse_long Y, double J_total) {
    const double step_J = J_total / 4.0;
    std::vector<double> eff_T(X * Y);
    std::vector<bool> swapped(X * Y, false);
    std::priority_queue<Grain, std::vector<Grain>, std::greater<Grain>> pq;
    for (SuiteSparse_long i = 0; i < X * Y; i++) {
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
        SuiteSparse_long r = top.idx / Y, c = top.idx % Y;
        SuiteSparse_long neighbors[4] = { (r>0)?(r-1)*Y+c:-1, (r<X-1)?(r+1)*Y+c:-1, (c>0)?r*Y+(c-1):-1, (c<Y-1)?r*Y+(c+1):-1 };
        for (int n = 0; n < 4; n++) {
            if (neighbors[n] != -1 && !swapped[neighbors[n]]) {
                eff_T[neighbors[n]] -= step_J;
                pq.push({neighbors[n], eff_T[neighbors[n]]});
            }
        }
    }
}

void precomputeCoolingMap(double Tgrid[], SuiteSparse_long X, SuiteSparse_long Y, double J_total) {
    const double step_J = J_total / 4.0;
    std::vector<double> eff_T(X * Y);
    std::vector<bool> reverted(X * Y, false);
    std::priority_queue<Grain, std::vector<Grain>, std::less<Grain>> pq;
    for (SuiteSparse_long i = 0; i < X * Y; i++) {
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
        SuiteSparse_long r = top.idx / Y, c = top.idx % Y;
        SuiteSparse_long neighbors[4] = { (r>0)?(r-1)*Y+c:-1, (r<X-1)?(r+1)*Y+c:-1, (c>0)?r*Y+(c-1):-1, (c<Y-1)?r*Y+(c+1):-1 };
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

/**
 * Calculates the resistivity of the VO2 insulating phase based on 
 * semiconductor temperature dependence.
 */
double getSemiconductorR(double T) {
    // Parameters for VO2:
    const double R_ref = 1000.0;  // Resistance at T_ref
    const double T_ref = 300.0;   // Reference temperature (Kelvin)
    const double alpha = 3500.0;  // Activation energy parameter (Ea / kB)

    return R_ref * std::exp(alpha * (1.0 / T - 1.0 / T_ref));
}

#endif