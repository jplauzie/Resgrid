//#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>
#include <iomanip>
#include <omp.h>
#include <random>

using std::cout;
using std::endl;
using std::setw;
using std::fixed;

// ============================================================================
// Circuit topology
// ============================================================================
// Node 0       : source bus bar (V=1). Connected to every first-row grid node
//                via 2/R off-diagonal entries (both row and column).
// Nodes 1..X*Y : grid nodes, node_idx(i,j) = 1 + i*Y + j.
// Eliminated   : sink bus bar (V=0). Its 2/R conductances to last-row nodes
//                contribute ONLY to those nodes' diagonals — there is no
//                off-diagonal column for it in the matrix.
//
// Result: the matrix is strictly SPD (not singular), suitable for Cholesky.
// RHS b[0]=1 sets the source voltage; all other b entries are 0.
// ============================================================================

#define CHECK_CUDA(func)                                                        \
{                                                                               \
    cudaError_t status = (func);                                                \
    if (status != cudaSuccess) {                                                \
        printf("CUDA API failed at line %d with error: %s (%d)\n",              \
               __LINE__, cudaGetErrorString(status), status);                   \
        return EXIT_FAILURE;                                                    \
    }                                                                           \
}

#define CHECK_CUSPARSE(func)                                                    \
{                                                                               \
    cusparseStatus_t status = (func);                                           \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                    \
        printf("CUSPARSE API failed at line %d with error: %s (%d)\n",          \
               __LINE__, cusparseGetErrorString(status), status);               \
        return EXIT_FAILURE;                                                    \
    }                                                                           \
}

#define CUSOLVER_CHECK(err)                                                     \
    do {                                                                        \
        cusolverStatus_t err_ = (err);                                          \
        if (err_ != CUSOLVER_STATUS_SUCCESS) {                                  \
            printf("cusolver error %d at %s:%d\n", err_, __FILE__, __LINE__);   \
            throw std::runtime_error("cusolver error");                         \
        }                                                                       \
    } while (0)


void printArray(double A[], int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++)
            cout << setw(5) << std::fixed << std::setprecision(3) << A[i*N+j] << " ";
        cout << endl;
    }
}
void printrows(int A[], int N) {
    for (int i = 0; i < N; i++) cout << A[i] << " ";
}
void printrowsfloat(double A[], int N) {
    for (int i = 0; i < N; i++) cout << A[i] << " ";
}
void printGArray(double A[], int N) {
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            cout << A[i*N+j] << " ";
}

// ---------------------------------------------------------------------------
// Grid initialisation
// ---------------------------------------------------------------------------
void Tgridset(double Tgrid[], int X, int Y)
{
    const double mean   = 345.0;
    const double stddev = 6.0;
    static std::mt19937 gen(41);
    static std::normal_distribution<double> dist(mean, stddev);
    for (int i = 0; i < X; i++)
        for (int j = 0; j < Y; j++)
            Tgrid[i*Y+j] = dist(gen);
}

void Rgridset(double Rgrid[], int X, int Y, double Rins)
{
    for (int i = 0; i < X; i++)
        for (int j = 0; j < Y; j++)
            Rgrid[i*Y+j] = Rins;
}

void Rgridupdater(double Rgrid[], double Tgrid[], int X, int Y, int T, double Rmetal)
{
    for (int i = 0; i < X; i++)
        for (int j = 0; j < Y; j++)
            if (T >= Tgrid[i*Y+j])
                Rgrid[i*Y+j] = Rmetal;
}

// ---------------------------------------------------------------------------
// Series conductance: g(a,b) = 2/(a+b)
// ---------------------------------------------------------------------------
static inline double g(double a, double b) { return 2.0 / (a + b); }

// ---------------------------------------------------------------------------
// Matrix index for grid node (i,j).  Node 0 is the source bus bar.
// ---------------------------------------------------------------------------
static inline int node_idx(int i, int j, int Y) { return 1 + i*Y + j; }

// ---------------------------------------------------------------------------
// compute_nnz2
//
// Number of lower-triangle entries (incl. diagonal) for the
// (X*Y+1) x (X*Y+1) admittance matrix.
//
// Per section:
//   Source node  (k=0)          :  1
//   First row    (i=0)          :  2 + 3*(Y-1)
//   Middle rows  (i=1..X-2)     :  (X-2)*(2+3*(Y-1))
//   Last row     (i=X-1)        :  2 + 3*(Y-1)   <-- NO source off-diag
//
// Total = 1 + X*(2+3*(Y-1)) = 1 + X*(3Y-1)
// ---------------------------------------------------------------------------
static inline int compute_nnz2(int X, int Y)
{
    return 1 + X * (3*Y - 1);
}

// ---------------------------------------------------------------------------
// constructRowlind / constructColind
//
// Fill row and column index arrays for the lower-triangle triplet.
// Both follow the same k-order as fillGvals.
//
// k-layout:
//   k=0                         : (0,0)                        source diag
//
//   First row (i=0):
//     k=1                       : (node(0,0), 0)               src off-diag
//     k=2                       : (node(0,0), node(0,0))       diag
//     j=1..Y-1:
//       3+(j-1)*3+0             : (node(0,j), node(0,j-1))     left
//       3+(j-1)*3+1             : (node(0,j), 0)               src off-diag
//       3+(j-1)*3+2             : (node(0,j), node(0,j))       diag
//
//   Middle row block b (b=0..X-3, grid row i=b+1):
//     base = 1+(b+1)*(2+3*(Y-1))
//     base+0                    : (node(i,0), node(i-1,0))     above
//     base+1                    : (node(i,0), node(i,0))       diag
//     j=1..Y-1:
//       base+2+(j-1)*3+0        : (node(i,j), node(i-1,j))    above
//       base+2+(j-1)*3+1        : (node(i,j), node(i,j-1))    left
//       base+2+(j-1)*3+2        : (node(i,j), node(i,j))      diag
//
//   Last row (i=X-1):
//     base = 1+(X-1)*(2+3*(Y-1))
//     base+0                    : (node(X-1,0), node(X-2,0))   above
//     base+1                    : (node(X-1,0), node(X-1,0))   diag
//     j=1..Y-1:
//       base+2+(j-1)*3+0        : (node(X-1,j), node(X-2,j))  above
//       base+2+(j-1)*3+1        : (node(X-1,j), node(X-1,j-1))left
//       base+2+(j-1)*3+2        : (node(X-1,j), node(X-1,j))  diag
// ---------------------------------------------------------------------------
void constructRowlind(int Rowlind[], int /*nnz*/, int X, int Y)
{
    int k = 0;

    Rowlind[k++] = 0;  // source diagonal

    // First row
    Rowlind[k++] = node_idx(0, 0, Y);
    Rowlind[k++] = node_idx(0, 0, Y);
    for (int j = 1; j < Y; j++) {
        Rowlind[k++] = node_idx(0, j, Y);
        Rowlind[k++] = node_idx(0, j, Y);
        Rowlind[k++] = node_idx(0, j, Y);
    }

    // Middle rows
    for (int i = 1; i <= X-2; i++) {
        Rowlind[k++] = node_idx(i, 0, Y);
        Rowlind[k++] = node_idx(i, 0, Y);
        for (int j = 1; j < Y; j++) {
            Rowlind[k++] = node_idx(i, j, Y);
            Rowlind[k++] = node_idx(i, j, Y);
            Rowlind[k++] = node_idx(i, j, Y);
        }
    }

    // Last row
    Rowlind[k++] = node_idx(X-1, 0, Y);
    Rowlind[k++] = node_idx(X-1, 0, Y);
    for (int j = 1; j < Y; j++) {
        Rowlind[k++] = node_idx(X-1, j, Y);
        Rowlind[k++] = node_idx(X-1, j, Y);
        Rowlind[k++] = node_idx(X-1, j, Y);
    }
}

void constructColind(int Colind[], int /*nnz*/, int X, int Y)
{
    int k = 0;

    Colind[k++] = 0;  // source diagonal

    // First row
    Colind[k++] = 0;                         // src off-diag
    Colind[k++] = node_idx(0, 0, Y);         // diag
    for (int j = 1; j < Y; j++) {
        Colind[k++] = node_idx(0, j-1, Y);  // left
        Colind[k++] = 0;                     // src off-diag
        Colind[k++] = node_idx(0, j,  Y);   // diag
    }

    // Middle rows
    for (int i = 1; i <= X-2; i++) {
        Colind[k++] = node_idx(i-1, 0, Y);  // above
        Colind[k++] = node_idx(i,   0, Y);  // diag
        for (int j = 1; j < Y; j++) {
            Colind[k++] = node_idx(i-1, j,   Y);  // above
            Colind[k++] = node_idx(i,   j-1, Y);  // left
            Colind[k++] = node_idx(i,   j,   Y);  // diag
        }
    }

    // Last row
    Colind[k++] = node_idx(X-2, 0, Y);  // above
    Colind[k++] = node_idx(X-1, 0, Y);  // diag
    for (int j = 1; j < Y; j++) {
        Colind[k++] = node_idx(X-2, j,   Y);  // above
        Colind[k++] = node_idx(X-1, j-1, Y);  // left
        Colind[k++] = node_idx(X-1, j,   Y);  // diag
    }
}

// ---------------------------------------------------------------------------
// fillGvals — compute conductance values in the same k-order as the index
// arrays.  Call fillGvalsSourceDiag() afterwards to set Gvals[0].
// ---------------------------------------------------------------------------
void fillGvals(double Gvals[], const double Rgrid[], int X, int Y)
{
    // Helper lambdas
    auto R = [&](int i, int j) -> double { return Rgrid[i*Y + j]; };
    auto G = [&](int ia, int ja, int ib, int jb) -> double {
        return g(Rgrid[ia*Y+ja], Rgrid[ib*Y+jb]);
    };

    int k = 0;
    k++;  // Gvals[0] filled by fillGvalsSourceDiag

    // ------------------------------------------------------------------
    // First row (i=0): source off-diagonals present
    // ------------------------------------------------------------------
    {
        double src  = 2.0 / R(0,0);
        double rgt  = (Y > 1) ? G(0,0, 0,1) : 0.0;
        double blw  = G(0,0, 1,0);
        Gvals[k++] = -src;                 // off-diag to source (node 0)
        Gvals[k++] =  src + rgt + blw;    // diagonal

        for (int j = 1; j < Y; j++) {
            double lft  = G(0,j, 0,j-1);
            double rgt2 = (j < Y-1) ? G(0,j, 0,j+1) : 0.0;
            double blw2 = G(0,j, 1,j);
            double src2 = 2.0 / R(0,j);
            Gvals[k++] = -lft;                       // left
            Gvals[k++] = -src2;                      // off-diag to source
            Gvals[k++] =  lft + rgt2 + blw2 + src2; // diagonal
        }
    }

    // ------------------------------------------------------------------
    // Middle rows (i=1..X-2): no bus bar off-diagonal
    // ------------------------------------------------------------------
    for (int i = 1; i <= X-2; i++) {
        double abv = G(i,0, i-1,0);
        double rgt = (Y > 1) ? G(i,0, i,1) : 0.0;
        double blw = G(i,0, i+1,0);
        Gvals[k++] = -abv;
        Gvals[k++] =  abv + rgt + blw;

        for (int j = 1; j < Y; j++) {
            double abv2 = G(i,j, i-1,j);
            double lft2 = G(i,j, i,j-1);
            double rgt2 = (j < Y-1) ? G(i,j, i,j+1) : 0.0;
            double blw2 = G(i,j, i+1,j);
            Gvals[k++] = -abv2;
            Gvals[k++] = -lft2;
            Gvals[k++] =  abv2 + lft2 + rgt2 + blw2;
        }
    }

    // ------------------------------------------------------------------
    // Last row (i=X-1): sink bus bar is DIAGONAL ONLY (node eliminated).
    // The 2/R term appears only in the diagonal, not as an off-diagonal
    // entry — there is no column for the sink node in the matrix.
    // ------------------------------------------------------------------
    {
        double abv  = G(X-1,0, X-2,0);
        double rgt  = (Y > 1) ? G(X-1,0, X-1,1) : 0.0;
        double sink = 2.0 / R(X-1,0);
        Gvals[k++] = -abv;
        Gvals[k++] =  abv + rgt + sink;   // diagonal absorbs sink term

        for (int j = 1; j < Y; j++) {
            double abv2  = G(X-1,j, X-2,j);
            double lft2  = G(X-1,j, X-1,j-1);
            double rgt2  = (j < Y-1) ? G(X-1,j, X-1,j+1) : 0.0;
            double sink2 = 2.0 / R(X-1,j);
            Gvals[k++] = -abv2;
            Gvals[k++] = -lft2;
            Gvals[k++] =  abv2 + lft2 + rgt2 + sink2;
        }
    }
}

// ---------------------------------------------------------------------------
// fillGvalsSourceDiag — sets Gvals[0] = -sum of source off-diagonal entries.
// Must be called AFTER fillGvals().
//
// Source off-diagonal entries live at:
//   k=1 (node(0,0)) and k=3+(j-1)*3+1 for j=1..Y-1
// ---------------------------------------------------------------------------
void fillGvalsSourceDiag(double Gvals[], int /*X*/, int Y)
{
    double sum = Gvals[1];  // node(0,0)
    for (int j = 1; j < Y; j++)
        sum += Gvals[3 + (j-1)*3 + 1];
    Gvals[0] = -sum;
}

// ---------------------------------------------------------------------------
// Merge sort (unchanged)
// ---------------------------------------------------------------------------
void merge(float array[], int const left, int const mid, int const right)
{
    int n1 = mid - left + 1, n2 = right - mid;
    auto* L = new float[n1]; auto* R = new float[n2];
    for (int i = 0; i < n1; i++) L[i] = array[left + i];
    for (int j = 0; j < n2; j++) R[j] = array[mid + 1 + j];
    int i = 0, j = 0, m = left;
    while (i < n1 && j < n2) array[m++] = (L[i] <= R[j]) ? L[i++] : R[j++];
    while (i < n1) array[m++] = L[i++];
    while (j < n2) array[m++] = R[j++];
    delete[] L; delete[] R;
}

void mergeSort(float array[], int const begin, int const end)
{
    if (begin >= end) return;
    int mid = begin + (end - begin) / 2;
    mergeSort(array, begin, mid);
    mergeSort(array, mid + 1, end);
    merge(array, begin, mid, end);
}
