//#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>
#include <iomanip>
#include <omp.h>

using std::cout;
using std::endl;
using std::setw;
using std::fixed;
//std::cout << std::setprecision(2) << std::fixed;

#define CHECK_CUDA(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        printf("CUDA API failed at line %d with error: %s (%d)\n",             \
               __LINE__, cudaGetErrorString(status), status);                  \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

#define CHECK_CUSPARSE(func)                                                   \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        printf("CUSPARSE API failed at line %d with error: %s (%d)\n",         \
               __LINE__, cusparseGetErrorString(status), status);              \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

//#define CHECK_CUSOLVER(func)                                                   \
//{                                                                              \
//    cusolverStatus_t status = (func);                                          \
//    if (status != CUSOLVER_STATUS_SUCCESS) {                                   \
//        printf("CUSOLVER API failed at line %d with error: %s (%d)\n",         \
//               __LINE__, cusolverGetErrorString(status), status);              \
//        return EXIT_FAILURE;                                                   \
//    }                                                                          \
//}

#define CUSOLVER_CHECK(err)                                                                        \
    do {                                                                                           \
        cusolverStatus_t err_ = (err);                                                             \
        if (err_ != CUSOLVER_STATUS_SUCCESS) {                                                     \
            printf("cusolver error %d at %s:%d\n", err_, __FILE__, __LINE__);                      \
            throw std::runtime_error("cusolver error");                                            \
        }                                                                                          \
    } while (0)


void printArray(double A[], int N) {
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			cout << setw(5) << std::fixed << std::setprecision(3) << A[i * N + j] << " ";
		}
		cout << endl;
	}
}

void printrows(int A[], int N) {
	for (int i = 0; i < N; i++) {
		cout << A[i] << " ";
	}
}

void printrowsfloat(double A[], int N) {
	for (int i = 0; i < N; i++) {
		cout << A[i] << " ";
	}
}


void printGArray(double A[], int N) {
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			cout << A[i * N + j] << " ";
		}
	}
}

void Tgridset(double Tgrid[], int X, int Y) {
	for (int i = 0; i < X; i++) {
		for (int j = 0; j < Y; j++) {
			Tgrid[i * Y + j] = 345;//rand()%100;
		}
	}
	return;
}

void Rgridset(double Rgrid[], int X, int Y, double Rins) {
	for (int i = 0; i < X; i++) {
		for (int j = 0; j < Y; j++) {
			Rgrid[i * Y + j] = Rins;
		}
	}
	return;
}

void Rgridupdater(double Rgrid[], double Tgrid[], int X, int Y, int T, double Rmetal) {
	for (int i = 0; i < X; i++) {
		for (int j = 0; j < Y; j++) {
			if (T >= Tgrid[i * Y + j]) {
				Rgrid[i * Y + j] = Rmetal;
			}
		}
	}
	return;
}

void Tgridupdater( double Tgrid[], int X, int Y, float T) {

	int* Tgridfake = new int[X * Y]();
	int* Tgridfake2 = new int[X * Y]();
	double couplingconst=1;

	for (int i = 0; i < X; i++) {
		for (int j = 0; j < Y; j++) {
			Tgridfake[i * Y + j] = 0;
		}
	}

	for (int i = 0; i < X*Y-1; i++) {

		if (i<X) {
			if (i%X==0) {
				//if ((Tgrid[i]> Tgrid[i+1])) {
					
					if ((Tgridfake[i]&0b00000000)==0) {
						//cout << "herezz" << (Tgridfake[i] & 0b00000001) <<endl;
						Tgrid[i] -= couplingconst;
						Tgridfake[i] = (Tgridfake[i] | 0b00000001);
						
						Tgridfake2[i + 1] = (Tgridfake[i] | 0b00000001);
					}
				//}
				if (Tgrid[i] > Tgrid[i + X]) {
					if (Tgridfake[i] == 0) {
						Tgrid[i] -= couplingconst;
						Tgridfake[i] = 3;
						Tgridfake2[i + 1] = 1;
					}

				}



			}
			if ((i % X != 0) &&(i % X != X-1)) {

			}
			if (i % X == X - 1) {

			}

		
		} 
		
		if( i>X && i< X*Y-Y) {
			if (i % X == 0) {

			}
			if ((i % X != 0) && (i % X != X - 1)) {

			}
			if (i % X == X - 1) {

			}
		
		} 
		
		if(i > X * Y - Y){
			if (i % X == 0) {

			}
			if ((i % X != 0) && (i % X != X - 1)) {

			}
			if (i % X == X - 1) {

			}
		
		}


	}



	delete(Tgridfake);
	delete(Tgridfake2);
	
	return;
}





void merge(float array[], int const left,
	int const mid, int const right)
{
	auto const subArrayOne = mid - left + 1;
	auto const subArrayTwo = right - mid;

	// Create temp arrays 
	auto* leftArray = new float[subArrayOne],
		* rightArray = new float[subArrayTwo];

	// Copy data to temp arrays leftArray[]  
	// and rightArray[] 
	for (int i = 0; i < subArrayOne; i++)
		leftArray[i] = array[left + i];
	for (int j = 0; j < subArrayTwo; j++)
		rightArray[j] = array[mid + 1 + j];

	// Initial index of first sub-array 
	// Initial index of second sub-array 
	auto indexOfSubArrayOne = 0,
		indexOfSubArrayTwo = 0;

	// Initial index of merged array 
	int indexOfMergedArray = left;

	// Merge the temp arrays back into  
	// array[left..right] 
	while (indexOfSubArrayOne < subArrayOne &&
		indexOfSubArrayTwo < subArrayTwo)
	{
		if (leftArray[indexOfSubArrayOne] <=
			rightArray[indexOfSubArrayTwo])
		{
			array[indexOfMergedArray] =
				leftArray[indexOfSubArrayOne];
			indexOfSubArrayOne++;
		}
		else
		{
			array[indexOfMergedArray] =
				rightArray[indexOfSubArrayTwo];
			indexOfSubArrayTwo++;
		}
		indexOfMergedArray++;
	}

	// Copy the remaining elements of 
	// left[], if there are any 
	while (indexOfSubArrayOne < subArrayOne)
	{
		array[indexOfMergedArray] =
			leftArray[indexOfSubArrayOne];
		indexOfSubArrayOne++;
		indexOfMergedArray++;
	}

	// Copy the remaining elements of 
	// right[], if there are any 
	while (indexOfSubArrayTwo < subArrayTwo)
	{
		array[indexOfMergedArray] =
			rightArray[indexOfSubArrayTwo];
		indexOfSubArrayTwo++;
		indexOfMergedArray++;
	}
}

// begin is for left index and end is 
// right index of the sub-array 
// of arr to be sorted */ 
void mergeSort(float array[],
	int const begin,
	int const end)
{
	// Returns recursively 
	if (begin >= end)
		return;

	auto mid = begin + (end - begin) / 2;
	mergeSort(array, begin, mid);
	mergeSort(array, mid + 1, end);
	merge(array, begin, mid, end);
}

double g(double const Rgrid[], double const  a, double const b)
{
	return 2 / (a + b);
}

void fillgvalsfirstrow(double G[], double Rgrid[], int const X, int const Y)
{
	G[1] = -2 / Rgrid[0];
	G[2] = g(Rgrid, Rgrid[0], Rgrid[0 + 1]) + g(Rgrid, Rgrid[0], Rgrid[0 + Y]) + 2 / Rgrid[0];

	int j = 0;
#pragma omp parallel for
	for (int i = 0; i < (Y - 2) * 3; i = i + 3) {
		j++;
		G[i + 3] = -2 / Rgrid[j];
		G[i + 4] = -g(Rgrid, Rgrid[j], Rgrid[j - 1]);
		G[i + 5] = g(Rgrid, Rgrid[j], Rgrid[j + 1]) + g(Rgrid, Rgrid[j], Rgrid[j + Y]) + g(Rgrid, Rgrid[j], Rgrid[j - 1]) + 2 / Rgrid[j];
	}
	j++;

	G[(Y - 1) * 3 - 2 - (Y - 1) + (Y + 1)] = -2 / Rgrid[j];
	G[(Y - 1) * 3 - 1 - (Y - 1) + (Y + 1)] = -g(Rgrid, Rgrid[j], Rgrid[j - 1]);
	G[(Y - 1) * 3 - (Y - 1) + (Y + 1)] = g(Rgrid, Rgrid[j], Rgrid[j - 1]) + g(Rgrid, Rgrid[j], Rgrid[j + Y]) + 2 / Rgrid[j];
}

void fillgvalsfirselement(double G[], double Rgrid[], int const X, int const Y)
{
	G[0] = -G[1];
	for (int i = 3; i < (Y - 1) * 3 + 1; i = i + 3) {
		G[0] = G[0] - G[i];
	}
}

void fillgvalsmid(double G[], double Rgrid[], int const X, int const Y, int nnz)
{
	int offset1 = 1 + (Y - 1) * 2 + (Y + 1);//(Y + 1) + 3 + ((Y - 1) * 3);
	int j = Y;
	//#pragma omp parallel for
	for (int k = 0; k < X - 2; k++) {

		int i = 0;
		offset1 = 1 + (Y - 1) * 2 + (Y + 1) - 2 * k;
		G[(k) * ((Y - 2) * 3 + 7) + offset1] = -g(Rgrid, Rgrid[j + Y * k + i], Rgrid[j + Y * k + i - Y]);
		G[(k) * ((Y - 2) * 3 + 7) + offset1 + 1] = g(Rgrid, Rgrid[j + Y * k + i], Rgrid[j + Y * k + i - Y]) + g(Rgrid, Rgrid[j + Y * k + i], Rgrid[j + Y * k + i + 1]) + g(Rgrid, Rgrid[j + Y * k + i], Rgrid[j + Y * k + i + Y]);

		int c = (k) * ((Y - 2) * 3 + 7);
		int d = j + Y * k;
		
		#pragma ivdep
		for (int i = 1; i < Y - 1; i++) {


			//cout << "offset2:" << offset1 << endl;
			//cout << "k:" << k << endl;
			//cout << "i:" << i << endl;
			offset1 = 1 + (Y - 1) * 2 + (Y + 1) - 2 * k - 1;
			//cout << "offset22:" << offset1 << endl;
			double h= Rgrid[d + i];
			double a = g(Rgrid, h, Rgrid[d + i - Y]);
			double b = g(Rgrid, h, Rgrid[d + i - 1]);
			int f = i * 3;
			
			G[c + f + offset1] = -a;
			G[c + f + offset1 + 1] = -b;
			G[c + f + offset1 + 2] = a + b + g(Rgrid, h, Rgrid[d + i + 1]) + g(Rgrid, h, Rgrid[d + i + Y]);

		}

		i = Y - 1;
		offset1 = 1 + (Y - 1) * 2 + (Y + 1) - 2 * k - 1;
		G[(k) * ((Y - 2) * 3 + 7) + i * 3 + offset1] = -g(Rgrid, Rgrid[j + Y * k + i], Rgrid[j + Y * k + i - Y]);
		G[(k) * ((Y - 2) * 3 + 7) + i * 3 + offset1 + 1] = -g(Rgrid, Rgrid[j + Y * k + i], Rgrid[j + Y * k + i - 1]);
		G[(k) * ((Y - 2) * 3 + 7) + i * 3 + offset1 + 2] = g(Rgrid, Rgrid[j + Y * k + i], Rgrid[j + Y * k + i - Y]) + g(Rgrid, Rgrid[j + Y * k + i], Rgrid[j + Y * k + i - 1]) + g(Rgrid, Rgrid[j + Y * k + i], Rgrid[j + Y * k + i + Y]);


	}
}

void fillgvalslastrow(double G[], double Rgrid[], int const X, int const Y, int nnz)
{
	G[(nnz - 1) + (Y + 1) - 4 * (Y - 1) - 3] = -g(Rgrid, Rgrid[X * Y - 1 - (Y - 1)], Rgrid[X * Y - 1 - (Y - 1) - Y]);
	G[(nnz - 1) + (Y + 1) - 4 * (Y - 1) - 2] = g(Rgrid, Rgrid[X * Y - 1 - (Y - 1)], Rgrid[X * Y - 1 - (Y - 1) + 1]) + g(Rgrid, Rgrid[X * Y - 1 - (Y - 1)], Rgrid[X * Y - 1 - (Y - 1) - Y]) + 2 / Rgrid[X * Y - 1 - (Y - 1)];

	int j = 0;
	int offset2 = (nnz - 1) + (Y + 1) - 4 * (Y - 1) - 2 - 2;

#pragma omp parallel for
	for (int i = 2; i < Y; i++) {
		G[offset2 + (i - 2) * 3 + 3] = -g(Rgrid, Rgrid[X * Y - 1 + (-(Y - 1) + i - 1)], Rgrid[X * Y - 1 - Y + (-(Y - 1) + i - 1)]);
		G[offset2 + (i - 2) * 3 + 4] = -g(Rgrid, Rgrid[X * Y - 1 + (-(Y - 1) + i - 1)], Rgrid[X * Y - 1 + (-(Y - 1) + i - 1) - 1]);
		G[offset2 + (i - 2) * 3 + 5] = g(Rgrid, Rgrid[X * Y - 1 + (-(Y - 1) + i - 1)], Rgrid[X * Y - 1 + (-(Y - 1) + i - 1) - 1]) + g(Rgrid, Rgrid[X * Y - 1 + (-(Y - 1) + i - 1)], Rgrid[X * Y - 1 + 1 + (-(Y - 1) + i - 1)]) + g(Rgrid, Rgrid[X * Y - 1 + (-(Y - 1) + i - 1)], Rgrid[X * Y - 1 - Y + (-(Y - 1) + i - 1)]) + 2 / Rgrid[X * Y - 1 + (-(Y - 1) + i - 1)];
		//G[offset2 + (i - 2) * 5 + 4] = -g(Rgrid, Rgrid[X * Y - 1 + (-(Y - 1) + i)], Rgrid[X * Y - 1 + (-(Y - 1) + i) - 1]);
		//cout << "here11:"  << X * Y - 1 - (i - 1) << endl;
		//cout << "here22:" << X * Y - 1 + (-(Y - 1) + i -1) << endl;
		//G[offset2 + (i - 2) * 5 + 5] = -2 / Rgrid[X * Y - 1 + (-(Y - 1) + i - 1)]; //+(-(Y - 1) + i + 1)
		j++;
		//int a = omp_get_thread_num();
		//cout << "Hello from thread: " << a << "\n";
	}
	j++;

	G[nnz - 1 - 2] = -g(Rgrid, Rgrid[X * Y - 1], Rgrid[X * Y - 1 - Y]);
	G[nnz - 1 - 1] = -g(Rgrid, Rgrid[X * Y - 1], Rgrid[X * Y - 1 - 1]);
	G[nnz - 1 - 0] = g(Rgrid, Rgrid[X * Y - 1], Rgrid[X * Y - 1 - 1]) + g(Rgrid, Rgrid[X * Y - 1], Rgrid[X * Y - 1 - Y]) + 2 / Rgrid[X * Y - 1];

}

void constructColind(int Colind[], int nnz, int const X, int const Y)
{
	Colind[0] = 0;
	Colind[1] = 1;
	Colind[2] = 1;

	for (int i = 0; i < (Y - 1); i++) {
		Colind[i * 3 + 3] = i + 2;
		Colind[i * 3 + 4] = i + 2;
		Colind[i * 3 + 5] = i + 2;
	}

	int offset = 1 + (Y - 1) * 2 + (Y + 1);

	for (int i = 0; i < (X - 2); i++)
	{
		for (int j = 0; j < 3 * (Y - 2); j++)
		{
			if (j == 0)
			{
				Colind[i * ((Y - 1) * 3 + 2) + offset] = 1 + i * (Y)+Y;
				Colind[i * ((Y - 1) * 3 + 2) + offset + 1] = 1 + i * (Y)+Y;//2 + j;
			}
			if (j > 0 && j < Y)
			{
				Colind[i * ((Y - 1) * 3 + 2) + j * 3 + offset - 1] = 1 + i * (Y)+j + Y;
				Colind[i * ((Y - 1) * 3 + 2) + j * 3 + offset] = 1 + i * (Y)+j + Y;
				Colind[i * ((Y - 1) * 3 + 2) + j * 3 + offset + 1] = 1 + i * (Y)+j + Y;
			}
		}
	}

	int offset2 = (nnz - 1) - (offset)+1;
	Colind[offset2 + 1] = 1 + (X - 2) * (Y)+Y;
	Colind[offset2 + 2] = 1 + (X - 2) * (Y)+Y;

	for (int i = 1; i < Y; i++) {
		Colind[offset2 + 2 + (i * 3) - 2] = 1 + (X - 2) * (Y)+i + Y;
		Colind[offset2 + 2 + (i * 3) - 1] = 1 + (X - 2) * (Y)+i + Y;
		Colind[offset2 + 2 + (i * 3)] = 1 + (X - 2) * (Y)+i + Y;
	}

}

void constructRowlind(int Rowlind[], int nnz, int const X, int const Y)
{
	Rowlind[0] = 0;
	Rowlind[1] = 0;
	Rowlind[2] = 1;

	for (int i = 0; i < (Y - 1); i++) {
		Rowlind[i * 3 + 3] = 0;
		Rowlind[i * 3 + 4] = i + 1;
		Rowlind[i * 3 + 5] = i + 2;
	}

	int offset = (Y - 2) * 2 + 2 + 1 + (Y + 1);

	for (int i = 0; i < (X - 2); i++)
	{
		for (int j = 0; j < 3 * (Y - 2); j++)
		{
			if (j == 0)
			{
				Rowlind[i * ((Y - 1) * 3 + 2) + offset] = 1 + i * (Y)+j;
				Rowlind[i * ((Y - 1) * 3 + 2) + offset + 1] = 1 + i * (Y)+Y;//2 + j;
			}
			if (j > 0 && j < Y)
			{
				Rowlind[i * ((Y - 1) * 3 + 2) + j * 3 + offset - 1] = 1 + i * (Y)+j;
				Rowlind[i * ((Y - 1) * 3 + 2) + j * 3 + offset] = 1 + i * (Y)+j + Y - 1;
				Rowlind[i * ((Y - 1) * 3 + 2) + j * 3 + offset + 1] = 1 + i * (Y)+j + Y;
			}

		}
	}

	int offset2 = (nnz - 1) - (2 + 3 * (Y - 1));
	Rowlind[offset2 + 1] = 1 + (X - 2) * (Y);
	Rowlind[offset2 + 2] = 1 + (X - 2) * (Y)+Y;


	for (int i = 1; i < Y; i++) {
		Rowlind[offset2 + 2 + (i * 3) - 2] = 1 + (X - 2) * (Y)+i;
		Rowlind[offset2 + 2 + (i * 3) - 1] = 1 + (X - 2) * (Y)+i + Y - 1;
		Rowlind[offset2 + 2 + (i * 3)] = 1 + (X - 2) * (Y)+i + Y;
	}

}






