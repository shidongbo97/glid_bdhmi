#define  _CRT_SECURE_NO_WARNINGS

#include <cuComplex.h>
#include <cstdio>
#include <ctime>
#include <cstring> 
#include <cstdlib>
#include <vector>
#include "matio.h"

#include "harmonicHessian.cuh"


template<class real>
inline void copyRealImag2Complex(const real *pr, const real *pi, real *const M, int n)
{
    for (int i = 0; i < n; i++) M[i * 2  ] = pr[i];
    for (int i = 0; i < n; i++) M[i * 2+1] = pi[i];
}

template<class Complex>
std::vector<Complex> complexMat2HostVecC(const matvar_t* m) 
{
    int n = m->nbytes/m->data_size;
    //thrust::host_vector<Complex> v(n);
    std::vector<Complex> v(n);

    using real = decltype(v[0].x);
    auto dat = (const mat_complex_split_t*)m->data;
    copyRealImag2Complex((const real*)(dat->Re), (const real*)(dat->Im), &(v[0].x), n);

    return v;
};

double getScalarFromMat(mat_t *pmat, const char* varname, double fallbackvalue = 0)
{
    matvar_t *mat = Mat_VarRead(pmat, varname);
    double v = fallbackvalue;
    if (mat) {
        v = (mat->data_type == MAT_T_DOUBLE) ? *(double*)(mat->data) : *(float*)(mat->data);
        Mat_VarFree(mat);
    }
    return v;
}


int main(int argc, const char* argv[]) 
{
    const char *matfile = argc>1?argv[1]:"aqp_test_double7.mat";

    printf("Loading mat file %s...\n\n", matfile);

    // open file and read its contents with matGetVariable
    mat_t *pmat = Mat_Open(matfile, MAT_ACC_RDONLY);
    if (pmat == NULL) {
        fprintf(stderr, "error loading file %s\n", matfile);
        return EXIT_FAILURE;
    }

    matvar_t *mat_D2 = Mat_VarRead(pmat, "D2");
    matvar_t *mat_phipsy = Mat_VarRead(pmat, "phipsyIters");
    matvar_t *mat_C2 = Mat_VarRead(pmat, "C2");
    matvar_t *mat_bP2P = Mat_VarRead(pmat, "bP2P");


    const double lambda = getScalarFromMat(pmat, "lambda");
    const double isoepower = getScalarFromMat(pmat, "isoepower", 1);

    if (!mat_D2->isComplex || !mat_phipsy->isComplex) {
        fprintf(stderr, "incorrect input, matrix D2 and phipsy should be complex, and DRIRIT and DRIIR should be real");
        return EXIT_FAILURE;
    }

    const int nIter = argc>2?atoi(argv[2]):1;

    assert(mat_D2->rank == 2);
    const int m = int(mat_D2->dims[0]);
    const int n = int(mat_D2->dims[1]);

    const int mh = getScalarFromMat(pmat, "n_hessian_sample", m);

    const int nP2P = int(mat_bP2P->dims[0]);

    auto isDouble = [](const matvar_t *v) { return v->data_type == MAT_T_DOUBLE; };
    const bool v[] = { isDouble(mat_D2), isDouble(mat_phipsy) };

    auto t0 = clock();

    if (v[0]) {
        using real = double;
        using Complex = cuDoubleComplex;
        using vecR = cuVector<real>;
        using vecC = cuVector<Complex>;

        vecC D2 = complexMat2HostVecC<Complex>(mat_D2);
        vecC phipsy = complexMat2HostVecC<Complex>(mat_phipsy);
        vecC C2 = complexMat2HostVecC<Complex>(mat_C2);
        vecC bP2P = complexMat2HostVecC<Complex>(mat_bP2P);

        vecC dphipsy(n * 2);
        vecR hessian(n * 4 * n * 4);

        hessianIsometric<Complex, real>(D2.data(), phipsy.data(), C2.data(), bP2P.data(),
            dphipsy.data(), m, mh, n, nP2P, isoepower, lambda, hessian.data(), nIter);
    }
    else {
        using real = float;
        using Complex = cuFloatComplex;
        using vecR = cuVector<real>;
        using vecC = cuVector<Complex>;

        vecC D2 = complexMat2HostVecC<Complex>(mat_D2);
        vecC phipsy = complexMat2HostVecC<Complex>(mat_phipsy);
        vecC C2 = complexMat2HostVecC<Complex>(mat_C2);
        vecC bP2P = complexMat2HostVecC<Complex>(mat_bP2P);

        vecC dphipsy(n * 2);
        vecR hessian(n * 4 * n * 4);

        hessianIsometric<Complex, real>(D2.data(), phipsy.data(), C2.data(), bP2P.data(),
            dphipsy.data(), m, mh, n, nP2P, isoepower, lambda, hessian.data(), nIter);
    }


    printf("Done\n");
	double dtt = (clock() - t0)/double(CLOCKS_PER_SEC);

    // clean up before exit
    Mat_VarFree(mat_D2);
    Mat_VarFree(mat_phipsy);
    Mat_VarFree(mat_C2);
    Mat_VarFree(mat_bP2P);

    if (Mat_Close(pmat) != 0) {
        fprintf(stderr, "Error closing file %s\n", matfile);
        return EXIT_FAILURE;
    }

	fprintf(stdout, "finished %d iterations in %.3fs(%.3fs/iteration)\n", nIter, dtt, dtt/(nIter>0?nIter:1));	

	cudaDeviceReset();
    return EXIT_SUCCESS;
}
