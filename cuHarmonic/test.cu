#define  _CRT_SECURE_NO_WARNINGS

#include "cuHarmonicDeform.cuh"
#include <cuComplex.h>
//#include <mat.h>
#include <cstdio>
#include <ctime>
#include <cstring> 
#include <cstdlib>
#include "matio.h"


template<class real>
inline void copyRealImag2Complex(const real *pr, const real *pi, real *const M, int n)
{
    for (int i = 0; i < n; i++) M[i * 2  ] = pr[i];
    for (int i = 0; i < n; i++) M[i * 2+1] = pi[i];
}

template<class Complex>
std::vector<Complex> complexMat2HostVecC(const matvar_t* m) 
{
    if (!m) return std::vector <Complex>();

    int n = m->nbytes/m->data_size;
    std::vector<Complex> v(n);

    using real = decltype(v[0].x);
    auto dat = (const mat_complex_split_t*)m->data;
    copyRealImag2Complex((const real*)(dat->Re), (const real*)(dat->Im), &(v[0].x), n);

    return v;
};

int getMatVarNumElements(matvar_t *mat)
{
    if (!mat) return 0;

    if (mat->rank == 1) return mat->dims[0];
    if (mat->rank == 2) return mat->dims[0]*mat->dims[1];
    if (mat->rank == 3) return mat->dims[0]*mat->dims[1]*mat->dims[2];

    fprintf(stderr, "unknown rank");
    return 0;
}

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


template<typename T>
std::vector<T> getMatDataReal(matvar_t *mat, matio_types verify_MatType)
{
    if (!mat) return std::vector<T>();

    if (mat->isComplex) {
        fprintf(stderr, "Incorrect input: %s should be real matrix\n", mat->name);
        exit(EXIT_FAILURE);
    }

    if (mat->data_type == MAT_T_CELL) {
        fprintf(stderr, "Incorrect input: %s must not be cell!\n", mat->name);
        exit(EXIT_FAILURE);
    }

    if (mat->data_type != verify_MatType) {
        fprintf(stderr, "Incorrect input: %s is of incorrect numerical type (int32/float/double)\n", mat->name);
        exit(EXIT_FAILURE);
    }

    const T* dbegin = (T*)(mat->data);
    return std::vector<T>(dbegin, dbegin+getMatVarNumElements(mat));
}


template<typename T>
cuVector<T> getMatDataGPUReal(matvar_t *mat, matio_types verify_MatType)
{
    return cuVector<T>(getMatDataReal<T>(mat, verify_MatType));
}

template<typename T>
cuVector<T> getMatDataGPUComplex(matvar_t *mat, matio_types verify_MatType) 
{
    if (!mat) return cuVector<T>();

    if ( mat->isComplex==0) {
        fprintf(stderr, "Incorrect input: %s should be Complex matrix\n", mat->name);
        exit(EXIT_FAILURE);
    }

    if (mat->data_type == MAT_T_CELL) {
        fprintf(stderr, "Incorrect input: %s must not be cell!\n", mat->name);
        exit(EXIT_FAILURE);
    }

    if (mat->data_type != verify_MatType) {
        fprintf(stderr, "Incorrect input: %s is of incorrect numerical type (int32/float/double)\n", mat->name);
        exit(EXIT_FAILURE);
    }

    return cuVector<T>(complexMat2HostVecC<T>(mat));
}
 
int main(int argc, const char* argv[]) 
{
    const char *matfile = argc>1?argv[1]:"aqp_test.mat";

    printf("Loading mat file %s...\n\n", matfile);

    // open file and read its contents with matGetVariable
    mat_t *pmat = Mat_Open(matfile, MAT_ACC_RDONLY);
    if (pmat == NULL) {
        fprintf(stderr, "error loading file %s\n", matfile);
        return EXIT_FAILURE;
    }

    matvar_t *mat_invM = Mat_VarRead(pmat, "invM");
    matvar_t *mat_D2 = Mat_VarRead(pmat, "D2");
    matvar_t *mat_C2 = Mat_VarRead(pmat, "C2");
    matvar_t *mat_bP2P = Mat_VarRead(pmat, "bP2P");
    matvar_t *mat_phipsyIters = Mat_VarRead(pmat, "phipsyIters");


    if ((mat_invM && !mat_invM->isComplex)
      || !mat_D2->isComplex || !mat_C2->isComplex
      ||!mat_bP2P->isComplex || !mat_phipsyIters->isComplex) {
        fprintf(stderr, "incorrect input, matrix should be complex");
        return EXIT_FAILURE;
    }

    const int isoetype = int(getScalarFromMat(pmat, "isometric_energy_type", ISO_ENERGY_SYMMETRIC_DIRICHLET));
    const double isoepow = getScalarFromMat(pmat, "isoepow", 1);
    const double lambda = getScalarFromMat(pmat, "lambda");
    const int nIter = argc>2?atoi(argv[2]):int( getScalarFromMat(pmat, "nIter") );
    const int enEvalsPerKernel = argc > 3 ? atoi(argv[3]) : 5;
    const double AQPKappa = argc>4?atof(argv[4]):getScalarFromMat(pmat, "AQPKappa");
    const int optimzationmethod = argc > 5 ? atoi(argv[5]) : OM_NEWTON_SPDH;


    assert(mat_D2->rank == 2 && mat_C2->rank == 2);
    int m = int(mat_D2->dims[0]);
    int n = int(mat_D2->dims[1]);
    int nP2P = int(mat_C2->dims[0]);  // = mxGetM(bP2P);

    matvar_t *mat_cageOffsets = Mat_VarRead(pmat, "cageOffsets");
    std::vector<int> cageOffsets = getMatDataReal<int>(mat_cageOffsets, MAT_T_INT32);

    matvar_t *mat_hessian_samples = Mat_VarRead(pmat, "hessian_samples");
    cuVector<int> hessian_samples = getMatDataGPUReal<int>(mat_hessian_samples, MAT_T_INT32);
    const int mh = hessian_samples.empty() ? m : hessian_samples.size();


    matvar_t *mat_sample_spacings = Mat_VarRead(pmat, "sample_spacings_half");
    matvar_t *mat_v = Mat_VarRead(pmat, "v");
    matvar_t *mat_E2 = Mat_VarRead(pmat, "E2");
    matvar_t *mat_L = Mat_VarRead(pmat, "L");
    matvar_t *mat_nextSamples = Mat_VarRead(pmat, "nextSampleInSameCage");



    auto isDouble = [](const matvar_t *v) { return v->data_type == MAT_T_DOUBLE; };
    std::vector<bool> v = { isDouble(mat_D2), isDouble(mat_C2), isDouble(mat_bP2P), isDouble(mat_phipsyIters), isDouble(mat_invM) };


    auto t0 = clock();
    std::vector<double> en;

    if (v[0]) {
        using real = double;
        using Complex = cuDoubleComplex;

        using gpuVector = cuVector<Complex>;
        const gpuVector invM = complexMat2HostVecC<Complex>(mat_invM);
        const gpuVector D2 = complexMat2HostVecC<Complex>(mat_D2);
        const gpuVector C2 = complexMat2HostVecC<Complex>(mat_C2);
        const gpuVector bP2P = complexMat2HostVecC<Complex>(mat_bP2P);
        gpuVector phipsyIters = complexMat2HostVecC<Complex>(mat_phipsyIters);

        matio_types runType = MAT_T_DOUBLE;
        const auto sample_spacings = getMatDataGPUReal<real>(mat_sample_spacings, runType);
        const auto validation_v = getMatDataGPUComplex<Complex>(mat_v, runType);
        const auto validation_E2 = getMatDataGPUComplex<Complex>(mat_E2, runType);
        const auto validation_L = getMatDataGPUReal<real>(mat_L, runType);
        const auto validation_nextSamples = getMatDataGPUReal<int>(mat_nextSamples, MAT_T_INT32);

        en = cuHarmonicDeform<Complex, real>(invM.data(), D2.data(), C2.data(), bP2P.data(), phipsyIters.data(), 
            hessian_samples.data(), 
            m, mh, n, cageOffsets, nP2P, isoetype, isoepow, lambda, AQPKappa, nIter,
            HarmonicMapValidationInput<Complex, real>(validation_v.data(), validation_E2.data(), validation_L.data(), validation_nextSamples.data()),
            sample_spacings.data(), enEvalsPerKernel, optimzationmethod);

        //en = test( gpuVector(D2).data(), m, n);

        phipsyIters = gpuVector();
    }
    else {
        using real = float;
        using Complex = cuFloatComplex;

        using gpuVector = cuVector<Complex>; 
        const gpuVector invM = complexMat2HostVecC<Complex>(mat_invM);
        const gpuVector D2 = complexMat2HostVecC<Complex>(mat_D2);
        const gpuVector C2 = complexMat2HostVecC<Complex>(mat_C2);
        const gpuVector bP2P = complexMat2HostVecC<Complex>(mat_bP2P);
        gpuVector phipsyIters = complexMat2HostVecC<Complex>(mat_phipsyIters);

        //en = cuHarmonicDeform<Complex, real>(invM.data(), D2.data(), C2.data(), bP2P.data(), phipsyIters.data(),
        //    m, mh, n, nP2P, isoepow, lambda, AQPKappa, nIter, enEvalsPerKernel, optimzationmethod);

        phipsyIters = gpuVector();
    }


    printf("Done\n");
	double dtt = (clock() - t0)/double(CLOCKS_PER_SEC);

    // clean up before exit
    Mat_VarFree(mat_invM);
    Mat_VarFree(mat_D2);
    Mat_VarFree(mat_C2);
    Mat_VarFree(mat_bP2P);
    Mat_VarFree(mat_phipsyIters);

    if (mat_cageOffsets) Mat_VarFree(mat_cageOffsets);
    if (mat_hessian_samples) Mat_VarFree(mat_hessian_samples);
    if (mat_sample_spacings) Mat_VarFree(mat_sample_spacings);
    if (mat_v) Mat_VarFree(mat_v);
    if (mat_E2) Mat_VarFree(mat_E2);
    if (mat_L) Mat_VarFree(mat_L);
    if (mat_nextSamples) Mat_VarFree(mat_nextSamples);

    if (Mat_Close(pmat) != 0) {
        fprintf(stderr, "Error closing file %s\n", matfile);
        return EXIT_FAILURE;
    }

	fprintf(stdout, "finished %d iterations in %.3fs(%.3fs/iteration), with en = %f\n", nIter, dtt, dtt/(nIter>0?nIter:1), en[0]);	

    hessian_samples.clear();  // release memory first, devicereset will nullify the pointer
	cudaDeviceReset();
    return EXIT_SUCCESS;
}


