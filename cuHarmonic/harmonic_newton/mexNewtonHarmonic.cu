#define  _CRT_SECURE_NO_WARNINGS

#include <cstdarg>
#include <iostream>
#include <gpu/mxGPUArray.h>
#include <mex.h>

#ifdef printf   // disable mexprintf from Matlab mex
#undef printf
#endif

#include "harmonicHessian.cuh"

class mystream : public std::streambuf
{
protected:
    virtual std::streamsize xsputn(const char *s, std::streamsize n) { mexPrintf("%.*s", n, s); return n; }
    virtual int overflow(int c = EOF) { if (c != EOF) { mexPrintf("%.1s", &c); } return 1; }
};

class scoped_redirect_cout
{
public:
    scoped_redirect_cout() { old_buf = std::cout.rdbuf(); std::cout.rdbuf(&mout); }
    ~scoped_redirect_cout() { std::cout.rdbuf(old_buf); }
private:
    mystream mout;
    std::streambuf *old_buf;
};


template<typename R>
R getFieldValueWithDefault(const mxArray* mat, const char* name, R defaultvalue)
{
    const mxArray *f = mat ? mxGetField(mat, 0, name) : nullptr;
    return f ? R(mxGetScalar(f)) : defaultvalue;
}

#define mexError(s) mexErrMsgTxt("invalid input to mex: " ##s)


void mexFunction(int nlhs, mxArray *plhs[], int nrhs, mxArray const *prhs[])
{
    scoped_redirect_cout mycout_redirect;

    // interface: Hessian = newtonHarmonic( D2, phipsy, C2, bP2P, lambda, nIter);

    mxInitGPU();   // Initialize the MathWorks GPU API.

    if (nrhs < 5)  mexError("not enough input");
    if (nrhs > 5 && !mxIsStruct(prhs[5])) mexError("params must be a struct");

    /* Throw an error if the input is not a GPU array. */
    if (!mxIsGPUArray(prhs[0]) || !mxIsGPUArray(prhs[1]) || !mxIsGPUArray(prhs[2]) || !mxIsGPUArray(prhs[3]))
        mexError("input must be GPU arrays");

    mxGPUArray const *D2     = mxGPUCreateFromMxArray(prhs[0]);
    mxGPUArray const *phipsy = mxGPUCreateFromMxArray(prhs[1]);
    mxGPUArray const *C2     = mxGPUCreateFromMxArray(prhs[2]);
    mxGPUArray const *bP2P   = mxGPUCreateFromMxArray(prhs[3]);

    const double lambda = mxGetScalar(prhs[4]);

    // Verify that the matrices really are a double array before extracting the pointer.
    if (mxGPUGetComplexity(D2)   != mxCOMPLEX || mxGPUGetComplexity(phipsy) != mxCOMPLEX
      ||mxGPUGetComplexity(C2)   != mxCOMPLEX || mxGPUGetComplexity(bP2P)   != mxCOMPLEX)
        mexError("input array must be in complex numbers");

    const mwSize *dims = mxGPUGetDimensions(D2); // mxGPUGetNumberOfDimensions(D2) == 2;
    const int m = int(dims[0]);   // # samples
    const int n = int(dims[1]);   // dim Cauchy coordinates
    const int nP2P = int(mxGPUGetDimensions(C2)[0]); 

    const mxArray *params = nrhs>5?prhs[5]:nullptr;
    const double isoepow = getFieldValueWithDefault<double>(params, "isometric_energy_power", 1);
    const int nIter = getFieldValueWithDefault<int>(params, "nIter", 1);
    const int mh = getFieldValueWithDefault<int>(params, "n_hessian_sample", m);

    /* Create a GPUArray to hold the result and get its underlying pointer. */
    const mwSize hessianDim[2] = { n*4, n*4 };
    mxGPUArray *const Hessian = nlhs>1?mxGPUCreateGPUArray(2, hessianDim, mxGPUGetClassID(D2), mxREAL, MX_GPU_DO_NOT_INITIALIZE):nullptr;
    if(nlhs>1) plhs[1] = mxGPUCreateMxArrayOnGPU(Hessian);

    void *pHessian = Hessian ? mxGPUGetData(Hessian) : nullptr;
    mxGPUArray *const dphipsy = mxGPUCopyFromMxArray(prhs[1]);

    // Now that we have verified the data type, extract a pointer to the input data on the device.
    if (mxGPUGetClassID(D2) == mxDOUBLE_CLASS){
        if(mxGPUGetClassID(phipsy) != mxDOUBLE_CLASS || mxGPUGetClassID(C2) != mxDOUBLE_CLASS || mxGPUGetClassID(bP2P) != mxDOUBLE_CLASS )
            mexError("input arrays should be all of the same precision type (double)");

        using Complex = cuDoubleComplex;
        using real = double;
        //////////////////////////////////////////////////////////////////////////
        hessianIsometric<Complex, real>( (Complex const *)(mxGPUGetDataReadOnly(D2)),
                                         (Complex const *)(mxGPUGetDataReadOnly(phipsy)),
                                         (Complex const *)(mxGPUGetDataReadOnly(C2)),
                                         (Complex const *)(mxGPUGetDataReadOnly(bP2P)),
                                         (Complex *)      (mxGPUGetData(dphipsy)),
            m, mh, n, nP2P, isoepow, lambda, (real*)pHessian, nIter);

    }
    else if (mxGPUGetClassID(D2) == mxSINGLE_CLASS) {
        if (mxGPUGetClassID(phipsy) != mxSINGLE_CLASS || mxGPUGetClassID(C2) != mxSINGLE_CLASS || mxGPUGetClassID(bP2P) != mxSINGLE_CLASS)
            mexError("input arrays should be all of the same precision type (single)");

        using Complex = cuFloatComplex;
        using real = float;
        //////////////////////////////////////////////////////////////////////////
        hessianIsometric<Complex, real>( (Complex const *)(mxGPUGetDataReadOnly(D2)),
                                         (Complex const *)(mxGPUGetDataReadOnly(phipsy)),
                                         (Complex const *)(mxGPUGetDataReadOnly(C2)),
                                         (Complex const *)(mxGPUGetDataReadOnly(bP2P)),
                                         (Complex *)      (mxGPUGetData(dphipsy)),
            m, mh, n, nP2P, isoepow, lambda, (real*)pHessian, nIter);
    }

    // Wrap the result up as a MATLAB gpuArray for return.
    plhs[0] = mxGPUCreateMxArrayOnGPU(dphipsy);

    // * The mxGPUArray pointers are host-side structures that refer to device
    // * data. These must be destroyed before leaving the MEX function.
    mxGPUDestroyGPUArray(D2);
    mxGPUDestroyGPUArray(phipsy);
    mxGPUDestroyGPUArray(C2);
    mxGPUDestroyGPUArray(bP2P);

    mxGPUDestroyGPUArray(dphipsy);

    if(Hessian)  mxGPUDestroyGPUArray(Hessian);
}