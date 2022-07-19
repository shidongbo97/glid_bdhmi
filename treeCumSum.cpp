//this is a MEX file for Matlab.

#include "mex.h"
#include "tmwtypes.h"
#include <iostream>
#include <complex>


void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	if(nrhs != 5)
	{
		mexErrMsgIdAndTxt("MyToolbox:treeCumSum:nrhs","Five inputs required.");
	}
	if(nlhs!=1) {
		mexErrMsgIdAndTxt("MyToolbox:treeCumSum:nlhs","One output required.");
	}

	if(!(mxGetM(prhs[0]) == 1 &&  mxGetN(prhs[0]) == 1 && mxIsUint32(prhs[0])))
	{
		mexErrMsgIdAndTxt("MyToolbox:treeCumSum:wrongInput", "First argument must be a single uint32.");
	}

	uint32_T rootVertexIndex = *((uint32_T*)mxGetData(prhs[0]));

	if(!(mxGetM(prhs[1]) == 1 &&  mxGetN(prhs[1]) == 1 && mxIsDouble(prhs[1]) && mxIsComplex(prhs[1])))
	{
		mexErrMsgIdAndTxt("MyToolbox:treeCumSum:wrongInput", "Second argument must be a complex scalar.");
	}

	double f_onAnchorVertex_real = *mxGetPr(prhs[1]);
	double f_onAnchorVertex_imag = *mxGetPi(prhs[1]);

	size_t n = mxGetM(prhs[2]);

	if(!(n > 1 && mxGetN(prhs[2]) == 1 && mxIsDouble(prhs[2]) && mxIsComplex(prhs[2])))
	{
		mexErrMsgIdAndTxt("MyToolbox:treeCumSum:wrongInput", "Third argument must be a complex column vector.");
	}
	if(!( ((mxGetM(prhs[3]) == n && mxGetN(prhs[3]) == 1) || (mxGetM(prhs[3]) == 1 && mxGetN(prhs[3]) == n)) && mxIsUint32(prhs[3]) ))
	{
		mexErrMsgIdAndTxt("MyToolbox:treeCumSum:wrongInput", "Forth argument must be a uint32 vector with same number of elements as the third argument.");
	}
	if(!( ((mxGetM(prhs[4]) == n && mxGetN(prhs[4]) == 1) || (mxGetM(prhs[4]) == 1 && mxGetN(prhs[4]) == n)) && mxIsUint32(prhs[4]) ))
	{
		mexErrMsgIdAndTxt("MyToolbox:treeCumSum:wrongInput", "Fifth argument must be a uint32 vector with same number of elements as the third argument.");
	}

	double* f_onEdges_real = mxGetPr(prhs[2]);
	double* f_onEdges_imag = mxGetPi(prhs[2]);

	uint32_T* startIndices = (uint32_T*)mxGetData(prhs[3]);
	uint32_T* endIndices = (uint32_T*)mxGetData(prhs[4]);

	plhs[0] = mxCreateDoubleMatrix((mwSize)(n + 1), 1, mxCOMPLEX);

	double* verticesData_real = mxGetPr(plhs[0]);
	double* verticesData_imag = mxGetPi(plhs[0]);

	verticesData_real[rootVertexIndex-1] = f_onAnchorVertex_real;
	verticesData_imag[rootVertexIndex-1] = f_onAnchorVertex_imag;

	for(size_t i = 0; i < n; i++)
	{
		verticesData_real[endIndices[i]-1] = verticesData_real[startIndices[i]-1] + f_onEdges_real[i];
		verticesData_imag[endIndices[i]-1] = verticesData_imag[startIndices[i]-1] + f_onEdges_imag[i];
	}
}
