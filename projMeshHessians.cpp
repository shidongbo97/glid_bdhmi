#define MATLAB_DEFAULT_RELEASE R2017b

#include <mex.h>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <iostream>
#include <chrono>

using namespace Eigen;


template<class mat, class cvec>
void mulMatrixMKM(const Matrix4d &K, cvec &R, cvec &I, mat& A2)
{
    const double k[] = { K(0,0), K(0,1), K(0,2), K(0,3), K(1,1), K(1,2), K(1,3), K(2,2), K(2,3), K(3,3) };
    Matrix2d B;
    B << k[0] + 2 * k[2] + k[7], -k[1] + k[3] - k[5] + k[8], 0, k[4] - 2 * k[6] + k[9];

    Matrix3d RtR, ItI, RtI;
    RtR << R(0)*R, R(1)*R, R(2)*R;
    ItI << I(0)*I, I(1)*I, I(2)*I;
    RtI << R(0)*I, R(1)*I, R(2)*I;
    A2.block<3, 3>(0, 0) = B(0,0)*RtR + B(1,1)*ItI + B(0,1)*(RtI + RtI.transpose());

    B << k[4] + 2 * k[6] + k[9], k[1] + k[3] - k[5] - k[8], 0, k[0] - 2 * k[2] + k[7];
    A2.block<3, 3>(3, 3) = B(0,0)*RtR + B(1,1)*ItI + B(0,1)*(RtI + RtI.transpose());

    B << k[1] + k[3] + k[5] + k[8], k[0] - k[7], k[9] - k[4], -k[1] + k[3] + k[5] - k[8];
    A2.block<3, 3>(3, 0) = B(0,0)*RtR + B(1,1)*ItI + B(0,1)*RtI + B(1,0)*RtI.transpose();

    A2.block<3, 3>(0, 3) = A2.block<3, 3>(3, 0).transpose();
}



template<class mat, class cmat, class cvec>
void projMeshHessians(mat&& h, cmat &alpha, cmat &beta, cvec &fzr, cvec &fzi, cvec &gzr, cvec &gzi, cmat &Dr, cmat &Di, int projMethod) 
{
    const int n = (int)alpha.cols();

    #pragma omp parallel for
	for (int i = 0; i < n; i++){
        const auto& currDr = Dr.col(i);
        const auto& currDi = Di.col(i);

       // K
        Matrix4d K;
        //K.block<2,2>(0, 0)<< 2 * alpha(0,i) + 4 * beta(0,i)*fzr(i)*fzr(i),
        //4 * beta(0,i)*fzi(i)*fzr(i),
        //4 * beta(0,i)*fzi(i)*fzr(i),
        //2 * alpha(0,i) + 4 * beta(0,i)*fzi(i)*fzi(i);

        K(0, 0) = 2 * alpha(0, i) + 4 * beta(0, i)*fzr(i)*fzr(i);
        K(0, 1) = 4 * beta(0, i)*fzi(i)*fzr(i);
        K(1, 0) = K(0, 1);
        K(1, 1) = 2 * alpha(0, i) + 4 * beta(0, i)*fzi(i)*fzi(i);

        K(0, 2) = 4 * beta(2,i)*fzr(i)*gzr(i);
        K(1, 2) = 4 * beta(2,i)*fzi(i)*gzr(i);
        K(0, 3) = 4 * beta(2,i)*fzr(i)*gzi(i);
        K(1, 3) = 4 * beta(2,i)*fzi(i)*gzi(i);

        //K.block<2, 2>(2, 0) = K.block<2, 2>(0, 2).transpose();
        K(2, 0) = K(0, 2);
        K(3, 0) = K(0, 3);
        K(2, 1) = K(1, 2);
        K(3, 1) = K(1, 3);

        K(2, 2) = 2 * alpha(1,i) + 4 * beta(1,i)*gzr(i)*gzr(i);
        K(3, 2) = 4 * beta(1,i)*gzi(i)*gzr(i);
        K(2, 3) = K(3, 2);
        K(3, 3) = 2 * alpha(1,i) + 4 * beta(1,i)*gzi(i)*gzi(i);

        // H
        if (projMethod == 1) {
            double v1v3 = currDr.squaredNorm() - currDi.squaredNorm();
            double v2v3 = 2 * currDr.dot(currDi);
            double v1v1 = currDr.squaredNorm() + currDi.squaredNorm();
            double a = v1v3 / v1v1, b = v2v3 / v1v1;
            double c = sqrt(1 - a * a - b * b);

            Matrix4d MOD;
            MOD.setZero(4, 4);
            MOD(0, 0) = 1;
            MOD(1, 1) = 1;
            MOD(2, 2) = c;
            MOD(3, 3) = c;
            MOD(0, 2) = a;
            MOD(0, 3) = b;
            MOD(1, 2) = -b;
            MOD(1, 3) = a; 

            const Matrix4d KM = MOD * K * MOD.transpose();
            SelfAdjointEigenSolver<Matrix4d> es(KM);
            Matrix4d V = es.eigenvectors();
            Vector4d d = es.eigenvalues().cwiseMax(0);

            MOD(2, 2) = 1 / c;
            MOD(3, 3) = 1 / c;
            MOD.block<2, 2>(0, 2) *= -1 / c;

            V = MOD * V;
            K.noalias() = V * d.asDiagonal()*V.transpose();
        }

        Map<Matrix<double, 6, 6> > H(h.data() + i * 36);
        mulMatrixMKM(K, currDr, currDi, H);
        
         // equivalent code for mulMatrixMKM 
        //Matrix<double, 4, 6> DBD;
        //DBD<<currDr.transpose(),currDi.transpose(),-currDi.transpose(),currDr.transpose(),currDr.transpose(),-currDi.transpose(),currDi.transpose(),currDr.transpose();
        //H = DBD.transpose()*K*DBD;

        if (projMethod == 2) {
            SelfAdjointEigenSolver<Matrix<double, 6, 6>> es(H);
            Matrix<double, 6, 6> V = es.eigenvectors();
            Matrix<double, 6, 1> d = es.eigenvalues().cwiseMax(0);
            H.noalias() = V * d.asDiagonal()*V.transpose();
        }
	}
}


inline void mexError(const std::string& error)
{
    mexErrMsgTxt(("invalid input to mex: " + error).c_str());
}

template<class time_point>
float elapsedseconds(time_point t)
{
    using namespace std::chrono;
    //duration_cast<microseconds>(steady_clock::now() - t).count() / 1000.f
    return duration_cast<duration<float>>(steady_clock::now() - t).count();
}

void mexFunction(int nlhs, mxArray *plhs[],	int nrhs, const mxArray*prhs[])
{
    // h = computeMeshHessian(alpha, beta, fz, gz, D, option, verbose)
    // alpha: 2 x n
    // beta:  3 x n
    // fz, gz: n
    // D:     3 x n

    using std::chrono::steady_clock;
    auto t = steady_clock::now();
    if (nrhs < 5)  mexError("not enough input");

	const size_t n = mxGetN(prhs[0]); // number of faces
    if (mxGetM(prhs[0]) != 2 || mxGetN(prhs[1]) != n || mxGetM(prhs[1]) != 3
        || mxGetNumberOfElements(prhs[2]) != n || mxGetNumberOfElements(prhs[3]) != n
        || mxGetM(prhs[4]) != 3 || mxGetN(prhs[4]) != n)  mexError("bad input, inconsistent dimensions");

	Map<const MatrixXd> alpha(mxGetPr(prhs[0]), 2, n);
	Map<const MatrixXd> beta(mxGetPr(prhs[1]), 3, n);
    Map<const VectorXd> fzr(mxGetPr(prhs[2]), n);
    Map<const VectorXd> fzi(mxGetPi(prhs[2]), n);
    Map<const VectorXd> gzr(mxGetPr(prhs[3]), n);
    Map<const VectorXd> gzi(mxGetPi(prhs[3]), n);
    Map<const MatrixXd> Dr(mxGetPr(prhs[4]), 3, n);
    Map<const MatrixXd> Di(mxGetPi(prhs[4]), 3, n);
	

	// init output
    int projMethod = nrhs > 5 ? int(mxGetScalar(prhs[5])) : 0;
    plhs[0] = mxCreateDoubleMatrix(36, n, mxREAL);
    bool verbose = nrhs > 6 ? mxIsLogicalScalarTrue(prhs[6]):false;

    int nthreads = 0;
    if (verbose) {
        printf("%-30s %fs\n", "preprocess", elapsedseconds(t));

#pragma omp parallel
        if (omp_get_thread_num() == 0)
            nthreads = omp_get_num_threads();
    }

    t = steady_clock::now();
    projMeshHessians(Map<MatrixXd>(mxGetPr(plhs[0]), 36, n), alpha, beta, fzr, fzi, gzr, gzi, Dr, Di, projMethod);

    if (verbose) {
        char str[100];
        sprintf(str, "project (method %d, %d threads)", projMethod, nthreads);
        printf("%-30s %fs\n", str, elapsedseconds(t));
    }
}
