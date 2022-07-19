#define MATLAB_DEFAULT_RELEASE R2017b

#include <mex.h>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <iostream>
#include <chrono>

using namespace Eigen;

enum ISOMETRIC_ENERGY_TYPE { ISO_ENERGY_SYMMETRIC_DIRICHLET = 0, ISO_ENERGY_EXP_SD, ISO_ENERGY_AMIPS, ISO_ENERGY_SARAP, ISO_ENERGY_HOOK, 
                             ISO_ENERGY_ARAP, ISO_ENERGY_BARAP, ISO_ENERGY_BCONF};
enum HESSIAN_PROJECT_TYPE { HESS_PROJ_NP = 0, HESS_PROJ_KP = 1, HESS_PROJ_FP4 = 2, HESS_PROJ_FP6 = 3, HESS_PROJ_CM = 4};

template<class real> double sqr(real x) { return x*x; };
template<class real> double pow3(real x) { return x*x*x; };
template<class real> double pow4(real x) { return sqr( sqr(x) ); };

double isometric_energy_from_fz2gz2(double fz2, double gz2, int energy_type, double s=1.)
{
    double e;
    switch (energy_type){
    case ISO_ENERGY_ARAP:
        return 2 * (fz2 + gz2 - 2 * sqrt(fz2) + 1);
    case ISO_ENERGY_BARAP:
        return 2 * (fz2 + gz2 - 2 * sqrt(fz2) + 1) + s*(fz2 - gz2) + s/(fz2-gz2);
    case ISO_ENERGY_BCONF:
        return s*(fz2 - gz2) + s/(fz2 - gz2) + gz2;
    case ISO_ENERGY_SYMMETRIC_DIRICHLET:
        e = (fz2 + gz2)*(1 + 1 / sqr(gz2 - fz2));   // TODO, shift energy to 0, to see if it make any difference numerically
        return s == 1. ? e : pow(e, s);
    case ISO_ENERGY_EXP_SD:
        //e = fabs((fz2 + gz2)*(1 + 1 / sqr(gz2 - fz2))-2);   // TODO, shift energy to 0
        e = (fz2 + gz2)*(1 + 1 / sqr(gz2 - fz2));
        return exp(s*e);
    case ISO_ENERGY_AMIPS:
        //return exp(abs((2 * s*(fz2 + gz2) + 1) / (fz2 - gz2) + (fz2 - gz2)) - 2 - 2 * s);  // shift energy to 0
        return exp((2 * s*(fz2 + gz2) + 1) / (fz2 - gz2) + (fz2 - gz2)); 
    case ISO_ENERGY_SARAP:
        fz2 = sqrt(fz2); gz2 = sqrt(gz2);
        return sqr(fz2+gz2-1) + sqr(1/(fz2-gz2) -1) ;
    case ISO_ENERGY_HOOK:
        return s*2*gz2/(fz2-gz2) + (1-s)*sqr(fz2 - gz2 -1)/2;
    default:
        assert(false); // not implemented
    }

    return 0;  
}

void alpha_beta_from_fz2gz2(double fz2, double gz2, double* alphas, double* betas, int energy_type, double s=1.)
{
    switch (energy_type){
    case ISO_ENERGY_ARAP: {
        alphas[0] = 2 - 2 / sqrt(fz2);
        alphas[1] = 2;

        if (betas) {
            betas[0] = 1 / pow3(sqrt(fz2));
            betas[1] = 0;
            betas[2] = 0;
        }
        break;
    }
    case ISO_ENERGY_BARAP:
        alphas[1] = 2 - s + s / sqr(fz2 - gz2);
        //alphas[0] = 2 - 2 / sqrt(fz2) + s  - s / sqr(fz2 - gz2);
        alphas[0] = 4 - alphas[1] - 2 / sqrt(fz2);

        if (betas) {
            betas[1] = 2 * s / pow3(fz2 - gz2);
            betas[2] = -betas[1];
            betas[0] = 1 / pow3(sqrt(fz2)) + betas[1];
        }
        break;
    case ISO_ENERGY_BCONF:
        alphas[0] = s - s / sqr(fz2 - gz2);
        alphas[1] = 1 - alphas[0];

        if (betas) {
            betas[0] = 2 * s / pow3(fz2 - gz2);
            betas[1] = betas[0];
            betas[2] = -betas[0];
        }
        break;
    case ISO_ENERGY_SYMMETRIC_DIRICHLET: {
        double e = isometric_energy_from_fz2gz2(fz2, gz2, ISO_ENERGY_SYMMETRIC_DIRICHLET, s);
        double w = (s == 1) ? 1 : pow(e, 1 - 1 / s)*s;
        alphas[0] = w*(1 - (fz2 + 3 * gz2) / pow3(fz2 - gz2));
        alphas[1] = w*(1 + (3 * fz2 + gz2) / pow3(fz2 - gz2));

        if (betas) {
            double c = 2 / pow4(fz2 - gz2);
            betas[0] = c * (fz2 + 5 * gz2);
            betas[1] = c * (5 * fz2 + gz2);
            betas[2] = -3 * c * (fz2 + gz2);
            if (s != 1) {
                double d = (s - 1) / s / e;
                betas[0] = betas[0] * w + sqr(alphas[0]) *d;
                betas[1] = betas[1] * w + sqr(alphas[1]) *d;
                betas[2] = betas[2] * w + alphas[0] * alphas[1] * d;
            }
        }
        break;
    }
    case ISO_ENERGY_EXP_SD: {
        double e = isometric_energy_from_fz2gz2(fz2, gz2, ISO_ENERGY_EXP_SD, s);
        alphas[0] = s*e*(1 - (fz2 + 3 * gz2) / pow3(fz2 - gz2));
        alphas[1] = s*e*(1 + (3 * fz2 + gz2) / pow3(fz2 - gz2));

        if (betas) {
            double c = 2 / pow4(fz2 - gz2);
            betas[0] = c * (fz2 + 5 * gz2) * s* e + sqr(alphas[0]) / e;
            betas[1] = c * (5 * fz2 + gz2) * s* e + sqr(alphas[1]) / e;
            betas[2] = -3 * c * (fz2 + gz2) * s* e + alphas[0] * alphas[1] / e;
        }
        break;
    }

    case ISO_ENERGY_AMIPS: {
        double e = isometric_energy_from_fz2gz2(fz2, gz2, ISO_ENERGY_AMIPS, s);
        alphas[0] = e*(1 - (4 * s*gz2 + 1) / sqr(fz2 - gz2));
        alphas[1] = e*(-1 + (4 * s*fz2 + 1) / sqr(fz2 - gz2));

        if (betas) {
            double c = 2 / pow3(fz2 - gz2);
            betas[0] = c*(4 * s*gz2 + 1);
            betas[1] = c*(4 * s*fz2 + 1);
            betas[2] = -c*(2 * s*(fz2 + gz2) + 1);

            betas[0] = betas[0] * e + sqr(alphas[0]) / e;
            betas[1] = betas[1] * e + sqr(alphas[1]) / e;
            betas[2] = betas[2] * e + alphas[0] * alphas[1] / e;
        }
        break;
    }
    case ISO_ENERGY_SARAP: {
        fz2 = sqrt(fz2); gz2 = sqrt(gz2);  // |fz|, |gz|
        double sig1 = fz2 + gz2, sig2 = fz2 - gz2;
        alphas[0] = (sig1 - 1 - (1 - sig2) / pow3(sig2)) / fz2;
        alphas[1] = (sig1 - 1 + (1 - sig2) / pow3(sig2));
        if (gz2 > 1e-40)  alphas[1] /= gz2;

        if (betas) {
            double t = (3 - 2 * sig2) / pow4(sig2) + 1;
            betas[0] = ( (1 - sig2) / pow3(sig2) - sig1 + 1 + fz2 * t ) / pow3(fz2) / 2;
            betas[1] = (-(1 - sig2) / pow3(sig2) - sig1 + 1 + gz2 * t) / pow3(gz2) / 2;
            betas[2] = (2 - t) / fz2 / gz2 / 2;
            if (gz2 < 1e-80) { betas[1] = 10; betas[2] = -6; }
        }
        break;
    }
    case ISO_ENERGY_HOOK: {
        double mu = s;
        double kappa = 1-s;
        double xmy = fz2 - gz2;
        double xpy = fz2+gz2;
        alphas[0] = (mu*(-2*gz2/sqr(xmy))+kappa*(xmy-1));
        alphas[1] = (mu*(2*fz2/sqr(xmy))-kappa*(xmy-1));

        if (betas) {
            betas[0] = (mu*(4*gz2/pow3(xmy))+kappa);
            betas[1] = (mu*(4*fz2/pow3(xmy) )+kappa);
            betas[2] = (-mu*2*xpy/pow3(xmy) -kappa);
        }
        break;
    }
    default:
        assert(false); // energy type not implemented
    }
}

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



template<class cvec>
void projMeshHessians(double *h, const double *alpha, const double *beta, double fzr, double fzi, double gzr, double gzi, cvec &Dr, cvec &Di, int projMethod) 
{
    // K
    Matrix4d K;
    //K.block<2,2>(0, 0)<< 2 * alpha[0] + 4 * beta[0]*fzr*fzr,
    //4 * beta(0,i)*fzi*fzr,
    //4 * beta(0,i)*fzi*fzr,
    //2 * alpha(0,i) + 4 * beta(0,i)*fzi*fzi;

    K(0, 0) = 2 * alpha[0] + 4 * beta[0]*fzr*fzr;
    K(0, 1) = 4 * beta[0]*fzi*fzr;
    K(1, 0) = K(0, 1);
    K(1, 1) = 2 * alpha[0] + 4 * beta[0]*fzi*fzi;

    K(0, 2) = 4 * beta[2]*fzr*gzr;
    K(1, 2) = 4 * beta[2]*fzi*gzr;
    K(0, 3) = 4 * beta[2]*fzr*gzi;
    K(1, 3) = 4 * beta[2]*fzi*gzi;

    //K.block<2, 2>(2, 0) = K.block<2, 2>(0, 2).transpose();
    K(2, 0) = K(0, 2);
    K(3, 0) = K(0, 3);
    K(2, 1) = K(1, 2);
    K(3, 1) = K(1, 3);

    K(2, 2) = 2 * alpha[1] + 4 * beta[1]*gzr*gzr;
    K(3, 2) = 4 * beta[1]*gzi*gzr;
    K(2, 3) = K(3, 2);
    K(3, 3) = 2 * alpha[1] + 4 * beta[1]*gzi*gzi;

    // H
    if (projMethod == HESS_PROJ_FP4) {
        double v1v3 = Dr.squaredNorm() - Di.squaredNorm();
        double v2v3 = 2 * Dr.dot(Di);
        double v1v1 = Dr.squaredNorm() + Di.squaredNorm();
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

    Map<Matrix<double, 6, 6> > H(h);
    mulMatrixMKM(K, Dr, Di, H);

    // equivalent code for mulMatrixMKM 
    //Matrix<double, 4, 6> DBD;
    //DBD<<currDr.transpose(),currDi.transpose(),-currDi.transpose(),currDr.transpose(),currDr.transpose(),-currDi.transpose(),currDi.transpose(),currDr.transpose();
    //H = DBD.transpose()*K*DBD;

    if (projMethod == HESS_PROJ_FP6) {
        SelfAdjointEigenSolver<Matrix<double, 6, 6>> es(H);
        Matrix<double, 6, 6> V = es.eigenvectors();
        Matrix<double, 6, 1> d = es.eigenvalues().cwiseMax(0);
        H.noalias() = V * d.asDiagonal()*V.transpose();
    }
}


template<class cmat>
double evalIsometricEnergy(const double *fzr, const double *fzi, const double *gzr, const double *gzi, cmat &Dr, cmat &Di, const double *area,
    int energy_type, double s, double *grad, double *hessian, int projMethod)
{
    const int n = (int)Dr.cols();
    double en = 0;

#pragma omp parallel for shared(fzr, fzi, gzr, gzi, Dr, Di, area, grad, hessian) reduction(+: en)
    for (int i = 0; i < n; i++) {
        double fz2 = sqr(fzr[i]) + sqr(fzi[i]);
        double gz2 = sqr(gzr[i]) + sqr(gzi[i]);

        en += isometric_energy_from_fz2gz2(fz2, gz2, energy_type, s)*area[i];

        if (grad) {
            Vector2d alphas;
            Vector3d  betas;
            alpha_beta_from_fz2gz2(fz2, gz2, alphas.data(), betas.data(), energy_type, s);
            alphas *= area[i] * 2;
            betas *= area[i] * 2;

            Map<Matrix<double, 6, 1>> g(grad + i * 6);
            g << Dr.col(i)*(fzr[i] * alphas[0] + gzr[i] * alphas[1]) + Di.col(i)*(-fzi[i] * alphas[0] + gzi[i] * alphas[1]),
                Dr.col(i)*(-fzi[i] * -alphas[0] + gzi[i] * alphas[1]) + Di.col(i)*(fzr[i] * alphas[0] - gzr[i] * alphas[1]);
            g *= 2;

            if (hessian) {
                if (projMethod == HESS_PROJ_KP) {
                    if (energy_type == ISO_ENERGY_ARAP || energy_type == ISO_ENERGY_EXP_SD
                        || (energy_type == ISO_ENERGY_SYMMETRIC_DIRICHLET && s >= 1)) {
                        if (alphas[0] < 0) {
                            betas[0] += alphas[0] / 2 / fz2;
                            alphas[0] = 0;
                        }
                    }
                    else if ((gz2 > 1e-50) && (abs(betas[0]) > 1e-50) && (abs(betas[1]) > 1e-50) && (abs(betas[2]) > 1e-50)) {
                        //general SPD modification, take care of identity map, where |gz|=0 causes problem for the following modification
                        //s1s2 = (alphas + 2 * betas(:, 1 : 2).*[fz2 gz2])*[1 1; 1 - 1];
                        //lambda34 = [s1s2(:, 1) sqrt(s1s2(:, 2). ^ 2 + 16 * betas(:, 3).^2.*fz2.*gz2)] * [1 1; 1 - 1];
                        //t1t2 = (lambda34 - 2 * alphas(:, 1) - 4 * betas(:, 1).*fz2). / (4 * betas(:, 3).*gz2);
                        //eigenvec34nrm2 = fz2 + gz2.*t1t2. ^ 2;
                        //lambda34 = max(lambda34, 0). / eigenvec34nrm2;

                        double t1 = alphas[0] + 2 * betas[0] * fz2;
                        double t2 = alphas[1] + 2 * betas[1] * gz2;
                        const double s1 = t1 + t2, s2 = t1 - t2;
                        t1 = sqrt(sqr(s2) + 16 * sqr(betas[2])*fz2*gz2);
                        double lambda3 = s1 + t1, lambda4 = s1 - t1;

                        t1 = (lambda3 - 2 * alphas[0] - 4 * betas[0] * fz2) / (4 * betas[2] * gz2);
                        t2 = (lambda4 - 2 * alphas[0] - 4 * betas[0] * fz2) / (4 * betas[2] * gz2);

                        lambda3 = std::max(lambda3, 0.) / (fz2 + gz2 * sqr(t1));
                        lambda4 = std::max(lambda4, 0.) / (fz2 + gz2 * sqr(t2));

                        alphas[0] = std::max(alphas[0], 0.);
                        alphas[1] = std::max(alphas[1], 0.);

                        betas[0] = lambda3 / 4 + lambda4 / 4 - alphas[0] / 2 / fz2;
                        betas[1] = lambda3 * sqr(t1) / 4 + lambda4 * sqr(t2) / 4 - alphas[1] / 2 / gz2;
                        betas[2] = lambda3 * t1 / 4 + lambda4 * t2 / 4;
                    }
                }
                else if (projMethod == HESS_PROJ_CM) {
                    if (energy_type == ISO_ENERGY_ARAP || energy_type == ISO_ENERGY_EXP_SD
                        || (energy_type == ISO_ENERGY_SYMMETRIC_DIRICHLET && s >= 1)) {
                        if (alphas[0] < 0) {
                            betas[0] += alphas[0] / 2 / fz2;
                            alphas[0] = 0;
                        }
                    }
                    else if (energy_type == ISO_ENERGY_SARAP) {
                        double fzn = std::sqrt(fz2);
                        double gzn = std::sqrt(gz2);
                        double gv = fzn - gzn;
                        double hvvm = (gv > 1.5) ? 6 / pow4(gv) - 4 / pow3(gv) : 0;
                        if (alphas[0] < 0) {
                            betas[0] += alphas[0] / 2 / fz2;
                            alphas[0] = 0;
                        }
                        if (alphas[1] < 0) {
                            betas[1] += alphas[1] / 2 / gz2;
                            alphas[1] = 0;
                        }
                        if (gzn > 1e-40) {
                            betas[0] -= hvvm / 4 / fz2 * area[i] * 2;
                            betas[1] -= hvvm / 4 / gz2 * area[i] * 2;
                            betas[2] += hvvm / 4 / fzn / gzn * area[i] * 2;
                        }
                    }
                    else if (energy_type == ISO_ENERGY_HOOK) {
                        double mu = s, kappa = 1 - s;
                        double xpy = fz2 + gz2, xmy = fz2 - gz2;
                        double hv = (-mu * xpy / sqr(xmy) + kappa * (xmy - 1));

                        alphas[0] = hv > 0 ? (mu / xmy + hv)*area[i] * 2 : mu / xmy * area[i] * 2;
                        alphas[1] = hv < 0 ? (mu / xmy - hv)*area[i] * 2 : mu / xmy * area[i] * 2;
                    }
                    else {
                        if(i==0)  fprintf(stderr, "unsupported energy type %d with param %f for Composite Majorizer\n", energy_type, s);
                    }
                }

                projMeshHessians(hessian + i * 36, alphas.data(), betas.data(), fzr[i], fzi[i], gzr[i], gzi[i], Dr.col(i), Di.col(i), projMethod);
            }
        }
    }

    return en;
}
inline void mexError(const std::string& error)
{
    mexErrMsgTxt(("invalid input to mex: " + error).c_str());
}

template<typename R>
R getFieldValueWithDefault(const mxArray* mat, const char* name, R defaultvalue)
{
    const mxArray *f = mat?mxGetField(mat, 0, name):nullptr;
    return f ? R(mxGetScalar(f)) : defaultvalue;
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
    // h = meshIsometryicEnergy(fz, gz, D, area, option)
    // fz, gz: n
    // D:  3 x n
    // area: n

    using std::chrono::steady_clock;
    auto t = steady_clock::now();
    if (nrhs < 4)  mexError("not enough input");

	const size_t n = mxGetNumberOfElements(prhs[0]); // number of faces
    if (   mxGetNumberOfElements(prhs[1]) != n || mxGetNumberOfElements(prhs[3]) != n
        || mxGetM(prhs[2]) != 3 || mxGetN(prhs[2]) != n)  mexError("bad input, inconsistent dimensions");

    const double *fzr = mxGetPr(prhs[0]);
    const double *fzi = mxGetPi(prhs[0]);
    const double *gzr = mxGetPr(prhs[1]);
    const double *gzi = mxGetPi(prhs[1]);
    Map<const MatrixXd> Dr(mxGetPr(prhs[2]), 3, n);
    Map<const MatrixXd> Di(mxGetPi(prhs[2]), 3, n);
    const double *Area = mxGetPr(prhs[3]);
	
    // get parameters
    const mxArray *params = nrhs>4?prhs[4]:nullptr;
    int hessProj = getFieldValueWithDefault<int>(params, "hessian_projection", HESS_PROJ_KP);
    int energy_type = getFieldValueWithDefault<int>(params, "energy_type", ISO_ENERGY_SYMMETRIC_DIRICHLET);
    double energy_param = getFieldValueWithDefault<double>(params, "energy_param", 1);
    bool verbose = getFieldValueWithDefault<bool>(params, "verbose", false);

    if (nlhs > 1) plhs[1] = mxCreateDoubleMatrix(6, n, mxREAL);
    if (nlhs > 2) plhs[2] = mxCreateDoubleMatrix(36, n, mxREAL);

    double *const grad = nlhs>1?mxGetPr(plhs[1]):nullptr;
    double *const hessian = nlhs>2?mxGetPr(plhs[2]):nullptr;

    int nthreads = 0;
    if (verbose) {
        printf("compute energy %d, energy param %f\n", energy_type, energy_param);
        printf("%-30s %fs\n", "preprocess", elapsedseconds(t));

#pragma omp parallel
        if (omp_get_thread_num() == 0)
            nthreads = omp_get_num_threads();
    }

    t = steady_clock::now();

    plhs[0] = mxCreateDoubleScalar( evalIsometricEnergy(fzr, fzi, gzr, gzi, Dr, Di, Area, energy_type, energy_param, grad, hessian, hessProj) );

    if (verbose) {
        char str[100];
        sprintf(str, "project (method %d, %d threads)", hessProj, nthreads);
        printf("%-30s %fs\n", str, elapsedseconds(t));
    }
}
