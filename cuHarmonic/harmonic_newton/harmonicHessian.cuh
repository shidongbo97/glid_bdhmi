#include "harmonic_map.cuh"

template<class Complex, class real>
void hessianIsometric(const Complex *d_D2, const Complex *d_phipsy, const Complex *d_C2, const Complex *d_bP2P, Complex *const dpp, int m, int mh, int n, int nP2P, real isoepow, real lambda, real *const hessian = nullptr, int nIter = 1)
{
    cuHarmonicMap<Complex, real> HM(d_D2, m, n, mh);
    HM.init();

    HM.isoepow = isoepow;

    HM.setupP2P(d_C2, nP2P, lambda);
    HM.update_bP2P(d_bP2P);
    HM.update_phipsy(d_phipsy);
    //HM.isometry_hessian(d_H);
    //HM.hessian_full(d_H);
    for(int it=0; it<nIter; it++)
        HM.computeNewtonStep(dpp);

    if (hessian) {
        //HM.update_fzgz();
        //for(int it=0; it<nIter; it++)
        HM.isometry_hessian();
        //myCopy_n(HM.h.data(), n*n * 16, hessian);
        //Complex* const pgrads = (Complex*)hessian + n * 2 * (n * 4 - 2);
        //myZeroFill(pgrads, n * 2 * 2);
        //HM.isometry_gradient(pgrads);
        //HM.p2p_gradient(pgrads + n * 2);
    }
}
