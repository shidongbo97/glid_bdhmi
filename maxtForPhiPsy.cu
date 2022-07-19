#include <cuda_runtime.h>
#include <math_constants.h>

__global__ void maxtForPhiPsy(double *t, const double2 * fz, const double2 * gz, const double2 *dfz, const double2 *dgz, int n, double kmax=1.) 
{
	const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i>=n) return;

    const double k2 = kmax*kmax;

    auto abs2 = [](const double2 &v) { return v.x*v.x + v.y*v.y; };
    auto sum  = [](const double2 &v) { return v.x + v.y; };
    auto dot  = [](const double2 &a, const double2 &b) { return a.x*b.x + a.y*b.y; };

    double a = k2*abs2(dfz[i]) - abs2(dgz[i]); 
    double b = 2*( k2* dot(fz[i], dfz[i]) - dot(gz[i], dgz[i]) );
    double c = k2*abs2(fz[i]) - abs2(gz[i]);

    double delta = b*b - 4*a*c;

    t[i] = CUDART_INF;  //CUDART_INF_F;  // for single precision
    if( !(a>0 && (delta<0 || b>0)) )
        t[i] = (-b-sqrt(delta))/a/2;
}
