#ifndef LOGSPIRAL_H
#define LOGSPIRAL_H

#define _USE_MATH_DEFINES
#include <Eigen/Dense>
#include <complex>
#include <limits>
using namespace std;

#undef min
#undef max

template<class R>
using vec2 = Eigen::Matrix < R, 1, 2 > ;


template<class T>
T rotationAngle(const vec2<T> &v1, const vec2<T> &v2) {
    // Apparently atan2 doesn't care about unit-length vectors
    T cosine = v1.dot(v2);
    //T sine = v1.cross(v2);
    T sine = v1[0] * v2[1] - v1[1] * v2[0];
    T angle = atan2(sine, cosine); // atan2 does sine and then cosine

    return (angle<0)? angle+2*M_PI : angle; // atan2 does [-pi, pi]
}



template <class T>
struct LogSpiral {
    vec2<T> p0;
    T c;
    T alpha;

    vec2<T> evaluate(const vec2<T> &p, T t) {
        T exponential = exp(c*t);
        vec2<T> diff = p-p0;

        T cosine = cos(t*alpha);
        T sine = sin(t*alpha);

        vec2<T> rotated(cosine*diff[0]-sine*diff[1],sine*diff[0]+cosine*diff[1]);

        return p0 + exponential * rotated;
    }
};

template<class T>
inline T evaluateFunction(T k, T phi, T b1, T b2, T b3, T b4) {
    return exp(k*phi) - (k*b1+b2)/(k*b3+b4);
}

template<class T>
inline T evaluateDerivative(T k, T phi, T b1, T b2, T b3, T b4) {
    T denom = b3*k+b4;
    return (b2*b3-b1*b4)/(denom*denom) + phi*exp(k*phi);
}

template<class T>
inline T evaluateSecondDerivative(T k, T phi, T b1, T b2, T b3, T b4) {
    T denom = b3*k+b4;
    return phi*phi*exp(k*phi) - 2*b3*(b2*b3-b1*b4)/(denom*denom*denom);
}

template<class T>
T findRoot(T phi, T b1, T b2, T b3, T b4, T left, T right) {
    // For now, we'll just use bisection with linear interpolation to find midpoint
    // Assume f(left)*f(right) < 0

    T leftEval = evaluateFunction(left, phi, b1, b2, b3, b4);
    T rightEval = evaluateFunction(right, phi, b1, b2, b3, b4);

    if (leftEval * rightEval > 0) return numeric_limits<T>::max();

    T midpoint = left - leftEval / (rightEval-leftEval) * (right-left);
    T midpointEval = evaluateFunction(midpoint, phi, b1, b2, b3, b4);

    int count = 0;
    while (true) {
        if (++count > 500) exit(0);

        if (fabs(midpointEval) < 1e-8 || fabs(right-left) < 1e-8) break;

        if (midpointEval * leftEval > 0) {
            left = midpoint;
            leftEval = midpointEval;
        } else {
            right = midpoint;
            rightEval = midpointEval;
        }

        // every five iterations cut interval in half in case something is broken
        if (count % 5 == 0) midpoint = (left + right) / 2;
        else midpoint = left - leftEval / (rightEval-leftEval) * (right-left);

        midpointEval = evaluateFunction(midpoint, phi, b1, b2, b3, b4);
    }

    return midpoint;
}

template<class T>
LogSpiral<T> computeSpiralParameters(vec2<T> s0, vec2<T> s1, vec2<T> t0, vec2<T> t1, T k, T gamma0, T phi) {
    LogSpiral<T> result;

    // alpha = arccot(k) -- also, who uses arccot?  srsly?
    T alpha;
    alpha = M_PI/2 - atan(k);
    //qWarning("alpha = %g", alpha);
    //qWarning("upper bound? %g", M_PI - gamma0);

    // they're complementary...
    T beta = M_PI - alpha;
    //qWarning("beta = %g", beta);

    // from figures 3-6, can get a vector pointing toward o from s1 if you take -t1 and rotate
    // clockwise by angle alpha

    vec2<T> d1 = -t1;
    d1.rotate(-alpha); // "rotate" function is counterclockwise, so need -alpha

    // similarly, take t0 and rotate counterclockwise by beta to point toward o
    vec2<T> d0 = t0;
    d0.rotate(beta);

    // to find base point, we have two rays to intersect
    // 1.  ray starting at s1 and direction pointer1
    // 2.  ray starting at s0 and direction pointer0

    vec2<T> delta = s1 - s0;
    T t = -d1.cross(delta)/d0.cross(d1);
    //T t2 = -d0.cross(delta)/d0.cross(d1);

    result.p0 = s0 + t * d0;

    //qWarning("Log spiral vertex = (%g, %g)", result.p0.x, result.p0.y);

    // We need the expansion and angular rates to finish off

    result.alpha = phi; // want to traverse angle phi in 1 time unit

    T r0 = (s0-result.p0).norm();
    T r1 = (s1-result.p0).norm();

    result.c = log(r1/r0);

    return result;
}

template<class T>
LogSpiral<T> fitSpiral(vec2<T> s0, vec2<T> s1, vec2<T> t0, vec2<T> t1);

#endif // LOGSPIRAL_H
