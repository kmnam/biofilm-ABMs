/**
 * Authors:
 *     Kee-Myoung Nam
 *
 * Last updated:
 *     3/18/2024
 */

#ifndef QUADRICS_HPP
#define QUADRICS_HPP

#include <cmath>
#include <complex>
#include <array>
#include <map>
#include <Eigen/Dense>
#include "polynomials.hpp"

using namespace Eigen;

/**
 * Return the Cartesian equation for a plane with the given normal vector
 * that passes through the given point. 
 */
template <typename RealType>
MultivariatePolynomial<RealType, 3> getPlane(const Ref<const Matrix<RealType, 3, 1> >& p,
                                             const Ref<const Matrix<RealType, 3, 1> >& n)
{
    std::map<std::array<int, 3>, std::complex<RealType> > coefmap;
    coefmap.insert({{0, 0, 0}, -n.dot(p)}); 
    coefmap.insert({{1, 0, 0}, n(0)}); 
    coefmap.insert({{0, 1, 0}, n(1)});
    coefmap.insert({{0, 0, 1}, n(2)});

    return MultivariatePolynomial<RealType, 3>(coefmap); 
}

/**
 * Return the Cartesian equation for a sphere with the given center and radius.
 */
template <typename RealType>
MultivariatePolynomial<RealType, 3> getSphere(const Ref<const Matrix<RealType, 3, 1> >& r,
                                              const RealType R)
{
    // The polynomial should read as 
    //
    // (x-rx)^2 + (y-ry)^2 + (z-rz)^2 - R^2,
    //
    // which expands to 
    //
    // x^2 - 2*rx*x + rx^2 + y^2 - 2*ry*y + ry^2 + z^2 - 2*rz*z + rz^2 - R^2
    const RealType rx = r(0);
    const RealType ry = r(1);
    const RealType rz = r(2);
    std::map<std::array<int, 3>, std::complex<RealType> > coefmap; 
    coefmap.insert({{0, 0, 0}, r.dot(r) - R * R}); 
    coefmap.insert({{1, 0, 0}, -2 * rx});
    coefmap.insert({{0, 1, 0}, -2 * ry});
    coefmap.insert({{0, 0, 1}, -2 * rz}); 
    coefmap.insert({{2, 0, 0}, 1});
    coefmap.insert({{0, 2, 0}, 1}); 
    coefmap.insert({{0, 0, 2}, 1});

    return MultivariatePolynomial<RealType, 3>(coefmap); 
}

/**
 * Return the Cartesian equation for a cylinder with the given center, 
 * orientation vector, and radius.
 */
template <typename RealType>
MultivariatePolynomial<RealType, 3> getCylinder(const Ref<const Matrix<RealType, 3, 1> >& r,
                                                const Ref<const Matrix<RealType, 3, 1> >& n,
                                                const RealType R)
{
    // First define the quadratic form for the cylinder through the origin
    // with the given orientation vector
    Matrix<std::complex<RealType>, 3, 3> Q
        = (Matrix<RealType, 3, 3>::Identity() - (n * n.transpose())).template cast<std::complex<RealType> >();

    // Then define the desired cylinder in terms of the center coordinates
    // and the quadratic form
    //
    // Namely, the equation should read as 
    //
    // a*(x-rx)^2 + b*(y-ry)^2 + c*(z-rz)^2 + 2*d*(x-rx)*(y-ry) + 2*e*(x-rx)*(z-rz)
    //     + 2*f*(y-ry)*(z-rz) - R^2,
    //
    // where the coefficients a, b, c, d, e, f are given in the quadratic
    // form. This polynomial expands to
    //
    // a*x^2 - 2*a*rx*x + a*rx^2
    //     + b*y^2 - 2*b*ry*y + b*ry^2
    //     + c*z^2 - 2*c*rz*z + c*rz^2
    //     + 2*d*x*y - 2*d*ry*x - 2*d*rx*y + 2*d*rx*ry
    //     + 2*e*x*z - 2*e*rz*x - 2*e*rx*z + 2*e*rx*rz
    //     + 2*f*y*z - 2*f*rz*y - 2*f*ry*z + 2*f*ry*rz - R^2
    const std::complex<RealType> a = Q(0, 0);
    const std::complex<RealType> b = Q(1, 1);
    const std::complex<RealType> c = Q(2, 2); 
    const std::complex<RealType> d = Q(0, 1);
    const std::complex<RealType> e = Q(0, 2); 
    const std::complex<RealType> f = Q(1, 2);
    const RealType rx = r(0);
    const RealType ry = r(1);
    const RealType rz = r(2);
    const RealType two = 2;
    std::map<std::array<int, 3>, std::complex<RealType> > coefmap;
    coefmap.insert({{0, 0, 0}, a*rx*rx + b*ry*ry + c*rz*rz + two*(d*rx*ry + e*rx*rz + f*ry*rz) - R*R});
    coefmap.insert({{1, 0, 0}, -two*(a*rx + d*ry + e*rz)});
    coefmap.insert({{0, 1, 0}, -two*(b*ry + d*rx + f*rz)});
    coefmap.insert({{0, 0, 1}, -two*(c*rz + e*rx + f*ry)});
    coefmap.insert({{1, 1, 0}, two * d}); 
    coefmap.insert({{1, 0, 1}, two * e});
    coefmap.insert({{0, 1, 1}, two * f});
    coefmap.insert({{2, 0, 0}, a});
    coefmap.insert({{0, 2, 0}, b});
    coefmap.insert({{0, 0, 2}, c});

    return MultivariatePolynomial<RealType, 3>(coefmap);
}

#endif
