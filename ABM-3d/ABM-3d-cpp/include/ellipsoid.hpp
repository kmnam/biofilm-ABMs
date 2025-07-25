/**
 * Utility functions regarding ellipsoidal bodies. 
 *
 * Authors:
 *     Kee-Myoung Nam
 *
 * Last updated:
 *     7/24/2025
 */

#ifndef BIOFILM_ELLIPSOID_HPP
#define BIOFILM_ELLIPSOID_HPP

#include <iomanip>
#include <cmath>
#include <utility>
#include <Eigen/Dense>
#include <boost/math/special_functions/ellint_1.hpp>
#include <boost/math/special_functions/ellint_2.hpp>
#include <boost/multiprecision/mpfr.hpp>
#include "utils.hpp"

using namespace Eigen; 

using std::sin;
using boost::multiprecision::sin;
using std::cos;
using boost::multiprecision::cos;
using std::sqrt;
using boost::multiprecision::sqrt;
using std::pow; 
using boost::multiprecision::pow; 
using std::atan2; 
using boost::multiprecision::atan2; 

/**
 * Get the rotation matrix that rotates the given vector to the unit vector, 
 * (1, 0, 0). 
 *
 * @param n Input orientation vector. 
 * @returns Rotation matrix. 
 */
template <typename T>
Matrix<T, 3, 3> rotationToXUnitVector(const Ref<const Matrix<T, 3, 1> >& n)
{
    // This uses the prescription given by: 
    //
    // https://math.stackexchange.com/questions/114512/
    // how-to-find-the-orthonormal-transformation-that-will-rotate-a-vector-to-the-x-ax
    T theta = atan2(-n(2), n(1));
    T alpha = atan2(-(cos(theta) * n(1) - sin(theta) * n(2)), n(0));
    Matrix<T, 3, 3> A, B; 
    A << 1,          0,           0,
         0, cos(theta), -sin(theta), 
         0, sin(theta),  cos(theta); 
    B << cos(alpha), -sin(alpha), 0,
         sin(alpha),  cos(alpha), 0,
                  0,           0, 1; 

    return B * A; 
}

/**
 * Get the quadratic form that represents the prolate ellipsoid with the
 * given center, orientation (long axis), radius (semi-minor axis length), and 
 * centerline half-length (semi-major axis length minus radius).
 *
 * This function returns the matrix A, vector b, and scalar c such that the
 * ellipsoid is given by the equation, 
 *
 * x.T * A * x + 2 * b.T * x = c.
 *
 * The interior of the ellipsoid satisfies x.T * A * x + 2 * b.T * x <= c.
 */
template <typename T>
std::tuple<Matrix<T, 3, 3>, Matrix<T, 3, 1>, T> getEllipsoidQuadraticForm(const Ref<const Matrix<T, 3, 1> >& r,
                                                                          const Ref<const Matrix<T, 3, 1> >& n, 
                                                                          const T R, 
                                                                          const T half_l)
{
    T a = half_l + R;    // Semi-major axis length
    T b = R;             // Semi-minor axis length
    
    // Get the rotation matrix that sends the orientation vector to (1, 0, 0)
    Matrix<T, 3, 3> rot = rotationToXUnitVector<T>(n);

    // Define the components of the quadratic form 
    Matrix<T, 3, 3> A, B;
    A << 1.0 / (a * a),             0,             0,
                     0, 1.0 / (b * b),             0,
                     0,             0, 1.0 / (b * b);
    B = rot.transpose() * A * rot;
    Matrix<T, 3, 1> v = -B * r;
    T c = 1.0 - r.dot(B * r);

    return std::make_tuple(B, v, c); 
}

/**
 * Project the given query point, a, onto the surface of the given ellipsoid
 * given by the quadratic form, 
 *
 * x.T * A * x <= 1,
 *
 * and center r, using the Lin-Han algorithm (Section 3 in Dai et al., SIAM
 * J. Optim., and Algorithm 2 in Jia et al., J. Comput. Appl. Math 2017).
 *
 * The problem is solved by first centering the ellipsoid at the origin, 
 * then translating the result by r. 
 */
template <typename T>
Matrix<T, 3, 1> projectOntoEllipsoid(const Ref<const Matrix<T, 3, 1> >& p,
                                     const Ref<const Matrix<T, 3, 3> >& A,
                                     const Ref<const Matrix<T, 3, 1> >& r,
                                     const T tol, const int max_iter,
                                     const bool verbose = false,
                                     const int verbose_iter = 100)
{
    // First solve the problem for the ellipsoid x.T * A * x <= 1 centered
    // at the origin
    //
    // Start by locating a point on the ellipsoid
    LDLT<Matrix<T, 3, 3> > cholesky(A);
    std::function<T(const Ref<const Matrix<T, 3, 1> >&)> eval
        = [&cholesky](const Ref<const Matrix<T, 3, 1> >& q) -> T
        {
            Matrix<T, 3, 1> d = cholesky.vectorD();
            Matrix<T, 3, 1> v = cholesky.matrixL().transpose() * (cholesky.transpositionsP() * q);
            return d(0) * v(0) * v(0) + d(1) * v(1) * v(1) + d(2) * v(2) * v(2); 
        }; 
    T p_val = eval(p); 
    Matrix<T, 3, 1> x = p / sqrt(p_val);

    // Now get the spectral radius of A 
    SelfAdjointEigenSolver<Matrix<T, 3, 3> > eigen(A);
    T rho = eigen.eigenvalues().maxCoeff(); 
    T gamma = 1.0 / rho;
    Matrix<T, 3, 1> g = A * p; 

    // Generate the initial iterate 
    int k = 0;
    Matrix<T, 3, 1> u = A * x;
    Matrix<T, 3, 1> v = p - x;
    T eps = u.dot(v) / (u.norm() * v.norm());
    while (k < max_iter && 1 - eps >= tol)
    {
        // Calculate the next iterate
        k++;  
        Matrix<T, 3, 1> c = x - gamma * u; 
        Matrix<T, 3, 1> w = c - p;
        T term = g.dot(w) / eval(w); 
        T eta = -term - sqrt(term * term - (p_val - 1) / eval(w));
        x = p + eta * w;
        u = A * x; 
        v = p - x; 
        eps = u.dot(v) / (u.norm() * v.norm());
    }

    return x + r;  
}

/**
 * Get the principal radii of curvature at the given point x along the surface
 * of the prolate ellipsoid with major axis orientation n, semi-major axis 
 * length R + half_l, and semi-minor axis length R. 
 */
template <typename T>
std::pair<T, T> getPrincipalRadiiOfCurvature(const Ref<const Matrix<T, 3, 1> >& n, 
                                             const T R, 
                                             const T half_l, 
                                             const Ref<const Matrix<T, 3, 1> >& x,
                                             const T umbilic_tol = 1e-8)
{
    // Get the rotation matrix that maps the orientation vector to (1, 0, 0)
    Matrix<T, 3, 3> rot = rotationToXUnitVector<T>(n);

    // Rotate the input point
    Matrix<T, 3, 1> y = rot * x;  

    // Convert to parametric coordinates u and v, where u is the azimuthal angle
    // (in [0, 2\pi)) and v is the polar angle (in [0, \pi))
    //
    // Here, we assume the conventions in:
    // https://mathworld.wolfram.com/Ellipsoid.html
    // 
    // Namely, we let a = b be the short semi-axis lengths and let c be the 
    // long semi-axis length 
    //
    // Note that, for curvature calculations, we do not need u to cover the 
    // entire range of [0, 2\pi), since the curvature is symmetric across each
    // axis 
    T a = R; 
    T c = half_l + R;

    // Calculate sin^2(u), cos^2(u), sin^2(v), cos^2(v)
    T cosv = y(0) / c;      // This is the coordinate corresponding to the long axis, c
    T cos2v = cosv * cosv; 
    T sin2v = 1 - cos2v;
    T v = acosSafe<T>(cosv);
    T sinv = sin(v); 
    T cosu = y(2) / (a * sinv);
    T cos2u = cosu * cosu; 
    T sin2u = 1 - cos2u;
    T u = acosSafe<T>(cosu);
    
    // Then calculate the Gaussian curvature ...
    T gauss_term1 = a * a * a * a * c * c; 
    T gauss_term2 = a * a * a * a * cos2v; 
    T gauss_term3 = c * c * a * a * sin2v; 
    T gauss = gauss_term1 / pow(gauss_term2 + gauss_term3, 2);

    // ... and the mean curvature ...
    T mean_numer = 6 * a * a + 2 * c * c + 2 * (a * a - c * c) * cos(2 * v);
    mean_numer *= a * a * c;
    T mean = mean_numer / (8 * pow(gauss_term2 + gauss_term3, 1.5));
    
    // ... from which we can calculate the principal radii of curvature
    //
    // If the square mean curvature and Gaussian curvature are very close
    // (i.e., x is an umbilic point), then the two principal curvatures are 
    // the same (the mean curvature)
    T principal1, principal2;
    if (abs(mean * mean - gauss) < umbilic_tol)
    {
        principal1 = mean; 
        principal2 = mean;
    }
    else 
    {
        principal1 = mean + sqrt(mean * mean - gauss);    // Maximum curvature 
        principal2 = mean - sqrt(mean * mean - gauss);    // Minimum curvature
    }
   
    // Return as (maximum radius of curvature, minimum radius of curvature) 
    return std::make_pair(1.0 / principal2, 1.0 / principal1);   
}

/**
 * Calculate a table of values for the function, 
 *
 * (E(e) / (1 - e^2) - K(e)) / (K(e) - E(e)), 
 *
 * where K(e) is the complete elliptic integral of the first kind,
 *
 * K(e) = \int_0^{\pi/2}{d\theta / \sqrt{1 - e^2 \cos^2{\theta}}} 
 *
 * and E(e) is the complete elliptic integral of the second kind, 
 *
 * E(e) = \int_0^{\pi/2}{\sqrt{1 - e^2 \cos^2{\theta}} d\theta}
 */
template <typename T>
Matrix<T, Dynamic, 4> getEllipticIntegralTable(const int n)
{
    Matrix<T, Dynamic, 4> ellip_table(n, 4); 
    Matrix<T, Dynamic, 1> e = Matrix<T, Dynamic, 1>::LinSpaced(n, 0.001, 0.999);
    for (int i = 0; i < n; ++i)
    {
        T Ke = boost::math::ellint_1<T>(e(i));
        T Ee = boost::math::ellint_2<T>(e(i)); 
        ellip_table(i, 0) = e(i);  
        ellip_table(i, 1) = Ke; 
        ellip_table(i, 2) = Ee; 
        ellip_table(i, 3) = (Ee / (1 - e(i) * e(i)) - Ke) / (Ke - Ee); 
    }

    return ellip_table; 
}

/**
 * Given a point on a spherocylinder's surface, solve for the principal
 * radii of curvature at the nearest point on the inscribed ellipsoid.
 *
 * This function assumes that the spherocylinder is centered at the origin
 * and has orientation vector (1, 0, 0). Moreover, the point on the 
 * spherocylinder's surface is determined in terms of the angle formed by
 * the orientation vector and the normalized overlap vector.  
 *
 * @param half_l Spherocylinder centerline half-length. 
 * @param R Spherocylinder radius. 
 * @param theta Angle between the orientation vector and the normalized 
 *              overlap vector.    
 * @param s Centerline coordinate along the spherocylinder determining tail
 *          of overlap vector.  
 * @param project_tol Tolerance for ellipsoid projection. 
 * @param project_max_iter Maximum number of iterations for ellipsoid projection. 
 * @returns Principal radii of curvature at the projection of the input point
 *          onto the inscribed ellipsoid.  
 */
template <typename T>
std::pair<T, T> projectAndGetPrincipalRadiiOfCurvature(const T half_l, const T R, 
                                                       const T theta, const T s,
                                                       const T project_tol = 1e-6,
                                                       const int project_max_iter = 1000,
                                                       const bool verbose = false, 
                                                       const int verbose_iter = 100)
{
    // Project outward from the centerline along the distance vector, until 
    // we identify points that lie outside the ellipsoids corresponding to 
    // the two cells 
    Matrix<T, 3, 1> r, n; 
    r << 0, 0, 0; 
    n << 1, 0, 0; 
    auto form = getEllipsoidQuadraticForm<T>(r, n, R, half_l); 
    Matrix<T, 3, 3> A = std::get<0>(form);
    Matrix<T, 3, 1> b = std::get<1>(form);
    T c = std::get<2>(form); 
    std::function<bool(const Ref<const Matrix<T, 3, 1> >&)> in_ellipsoid
        = [&A, &b, &c](const Ref<const Matrix<T, 3, 1> >& q) -> bool
        {
            return (q.dot(A * q) + 2 * b.dot(q) <= c); 
        };
    Matrix<T, 3, 1> dnorm; 
    dnorm << cos(theta), sin(theta), 0; 
    Matrix<T, 3, 1> u = r + s * n + R * dnorm; 
    while (in_ellipsoid(u))
        u += 0.1 * dnorm;

    // Now project this point onto the ellipsoid surface 
    u = projectOntoEllipsoid<T>(
        u, A, r, project_tol, project_max_iter, verbose, verbose_iter
    );

    // Compute the principal radii of curvature at this point 
    return getPrincipalRadiiOfCurvature<T>(n, R, half_l, u); 
}

/**
 * Given a point on a spherocylinder's surface, solve for the principal
 * radii of curvature at the nearest point on the inscribed ellipsoid.
 *
 * @param r Spherocylinder center. 
 * @param n Spherocylinder orientation. 
 * @param half_l Spherocylinder centerline half-length. 
 * @param R Spherocylinder radius. 
 * @param dnorm Normalized overlap vector along which the point on the 
 *              spherocylinder is determined. 
 * @param s Centerline coordinate along the spherocylinder determining tail
 *          of overlap vector.  
 * @param project_tol Tolerance for ellipsoid projection. 
 * @param project_max_iter Maximum number of iterations for ellipsoid projection. 
 * @returns Principal radii of curvature at the projection of the input point
 *          onto the inscribed ellipsoid.  
 */
template <typename T>
std::pair<T, T> projectAndGetPrincipalRadiiOfCurvature(const Ref<const Matrix<T, 3, 1> >& r, 
                                                       const Ref<const Matrix<T, 3, 1> >& n,
                                                       const T half_l, const T R, 
                                                       const Ref<const Matrix<T, 3, 1> >& dnorm,
                                                       const T s,
                                                       const T project_tol = 1e-6,
                                                       const int project_max_iter = 1000,
                                                       const bool verbose = false, 
                                                       const int verbose_iter = 100)
{
    // Project outward from the centerline along the distance vector, until 
    // we identify points that lie outside the ellipsoids corresponding to 
    // the two cells
    Matrix<T, 3, 1> origin = Matrix<T, 3, 1>::Zero();  
    auto form = getEllipsoidQuadraticForm<T>(origin, n, R, half_l); 
    Matrix<T, 3, 3> A = std::get<0>(form);
    std::function<bool(const Ref<const Matrix<T, 3, 1> >&)> in_ellipsoid
        = [&A, &r](const Ref<const Matrix<T, 3, 1> >& q) -> bool
        {
            return ((q - r).dot(A * (q - r)) <= 1.0); 
        };
    Matrix<T, 3, 1> u = r + s * n + R * dnorm; 
    while (in_ellipsoid(u))
        u += 0.1 * dnorm;

    // Now project this point onto the ellipsoid surface 
    u = projectOntoEllipsoid<T>(
        u, A, r, project_tol, project_max_iter, verbose, verbose_iter
    );

    // Compute the principal radii of curvature at this point 
    return getPrincipalRadiiOfCurvature<T>(n, R, half_l, u); 
}

#endif
