/**
 * Utility functions regarding ellipsoidal bodies. 
 *
 * Authors:
 *     Kee-Myoung Nam
 *
 * Last updated:
 *     6/3/2025
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

using namespace Eigen; 

using std::sin;
using boost::multiprecision::sin;
using std::cos;
using boost::multiprecision::cos;
using std::acos; 
using boost::multiprecision::acos; 
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
 * This function returns the matrix A and vector b such that the ellipsoid
 * is given by the equation, 
 *
 * x.T * A * x + b.T * x = 1. 
 */
template <typename T>
std::pair<Matrix<T, 3, 3>, Matrix<T, 3, 1> > getEllipsoidQuadraticForm(const Ref<const Matrix<T, 3, 1> >& r,
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
    Matrix<T, 3, 1> v = -2 * B * r; 
    T c = 1.0 - r.dot(B * r);
    B /= c; 
    v /= c; 

    return std::make_pair(B, v); 
}

/**
 * Project the given query point, a, onto the surface of the given ellipsoid
 * (given by matrix A and vector b), using the alternating direction method 
 * of multipliers method (ADMM) algorithm proposed by:
 *
 * Jia et al., Comparison of several fast algorithms for projection onto an
 * ellipsoid, J. Comput. Appl. Math. (2017) 319, 320-337. 
 */
template <typename T>
Matrix<T, 3, 1> projectOntoEllipsoid(const Ref<const Matrix<T, 3, 1> >& a,
                                     const Ref<const Matrix<T, 3, 3> >& A, 
                                     const Ref<const Matrix<T, 3, 1> >& b,
                                     const T tol, const int max_iter,
                                     const bool verbose = false)
{
    int k = 0; 
    Matrix<T, 3, 1> x = a / sqrt(a.norm());         // Initialize as suggested by Jia et al. 
    Matrix<T, 3, 1> l = Matrix<T, 3, 1>::Ones();    // Initialize to all-ones vector
    
    // Set theta to 1 / sqrt(cond), where cond is the condition number of A 
    JacobiSVD<Matrix<T, 3, 3> > svd(A);
    T max_singval = svd.singularValues()(0);
    T min_singval = svd.singularValues()(2);
    T theta = 1.0 / sqrt(max_singval / min_singval);  

    // Calculate auxiliary matrices and vectors that are fixed throughout
    // the optimization 
    Matrix<T, 3, 3> B = A.llt().matrixL();
    Matrix<T, 3, 1> bbar = (-B.transpose()).fullPivLu().solve(b);
    const T r = sqrt(1.0 + pow(bbar.norm(), 2));
    Matrix<T, 3, 3> inv_Abar = Matrix<T, 3, 3>::Identity() + theta * B.transpose() * B;
    FullPivLU<Matrix<T, 3, 3> > Abar_decomp(inv_Abar); 

    // Initialize and calculate the initial residual
    Matrix<T, 3, 1> y = B * x - bbar;
    Matrix<T, 9, 1> res; 
    res(Eigen::seq(0, 2)) = x - a - B.transpose() * l;
    if ((y - l).norm() <= r)
        res(Eigen::seq(3, 5)) = l; 
    else 
        res(Eigen::seq(3, 5)) = y - (y - l) / (y - l).norm() * r; 
    res(Eigen::seq(6, 8)) = B * x - y - bbar;

    // Print initial iterate, if desired 
    if (verbose) 
    {
        std::cout << std::setprecision(10); 
        std::cout << "Query point: (" << a(0) << ", " << a(1) << ", " << a(2) << ")" << std::endl;
        std::cout << "Ellipsoid matrix:\n[[" << A(0, 0) << ", " << A(0, 1) << ", " << A(0, 2) << "]"  << std::endl
                                     << " [" << A(1, 0) << ", " << A(1, 1) << ", " << A(1, 2) << "]"  << std::endl 
                                     << " [" << A(2, 0) << ", " << A(2, 1) << ", " << A(2 ,2) << "]]" << std::endl;
        std::cout << "Square root:\n[[" << B(0, 0) << ", " << B(0, 1) << ", " << B(0, 2) << "]"  << std::endl
                                << " [" << B(1, 0) << ", " << B(1, 1) << ", " << B(1, 2) << "]"  << std::endl 
                                << " [" << B(2, 0) << ", " << B(2, 1) << ", " << B(2 ,2) << "]]" << std::endl;
        std::cout << "Ellipsoid vector: (" << b(0) << ", " << b(1) << ", " << b(2) << ")" << std::endl; 
        std::cout << "Theta = " << theta << std::endl;  
        std::cout << "Initial iterate: (" << x(0) << ", " << x(1) << ", " << x(2) << ")" << std::endl; 
        std::cout << "- Residual = " << res.norm() << std::endl; 
    }

    // Calculate the next iterate ... 
    while (k < max_iter && res.norm() > tol)
    {
        Matrix<T, 3, 1> u = a + B.transpose() * l + theta * B.transpose() * (y + bbar); 
        x = Abar_decomp.solve(u);
        Matrix<T, 3, 1> w = B * x - l / theta - bbar;
        if (w.norm() <= r) 
            y = w; 
        else 
            y = r * w / w.norm();
        l -= theta * (B * x - y - bbar);

        // Calculate residual for the new iterate 
        res(Eigen::seq(0, 2)) = x - a - B.transpose() * l;
        if ((y - l).norm() <= r)
            res(Eigen::seq(3, 5)) = l; 
        else 
            res(Eigen::seq(3, 5)) = y - (y - l) / (y - l).norm() * r; 
        res(Eigen::seq(6, 8)) = B * x - y - bbar;  
        k++;

        // Print k-th iterate, if desired 
        if (verbose && k % 100 == 0) 
        {
            std::cout << "Iterate " << k << ": (" << x(0) << ", " << x(1) << ", " << x(2) << ")" << std::endl; 
            std::cout << "- Residual = " << res.norm() << std::endl; 
        }
    }

    // Print a warning message if neither termination criterion was satisfied
    if (verbose && (k > max_iter && res.norm() > tol))
        std::cout << "[WARN] Ellipsoid projection terminated without reaching "
                  << "desired tolerance" << std::endl; 

    return x;  
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
                                             const Ref<const Matrix<T, 3, 1> >& x)
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
    T v = acos(y(2) / c);
    T u = acos(y(0) / (sin(v) * a));
    
    // Then calculate the Gaussian curvature ...
    T gauss_term1 = a * a * a * a * c * c; 
    T gauss_term2 = a * a * a * a * cos(v) * cos(v);
    T gauss_term3 = c * c * a * a * sin(v) * sin(v); 
    T gauss = gauss_term1 / pow(gauss_term2 + gauss_term3, 2);

    // ... and the mean curvature ... 
    T mean_numer = 6 * a * a + 2 * c * c + 2 * (a * a - c * c) * cos(2 * v); 
    mean_numer *= a * a * c; 
    T mean = mean_numer / (8 * pow(gauss_term2 + gauss_term3, 1.5));
    
    // ... from which we can calculate the principal radii of curvature 
    T principal1 = mean + sqrt(mean * mean - gauss);    // Maximum curvature 
    T principal2 = mean - sqrt(mean * mean - gauss);    // Minimum curvature
   
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
        ellip_table(i, 3) = (Ee / (1 - e * e) - Ke) / (Ke - Ee); 
    }

    return ellip_table; 
}

#endif
