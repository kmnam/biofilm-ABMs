/**
 * Functions for computing JKR contact areas and forces between contacting
 * bodies.  
 *
 * Authors:
 *     Kee-Myoung Nam
 *
 * Last updated:
 *     7/24/2025
 */

#ifndef BIOFILM_JKR_HPP
#define BIOFILM_JKR_HPP

#include <cmath>
#include <algorithm>
#include <Eigen/Dense>
#include <boost/multiprecision/mpfr.hpp>
#include "polynomials.hpp"
#include "ellipsoid.hpp"

using namespace Eigen; 

using std::cos; 
using boost::multiprecision::cos; 
using std::acos;
using boost::multiprecision::acos;
using std::sqrt;
using boost::multiprecision::sqrt;

/**
 * An enum that enumerates the different JKR force types. 
 */
enum class JKRMode
{
    ISOTROPIC = 0,
    ANISOTROPIC = 1
};

/**
 * Solve for the Hertzian contact area between two contacting ellipsoidal
 * bodies.
 *
 * This calculation follows the approach prescribed in Barber (Eqs. 3.35 and
 * 3.36; see also Eqs. 3.19 and 3.29). 
 *
 * @param delta Overlap between the two contacting bodies. 
 * @param Rx1 Principal radius of curvature of body 1 along long axis. 
 * @param Ry1 Principal radius of curvature of body 1 along short axis.
 * @param Rx2 Principal radius of curvature of body 2 along long axis. 
 * @param Ry2 Principal radius of curvature of body 2 along short axis.
 * @param n1 Orientation vector of body 1.
 * @param n2 Orientation vector of body 2. 
 * @param ellip_table Values of the elliptic integral function for various
 *                    eccentricities between 0 and 1. 
 */
template <typename T>
T hertzContactArea(const T delta, const T Rx1, const T Ry1, const T Rx2, 
                   const T Ry2, const Ref<const Matrix<T, 3, 1> >& n1,
                   const Ref<const Matrix<T, 3, 1> >& n2, 
                   const Ref<const Matrix<T, Dynamic, 4> >& ellip_table)
{
    // First calculate B and A ... 
    T sum = 0.5 * (1.0 / Rx1 + 1.0 / Ry1 + 1.0 / Rx2 + 1.0 / Ry2);
    T theta = acosSafe<T>(n1.dot(n2));
    T delta1 = (1.0 / Rx1 - 1.0 / Ry1); 
    T delta2 = (1.0 / Rx2 - 1.0 / Ry2); 
    T diff = 0.5 * sqrt(
        delta1 * delta1 + delta2 * delta2 + 2 * delta1 * delta2 * cos(2 * theta)
    );
    T B = 0.5 * (sum + diff);
    T A = sum - B;

    // ... and check that B > A
    if (B < A)
    {
        // If not, then switch the x- and y-axes and recalculate 
        T Rx1_ = Ry1; 
        T Ry1_ = Rx1; 
        T Rx2_ = Ry2; 
        T Ry2_ = Rx2;
        delta1 = (1.0 / Rx1_ - 1.0 / Ry1_); 
        delta2 = (1.0 / Rx2_ - 1.0 / Ry2_);
        diff = 0.5 * sqrt(
            delta1 * delta1 + delta2 * delta2 + 2 * delta1 * delta2 * cos(2 * theta)
        ); 
        B = 0.5 * (sum + diff); 
        A = sum - B;
    }
    T ratio = B / A; 

    // Column 0: eccentricity 
    // Column 1: value of K(e), complete elliptic integral of first kind 
    // Column 2: value of E(e), complete elliptic integral of second kind
    // Column 3: value of LHS of Eqn. 3.29 in Barber, which is equal to B / A
    int nearest_idx; 
    if (ratio < ellip_table(0, 3))
    {
        nearest_idx = 0; 
    }
    else if (ratio > ellip_table(ellip_table.rows() - 1, 3))
    {
        nearest_idx = ellip_table.rows() - 1; 
    }
    else 
    {
        for (int i = 1; i < ellip_table.rows(); ++i)
        {
            if (ratio >= ellip_table(i - 1, 3) && ratio < ellip_table(i, 3))
            {
                T d1 = abs(ratio - ellip_table(i - 1, 3)); 
                T d2 = abs(ratio - ellip_table(i, 3));
                if (d1 < d2)
                    nearest_idx = i - 1; 
                else
                    nearest_idx = i; 
            }
        }
    }
    T eccentricity = ellip_table(nearest_idx, 0);
    T Ke = ellip_table(nearest_idx, 1); 
    T Ee = ellip_table(nearest_idx, 2);

    // From here, get the contact area dimensions
    T e2 = eccentricity * eccentricity;  
    T a = sqrt(delta * (1 - Ee / Ke) / (A * e2)); 
    T area = boost::math::constants::pi<T>() * a * a * sqrt(1 - e2);

    return area; 
}

/**
 * Solve for the JKR contact radius for a circular contact area, given the 
 * overlap, equivalent radius and elastic modulus of the contact bodies,
 * and adhesion energy density.
 *
 * @param delta Overlap between the two contacting bodies. 
 * @param R Equivalent radius of the two contacting bodies.
 * @parma E Equivalent elastic modulus of the two contacting bodies. 
 * @param gamma Surface energy density. 
 * @param imag_tol Tolerance for determining whether a root for the the JKR
 *                 contact radius polynomial is real.
 * @param aberth_tol Tolerance for Aberth-Ehrlich method. 
 * @returns The two possible JKR contact radii. Negative values will also 
 *          be returned (in which case only the positive value is a valid 
 *          radius). 
 */
template <typename T, int N = 100>
std::pair<T, T> jkrContactRadius(const T delta, const T R, const T E,
                                 const T gamma, const T imag_tol = 1e-20,
                                 const T aberth_tol = 1e-20)
{
    typedef boost::multiprecision::number<boost::multiprecision::mpfr_float_backend<N> >  RealType;
    typedef boost::multiprecision::number<boost::multiprecision::mpc_complex_backend<N> > ComplexType;

    // Solve the quartic equation for the square root of the contact area
    Matrix<RealType, Dynamic, 1> coefs(5);
    RealType delta_ = static_cast<RealType>(delta); 
    RealType gamma_ = static_cast<RealType>(gamma);
    RealType R_ = static_cast<RealType>(R);
    RealType E_ = static_cast<RealType>(E);  
    coefs << delta_ * delta_ * R_ * R_,
             -4 * R_ * R_  * boost::math::constants::pi<RealType>() * gamma_ / E_, 
             -2 * R_ * delta_, 
             0.0, 1.0;

    // Solve for the roots of the polynomial with the specified precision
    HighPrecisionPolynomial<N> p(coefs);
    Matrix<ComplexType, Dynamic, 1> roots = p.solveAberth(static_cast<RealType>(aberth_tol));

    // Return both of the two positive real roots
    std::vector<RealType> roots_real; 
    for (int i = 0; i < roots.size(); ++i)
    {
        if (abs(imag(roots(i))) < static_cast<RealType>(imag_tol))
            roots_real.push_back(real(roots(i))); 
    }

    // If there are fewer than 2 real roots identified, raise an exception
    if (roots_real.size() < 2)
    {
        throw std::runtime_error(
            "JKR contact radius is undefined; invalid number of real roots found "
            "from radius-overlap polynomial"
        );
    }
    // If there are exactly 2 real roots identified, return both 
    else if (roots_real.size() == 2)
    {
        // Sort the real roots in ascending order 
        std::sort(roots_real.begin(), roots_real.end()); 
        return std::make_pair(roots_real[0], roots_real[1]);
    } 
    else   // Otherwise, find the 2 real roots with the smallest imaginary parts
    {
        std::vector<std::pair<RealType, int> > roots_imag; 
        for (int i = 0; i < roots.size(); ++i)
        {
            if (abs(imag(roots(i))) < static_cast<RealType>(imag_tol))
                roots_imag.push_back(std::make_pair(imag(roots(i)), i)); 
        }
        std::sort(
            roots_imag.begin(), roots_imag.end(), 
            [](std::pair<RealType, int> a, std::pair<RealType, int> b)
            {
                return (abs(a.first) < abs(b.first)); 
            }
        );

        // Gather and sort the real roots in ascending order
        roots_real.clear();
        roots_real.push_back(real(roots(roots_imag[0].second)));
        roots_real.push_back(real(roots(roots_imag[1].second))); 
        std::sort(roots_real.begin(), roots_real.end());   
        return std::make_pair(roots_real[0], roots_real[1]);
    } 
}

/**
 * Solve for the estimated JKR contact area for an elliptical contact 
 * area.
 *
 * This function assumes that the two bodies are prolate ellipsoids whose
 * major semi-axis lengths are given by half_l1 + R and half_l2 + R, and
 * whose minor semi-axis lengths are given by R. 
 *
 * @param n1 Orientation of body 1 (long axis). 
 * @param half_l1 Half-length of body 1 centerline.
 * @param n2 Orientation of body 2 (long axis). 
 * @param half_l2 Half-length of body 2 centerline. 
 * @param R Body radius. 
 * @param d12 Overlap vector. 
 * @param s Centerline coordinate along body 1 determining tail of overlap vector.  
 * @param t Centerline coordinate along body 2 determining head of overlap vector. 
 * @param E0 Elastic modulus. 
 * @param gamma Surface energy density. 
 * @param ellip_table Pre-computed table containing values for the elliptic
 *                    integral function for various eccentricities between 0
 *                    and 1. 
 * @param project_tol Tolerance for ellipsoid projection. 
 * @param project_max_iter Maximum number of iterations for ellipsoid projection. 
 * @param imag_tol Tolerance for determining whether a root for the the JKR
 *                 contact radius polynomial is real.
 * @param aberth_tol Tolerance for the Aberth-Ehrlich method.
 * @returns Estimated JKR contact area.  
 */
template <typename T, int N = 100>
T jkrContactAreaEllipsoid(const Ref<const Matrix<T, 3, 1> >& r1, 
                          const Ref<const Matrix<T, 3, 1> >& n1, const T half_l1,  
                          const Ref<const Matrix<T, 3, 1> >& r2, 
                          const Ref<const Matrix<T, 3, 1> >& n2, const T half_l2,
                          const T R, const Ref<const Matrix<T, 3, 1> >& d12,
                          const T s, const T t, const T E0, const T gamma,
                          const Ref<const Matrix<T, Dynamic, 4> >& ellip_table,
                          const T project_tol = 1e-6, const int project_max_iter = 100,
                          const T imag_tol = 1e-20, const T aberth_tol = 1e-20)
{
    // Compute overlap between the two cells
    T dist = d12.norm(); 
    Matrix<T, 3, 1> d12n = d12 / dist; 
    T delta = 2 * R - dist;

    // Get the principal radii of curvature at the projected contact points
    std::pair<T, T> radii1 = projectAndGetPrincipalRadiiOfCurvature<T>(
        r1, n1, half_l1, R, d12n, s, project_tol, project_max_iter
    ); 
    std::pair<T, T> radii2 = projectAndGetPrincipalRadiiOfCurvature<T>(
        r2, n2, half_l2, R, -d12n, t, project_totl, project_max_iter
    );
    T Rx1 = radii1.first; 
    T Ry1 = radii1.second; 
    T Rx2 = radii2.first; 
    T Ry2 = radii2.second;
     
    // Compute the expected contact area for a Hertzian contact 
    T area = hertzContactArea<T>(delta, Rx1, Ry1, Rx2, Ry2, n1, n2, ellip_table);

    // Compute the correction factor to account for JKR-based adhesion
    //
    // TODO Take care of the possibility of hysteresis (in which case the 
    // lower radius value may need to be used) 
    std::pair<T, T> jkr_radius = jkrContactRadius<T, N>(
        delta, R, E0, gamma, imag_tol, aberth_tol
    );
    T jkr_radius_factor = jkr_radius.second / sqrt(R * delta);

    return area * jkr_radius_factor * jkr_radius_factor;  
}

#endif
