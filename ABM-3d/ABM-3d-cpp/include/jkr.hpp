/**
 * Functions for computing JKR contact areas and forces between contacting
 * bodies.  
 *
 * Authors:
 *     Kee-Myoung Nam
 *
 * Last updated:
 *     8/4/2025
 */

#ifndef BIOFILM_JKR_HPP
#define BIOFILM_JKR_HPP

#include <cmath>
#include <algorithm>
#include <Eigen/Dense>
#include <boost/multiprecision/mpfr.hpp>
#include "polynomials.hpp"
#include "ellipsoid.hpp"
#include "newton.hpp"

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
 * @param ellip_table Pre-computed table containing values for the elliptic
 *                    integral function for various eccentricities between 0
 *                    and 1.
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
    int nearest_idx = 0; 
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

        // Return as type T 
        return std::make_pair(
            static_cast<T>(roots_real[0]), static_cast<T>(roots_real[1])
        );
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

        // Return as type T 
        return std::make_pair(
            static_cast<T>(roots_real[0]), static_cast<T>(roots_real[1])
        );
    } 
}

/**
 * Solve for the estimated JKR contact area for an elliptical contact area,
 * according to the model given by Giudici et al., J. Phys. D (2024). 
 *
 * This function assumes that the two bodies are prolate ellipsoids whose
 * major semi-axis lengths are given by half_l1 + R and half_l2 + R, and
 * whose minor semi-axis lengths are given by R. 
 *
 * @param r1 Center of body 1. 
 * @param n1 Orientation of body 1 (long axis). 
 * @param half_l1 Half-length of body 1 centerline.
 * @param r2 Center of body 2.
 * @param n2 Orientation of body 2 (long axis). 
 * @param half_l2 Half-length of body 2 centerline. 
 * @param R Body radius. 
 * @param d12 Overlap vector. 
 * @param s Centerline coordinate along body 1 determining tail of overlap vector.  
 * @param t Centerline coordinate along body 2 determining head of overlap vector. 
 * @param E0 Elastic modulus. 
 * @param gamma Surface energy density.
 * @param max_overlap If non-negative, cap the overlap distance at this 
 *                    maximum value.
 * @param min_aspect_ratio Minimum aspect ratio of the contact area. 
 * @param max_aspect_ratio Maximum aspect ratio of the contact area.  
 * @param project_tol Tolerance for ellipsoid projection. 
 * @param project_max_iter Maximum number of iterations for ellipsoid projection.
 * @param newton_tol Tolerance for the Newton-Raphson method.  
 * @param newton_max_iter Maximum number of iterations for the Newton-Raphson
 *                        method.
 * @param verbose If true, print intermittent output to stdout.  
 * @returns Estimated JKR contact area, as well as the contact area dimensions,
 *          in terms of the equivalent radius and the aspect ratio.  
 */
template <typename T, int N = 100>
std::tuple<T, T, T> jkrContactAreaEllipsoid(const Ref<const Matrix<T, 3, 1> >& r1, 
                                            const Ref<const Matrix<T, 3, 1> >& n1,
                                            const T half_l1,  
                                            const Ref<const Matrix<T, 3, 1> >& r2, 
                                            const Ref<const Matrix<T, 3, 1> >& n2,
                                            const T half_l2, const T R,
                                            const Ref<const Matrix<T, 3, 1> >& d12,
                                            const T s, const T t, const T E0,
                                            const T gamma, const T max_overlap = -1,
                                            const T min_aspect_ratio = 0.01,
                                            const T max_aspect_ratio = 100.0,
                                            const T project_tol = 1e-6,
                                            const int project_max_iter = 100,
                                            const T newton_tol = 1e-8,
                                            const int newton_max_iter = 1000,
                                            const bool verbose = false)
{
    // Compute overlap between the two cells
    T dist = d12.norm(); 
    Matrix<T, 3, 1> d12n = d12 / dist; 
    T delta = 2 * R - dist;

    // Cap the overlap at the maximum value if given 
    if (max_overlap >= 0 && delta > max_overlap)
        delta = max_overlap;  

    // Get the principal radii of curvature at the projected contact points
    std::pair<T, T> radii1 = projectAndGetPrincipalRadiiOfCurvature<T>(
        r1, n1, half_l1, R, d12n, s, project_tol, project_max_iter
    ); 
    std::pair<T, T> radii2 = projectAndGetPrincipalRadiiOfCurvature<T>(
        r2, n2, half_l2, R, -d12n, t, project_tol, project_max_iter
    );
    T Rx1 = radii1.first; 
    T Ry1 = radii1.second; 
    T Rx2 = radii2.first; 
    T Ry2 = radii2.second;

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

    // Calculate the composite radii of curvature (A < B, so Rx > Ry)
    T Rx = 1.0 / (2 * A); 
    T Ry = 1.0 / (2 * B);

    // Calculate the curvature parameters \lambda and R (which we denote
    // by R_), and the eccentricity e
    T lambda = sqrt(Ry / Rx);
    T R_ = sqrt(Rx * Ry);  

    // Define auxiliary function for the expressions given in Giudici et al.
    // (2025)
    //
    // In particular, we specify a function that calculates \delta - F(g), 
    // where g is the aspect ratio and F(g) is the right-hand side in Eqn. 9
    std::function<T(const T)> func = [&Rx, &Ry, &lambda, &R_, &E0, &gamma, &delta](const T g) -> T
    {
        // Calculate the eccentricity 
        T e = sqrt(1 - g * g); 

        // Calculate the elliptic integrals (Eqns. 6, 7, 8) 
        T Ke = boost::math::ellint_1<T>(e); 
        T Ee = boost::math::ellint_2<T>(e);
        T De = (Ke - Ee) / (e * e); 
        T Be = Ke - De;
        T Ce = (De - Be) / (e * e);
        T g2Ce = g * g * Ce;

        // Calculate the non-dimensional parameters alpha and beta (Eqns. 10
        // and 11)
        T l2 = lambda * lambda; 
        T alpha_numer = l2 * (Be + g2Ce) + g2Ce; 
        T beta_numer = (l2 + 1) * Ce + De;
        T denom = lambda * sqrt(g) * (Be * Ce + Be * De + g2Ce * De);
        T alpha = alpha_numer / denom; 
        T beta = beta_numer / denom;
        T g2beta = g * g * beta;  

        // Calculate the equivalent contact radius (Eqns. 17 and 18)
        T sqrt_g = sqrt(g);
        T term1 = (sqrt_g - 1) * (alpha * Be + g2beta * De); 
        T term2 = Ke * (g2beta - alpha * sqrt_g); 
        T f_numer = lambda * (term1 + term2);
        T term3 = g2beta * (2 * sqrt_g + 1);  
        T term4 = alpha * (sqrt_g + 2); 
        T f_denom = 1.0 / (term3 - term4);
        T f = f_numer / f_denom; 
        T c_numer = 12 * (sqrt_g - 1) * (sqrt_g - 1) * pow(g, 2.5) * lambda;
        T term5 = (g - sqrt_g) * (pow(g, 1.5) + l2) * (g2beta - alpha);
        T c_denom = f - term5;
        T c = pow((c_numer / c_denom) * (gamma * R_ * R_ / E0), 1.0 / 3.0);  

        // Calculate the force magnitude (Eqn. 15), then multiply by
        // sqrt(g) * R / (pi * c * E0)
        T c2 = c * c; 
        T force_numer = (2 * sqrt_g + 1) * g2beta - (sqrt_g + 2) * alpha;
        T force_denom = (sqrt_g - 1) * g;
        T force = c2 * force_numer / (3 * force_denom);

        // Calculate the non-dimensional pressure (Eqn. 12) 
        T p0 = c2 * (alpha + g2beta) / (3 * g) + force;  

        // Calculate the function (Eqn. 9) 
        return delta - (p0 * Ke - 0.5 * c2 * (alpha * Be / g - g * beta * De)) / R_;  
    };

    // Compute the root of this function using the Newton-Raphson method
    auto result = newtonRaphson<T>(
        func, 1.0, 1e-8, min_aspect_ratio, max_aspect_ratio, newton_tol, 
        newton_max_iter, verbose  
    ); 
    T g = result.first; 

    // Calculate the eccentricity 
    T e = sqrt(1 - g * g); 

    // Calculate the elliptic integrals (Eqns. 6, 7, 8) 
    T Ke = boost::math::ellint_1<T>(e); 
    T Ee = boost::math::ellint_2<T>(e);
    T De = (Ke - Ee) / (e * e); 
    T Be = Ke - De;
    T Ce = (De - Be) / (e * e);
    T g2Ce = g * g * Ce;

    // Calculate the non-dimensional parameters alpha and beta (Eqns. 10 and 11)
    T l2 = lambda * lambda; 
    T alpha_numer = l2 * (Be + g2Ce) + g2Ce; 
    T beta_numer = (l2 + 1) * Ce + De;
    T denom = lambda * sqrt(g) * (Be * Ce + Be * De + g2Ce * De);
    T alpha = alpha_numer / denom; 
    T beta = beta_numer / denom;
    T g2beta = g * g * beta;  

    // Calculate the equivalent contact radius (Eqns. 17 and 18)
    T sqrt_g = sqrt(g);
    T term1 = (sqrt_g - 1) * (alpha * Be + g2beta * De); 
    T term2 = Ke * (g2beta - alpha * sqrt_g); 
    T f_numer = lambda * (term1 + term2);
    T term3 = g2beta * (2 * sqrt_g + 1);  
    T term4 = alpha * (sqrt_g + 2); 
    T f_denom = 1.0 / (term3 - term4);
    T f = f_numer / f_denom; 
    T c_numer = 12 * (sqrt_g - 1) * (sqrt_g - 1) * pow(g, 2.5) * lambda;
    T term5 = (g - sqrt_g) * (pow(g, 1.5) + l2) * (g2beta - alpha);
    T c_denom = f - term5;
    T c = pow((c_numer / c_denom) * (gamma * R_ * R_ / E0), 1.0 / 3.0);  

    // Calculate the force magnitude (Eqn. 15)
    T c3 = c * c * c; 
    T force_numer = (2 * sqrt_g + 1) * g2beta - (sqrt_g + 2) * alpha;
    T force_denom = (sqrt_g - 1) * pow(g, 1.5) * R_;
    T force = boost::math::constants::third_pi<T>() * E0 * c3 * force_numer / (3 * force_denom);

    return std::make_tuple(force, c, g); 
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
 * @param max_overlap If non-negative, cap the overlap distance at this 
 *                    maximum value. 
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
                          const T max_overlap = -1, const T project_tol = 1e-6,
                          const int project_max_iter = 100,
                          const T imag_tol = 1e-20, const T aberth_tol = 1e-20)
{
    // Compute overlap between the two cells
    T dist = d12.norm(); 
    Matrix<T, 3, 1> d12n = d12 / dist; 
    T delta = 2 * R - dist;

    // Cap the overlap at the maximum value if given 
    if (max_overlap >= 0 && delta > max_overlap)
        delta = max_overlap;  

    // Get the principal radii of curvature at the projected contact points
    std::pair<T, T> radii1 = projectAndGetPrincipalRadiiOfCurvature<T>(
        r1, n1, half_l1, R, d12n, s, project_tol, project_max_iter
    ); 
    std::pair<T, T> radii2 = projectAndGetPrincipalRadiiOfCurvature<T>(
        r2, n2, half_l2, R, -d12n, t, project_tol, project_max_iter
    );
    T Rx1 = radii1.first; 
    T Ry1 = radii1.second; 
    T Rx2 = radii2.first; 
    T Ry2 = radii2.second;
     
    // Compute the expected contact area for a Hertzian contact 
    T area = hertzContactArea<T>(delta, Rx1, Ry1, Rx2, Ry2, n1, n2, ellip_table);

    // Get the equivalent contact radius and overlap 
    T equiv_radius = sqrt(area / boost::math::constants::pi<T>());
    T equiv_overlap = equiv_radius * equiv_radius / R;  

    // Compute the correction factor to account for JKR-based adhesion
    std::pair<T, T> jkr_radius = jkrContactRadius<T, N>(
        equiv_overlap, R, E0, gamma, imag_tol, aberth_tol
    );
    T jkr_radius_factor = jkr_radius.second / equiv_radius; 

    return area * jkr_radius_factor * jkr_radius_factor;  
}

/**
 * Calculate the equilibrium cell-cell distance for the given JKR surface 
 * energy density. 
 *
 * @param R Cell radius (including the EPS). 
 * @param E0 Elastic modulus.
 * @param gamma Surface energy density.
 * @param dinit Initial overlap distance.  
 * @param increment_overlap Increment for finite differences approximation.
 * @param tol Tolerance for Newton's method.  
 * @param imag_tol Tolerance for determining whether a root for the the JKR
 *                 contact radius polynomial is real.
 * @param aberth_tol Tolerance for Aberth-Ehrlich method.
 * @returns Equilibrium cell-cell distance.  
 */
template <typename T, int N = 100>
T jkrEquilibriumDistance(const T R, const T E0, const T gamma, const T dinit, 
                         const T dmin, const T increment_overlap = 1e-8,
                         const T newton_tol = 1e-8, const int max_iter = 1000,
                         const T imag_tol = 1e-20, const T aberth_tol = 1e-20,
                         const bool verbose = false)
{
    T overlap = 2 * R - dinit;
    T max_overlap = 2 * R - dmin;   
    T update = std::numeric_limits<T>::infinity();

    // Define a function that maps the overlap distance to the JKR force 
    auto func = [&R, &E0, &gamma, &imag_tol, &aberth_tol](const T x) -> T
    {
        T radius = jkrContactRadius<T, N>(x, R, E0, gamma, imag_tol, aberth_tol).second;
        T prefactor1 = static_cast<T>(4) / static_cast<T>(3) * E0 / R;
        T prefactor2 = 4 * sqrt(boost::math::constants::pi<T>() * gamma * E0); 
        T f_hertz = prefactor1 * radius * radius * radius; 
        T f_jkr = prefactor2 * pow(radius, 1.5); 
        return f_hertz - f_jkr;
    };

    // Find the equilibrium distance using the Newton-Raphson method
    auto result = newtonRaphson<T>(
        func, overlap, increment_overlap, 0.0, max_overlap, newton_tol,
        max_iter, verbose
    ); 

    return 2 * R - result.first; 
}

/**
 * Calculate the surface energy density for which the desired Hertz-JKR 
 * equilibrium cell-cell distance is achieved.
 *
 * @param R Cell radius (including the EPS). 
 * @param E0 Elastic modulus.
 * @param deq_target Target equilibrium cell-cell distance. 
 * @param min_gamma Minimum surface energy density.
 * @param max_gamma Maximum surface energy density.  
 * @param dinit Initial overlap distance.  
 * @param tol Tolerance for (steepest) gradient descent.
 * @param log_increment_gamma Increment for finite differences approximation 
 *                            w.r.t log10(gamma) during gradient descent. 
 * @param increment_overlap Increment for finite differences approximation
 *                          w.r.t overlap in jkrEquilibriumDistance().
 * @param newton_tol Tolerance for Newton's method in jkrEquilibriumDistance().  
 * @param imag_tol Tolerance for determining whether a root for the the JKR
 *                 contact radius polynomial is real.
 * @param aberth_tol Tolerance for Aberth-Ehrlich method.
 * @param verbose If true, print iteration details to stdout.
 * @returns Optimal surface energy density for the desired equilibrium 
 *          cell-cell distance.  
 */
template <typename T, int N = 100>
T jkrOptimalSurfaceEnergyDensity(const T R, const T Rcell, const T E0,
                                 const T deq_target, const T min_gamma,
                                 const T max_gamma,
                                 const T log_increment_gamma = 1e-6,
                                 const T increment_overlap = 1e-8,
                                 const T outer_newton_tol = 1e-8,
                                 const T inner_newton_tol = 1e-8,
                                 const int outer_max_iter = 1000, 
                                 const int inner_max_iter = 1000,
                                 const T imag_tol = 1e-20,
                                 const T aberth_tol = 1e-20,
                                 const bool verbose = false)
{
    // Get an initial value for gamma
    T log_gamma = (log10(min_gamma) + log10(max_gamma)) / 2.0; 

    // Define a function that maps the log surface adhesion energy density 
    // to the JKR equilibrium distance minus the target distance  
    auto func = [
        &R, &Rcell, &E0, &deq_target, &increment_overlap, &inner_newton_tol,
        &inner_max_iter, &imag_tol, &aberth_tol, &verbose
    ](const T x) -> T
    {
        T dinit = 0.75 * 2 * R;
        T dmin = 2 * Rcell; 
        T gamma = pow(10.0, x);  
        T deq = jkrEquilibriumDistance<T>(
            R, E0, gamma, dinit, dmin, increment_overlap, inner_newton_tol,
            inner_max_iter, imag_tol, aberth_tol, false
        );
        return deq - deq_target;  
    };

    // Find the equilibrium distance using the Newton-Raphson method
    auto result = newtonRaphson<T>(
        func, log_gamma, log_increment_gamma, log10(min_gamma), log10(max_gamma),
        outer_newton_tol, outer_max_iter, verbose
    ); 

    return pow(10.0, result.first);
}

#endif
