/**
 * Functions for computing JKR contact areas and forces between contacting
 * bodies.  
 *
 * Authors:
 *     Kee-Myoung Nam
 *
 * Last updated:
 *     11/4/2025
 */

#ifndef BIOFILM_JKR_HPP
#define BIOFILM_JKR_HPP

#include <cmath>
#include <algorithm>
#include <Eigen/Dense>
#include <boost/multiprecision/mpfr.hpp>
#include "distances.hpp"
#include "polynomials.hpp"
#include "ellipsoid.hpp"
#include "rootFinding.hpp"

using namespace Eigen; 

using std::min;
using boost::multiprecision::min; 
using std::cos; 
using boost::multiprecision::cos; 
using std::acos;
using boost::multiprecision::acos;
using std::sqrt;
using boost::multiprecision::sqrt;
using std::cbrt; 
using boost::multiprecision::cbrt; 

/**
 * An enum that enumerates the different JKR force types. 
 */
enum class JKRMode
{
    ISOTROPIC = 0,
    ANISOTROPIC = 1
};

/**
 * Solve for the JKR contact radius for a circular contact area, given the 
 * overlap, equivalent radius and elastic modulus of the contact bodies,
 * and adhesion energy density.
 *
 * @param delta Overlap between the two contacting bodies. 
 * @param R Equivalent radius of the two contacting bodies.
 * @parma E Equivalent elastic modulus of the two contacting bodies. 
 * @param gamma Surface adhesion energy density. 
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
 * Compute the elliptic integrals K(e), E(e), D(e), B(e), and C(e) defined 
 * by Giudici et al. J. Phys. D (2025) [see also Johnson & Greenwood (2005)]. 
 *
 * @param e Eccentricity.
 * @param tol Tolerance for low eccentricity.  
 * @returns Values of the five elliptic integrals. 
 */
template <typename T>
std::tuple<T, T, T, T, T> jkrEllipticIntegrals(const T e, const T tol = 1e-8)
{
    // If the eccentricity is close to zero, calculate the elliptic integrals 
    // via their limits at zero
    T Ke, Ee, De, Be, Ce; 
    if (e < tol)
    {
        Ke = boost::math::constants::half_pi<T>();
        Ee = boost::math::constants::half_pi<T>(); 
        De = boost::math::constants::quarter_pi<T>(); 
        Be = boost::math::constants::quarter_pi<T>(); 
        Ce = boost::math::constants::quarter_pi<T>() / 4; 
    }
    else    // Otherwise, compute them directly 
    {
        T e2 = e * e;  
        Ke = boost::math::ellint_1<T>(e); 
        Ee = boost::math::ellint_2<T>(e);
        De = (Ke - Ee) / e2;
        Be = Ke - De;
        Ce = (De - Be) / e2;
    }

    return std::make_tuple(Ke, Ee, De, Be, Ce); 
}

/**
 * Compute the dimensionless pressure coefficients \alpha and \beta in Giudici
 * et al. J. Phys. D (2025) [see also Johnson & Greenwood (2005)]. 
 *
 * @param g Slenderness of the contact area, b / a.
 * @param lambda Square root of the ratio of the principal radii of curvature. 
 * @param Be, Ce, De Composite elliptic integrals, as defined in Giudici 
 *                   et al. (2025).
 * @returns Corresponding values of \alpha and \beta.  
 */
template <typename T>
std::pair<T, T> jkrAlphaBeta(const T g, const T lambda, const T Be, const T Ce, 
                             const T De)
{
    T g2Ce = g * g * Ce; 
    T l2 = lambda * lambda;
    T alpha_numer = l2 * (Be + g2Ce) + g2Ce; 
    T beta_numer = (l2 + 1) * Ce + De;
    T denom = lambda * (Be * Ce + Be * De + g2Ce * De); 
    T alpha = alpha_numer / denom; 
    T beta = beta_numer / denom;

    return std::make_pair(alpha, beta); 
}

/**
 * Compute the JKR contact radius, c, as defined by Giudici et al. J. Phys. D
 * (2025) (see Eqns. 17 and 18).
 *
 * @param g Slenderness of the contact area, b / a. 
 * @param lambda Square root of the ratio of the principal radii of curvature.
 * @param R Geometric mean of the principal radii of curvature. 
 * @param gamma Surface adhesion energy density. 
 * @param E Elastic modulus. 
 * @param Ke Complete elliptic integral of the first kind, evaluated at the 
 *           eccentricity corresponding to the given slenderness (g). 
 * @param Be, De Composite elliptic integrals, as defined in Giudici et al. 
 *               (2025). 
 * @param alpha, beta Dimensionless pressure coefficients, as defined in 
 *                    Giudici et al. (2025).
 * @returns Corresponding JKR contact radius.  
 */
template <typename T>
T jkrContactRadiusElliptical(const T g, const T lambda, const T R, const T gamma,
                             const T E, const T Ke, const T Be, const T De,
                             const T alpha, const T beta)
{
    T g2beta = g * g * beta;
    T l2 = lambda * lambda; 
    T g12 = sqrt(g); 
    T g32 = g12 * g; 
    T g52 = g32 * g; 
    T f_numer = lambda * ((g12 - 1) * (alpha * Be + g2beta * De) + Ke * (g2beta - alpha * g12)); 
    T f_denom = g2beta * (2 * g12 + 1) - alpha * (g12 + 2); 
    T f = f_numer * f_denom;
    T dgamma = 2 * gamma;    // Surface energy conversion  
    T c_numer = 12 * dgamma * R * R * pow(g12 - 1, 2) * g52 * lambda; 
    T c_denom = E * (f - (g - g12) * (g32 + l2) * (g2beta - alpha));
    T c = pow(c_numer / c_denom, 1. / 3.);

    return c;   
}

/**
 * Compute the JKR force, W, as defined by Giudici et al. J. Phys. D (2025)
 * (see Eqn. 15).
 *
 * @param g Slenderness of the contact area, b / a. 
 * @param R Geometric mean of the principal radii of curvature. 
 * @param E Elastic modulus. 
 * @param alpha, beta Dimensionless pressure coefficients, as defined in 
 *                    Giudici et al. (2025).
 * @param c JKR contact radius. 
 * @returns Corresponding JKR contact force magnitude. 
 */
template <typename T>
T jkrContactForceElliptical(const T g, const T R, const T E, const T alpha,
                            const T beta, const T c)
{
    T c3 = c * c * c;
    T g12 = sqrt(g);
    T g32 = g12 * g; 
    T numer = (2 * g12 + 1) * g * g * beta - (g12 + 2) * alpha; 
    T denom = (g12 - 1) * g32 * R; 
    T force = boost::math::constants::third_pi<T>() * E * c3 * numer / denom;

    return force;  
} 

/**
 * Compute the non-dimensional pressure, p_0, as defined by Giudici et al. 
 * J. Phys. D (2025) [see Eqn. 12; see also Johnson & Greenwood (2005)].
 *
 * @param g Slenderness of the contact area, b / a. 
 * @param R Geometric mean of the principal radii of curvature. 
 * @param E Elastic modulus. 
 * @param alpha, beta Dimensionless pressure coefficients, as defined in 
 *                    Giudici et al. (2025).
 * @param c JKR contact radius.
 * @param force JKR contact force magnitude.  
 * @returns Corresponding non-dimensional pressure. 
 */
template <typename T>
T jkrPressure(const T g, const T R, const T E, const T alpha, const T beta,
              const T c, const T force)
{
    T g2beta = g * g * beta; 
    T term1 = c * c * (alpha + g2beta) / (3 * g); 
    T term2 = sqrt(g) * R * force / (boost::math::constants::pi<T>() * c * E); 

    return term1 + term2; 
}

/**
 * Compute the overlap distance, \delta, corresponding to the aspect ratio,
 * g, of the JKR contact region, using the expressions given by Giudici 
 * et al. (2025).
 *
 * The aspect ratio is assumed to be less than 1. 
 *
 * @param lambda Square root of the ratio of the equivalent radii of curvature
 *               from the two bodies. 
 * @param R Geometric mean of the equivalent radii of curvature from the two
 *          bodies. 
 * @param E0 Elastic modulus. 
 * @param gamma Surface adhesion energy density. 
 * @param g Contact region aspect ratio. Assumed to be less than 1.  
 * @returns Corresponding overlap distance. 
 */
template <typename T>
T jkrOverlapFromAspectRatio(const T lambda, const T R, const T E0, const T gamma,
                            const T g)
{
    // Calculate the eccentricity
    T e = sqrt(1 - g * g); 

    // Calculate the elliptic integrals (Eqns. 6, 7, 8) 
    auto ell = jkrEllipticIntegrals<T>(e); 
    T Ke = std::get<0>(ell); 
    T Ee = std::get<1>(ell); 
    T De = std::get<2>(ell);
    T Be = std::get<3>(ell); 
    T Ce = std::get<4>(ell); 

    // Calculate the non-dimensional parameters alpha and beta (Eqns. 10
    // and 11)
    auto ab = jkrAlphaBeta<T>(g, lambda, Be, Ce, De);
    T alpha = ab.first;  
    T beta = ab.second; 

    // Calculate the equivalent contact radius (Eqns. 17 and 18)
    T c = jkrContactRadiusElliptical<T>(g, lambda, R, gamma, E0, Ke, Be, De, alpha, beta); 

    // Calculate the force magnitude (Eqn. 15)
    T force = jkrContactForceElliptical<T>(g, R, E0, alpha, beta, c); 

    // Calculate the non-dimensional pressure (Eqn. 12) 
    T p0 = jkrPressure<T>(g, R, E0, alpha, beta, c, force);

    // Calculate the function (Eqn. 9) 
    return (p0 * Ke - 0.5 * c * c * ((alpha * Be / g) + (g * beta * De))) / R;
}

/**
 * Solve for the JKR contact area aspect ratio that matches the given overlap
 * distance and input parameters, according to the model given by Giudici
 * et al., J. Phys. D (2025).
 *
 * @param Rx Larger equivalent principal radius of curvature at the contact
 *           point.  
 * @param Ry Smaller equivalent principal radius of curvature at the contact
 *           point.
 * @param E0 Elastic modulus. 
 * @param gamma Surface adhesion energy density.
 * @param delta Overlap distance. 
 * @param min_aspect_ratio Minimum aspect ratio.
 * @param max_aspect_ratio Maximum aspect ratio for anisotropic contacts.  
 * @param brent_tol Tolerance for Brent's method. 
 * @param brent_max_iter Maximum number of iterations for Brent's method.
 * @param init_bracket_dx Increment for bracket initialization. 
 * @param n_tries_bracket Number of attempts for bracket initialization. 
 * @param verbose If true, print intermittent output to stdout.  
 * @returns JKR contact area aspect ratio. 
 */
template <typename T>
T jkrAspectRatioFromOverlap(const T Rx, const T Ry, const T E0, const T gamma,
                            const T delta, const T min_aspect_ratio = 0.001, 
                            const T max_aspect_ratio = 0.999,
                            const T brent_tol = 1e-8,
                            const int brent_max_iter = 1000,
                            const T init_bracket_dx = 1e-3,
                            const int n_tries_bracket = 5,
                            const bool verbose = false)
{
    // If Rx == Ry, return g = 1
    if (abs(Rx - Ry) < 1e-8)
        return 1; 

    // Calculate the curvature parameters \lambda and R
    T lambda = sqrt(Ry / Rx); 
    T R = sqrt(Rx * Ry); 

    // Compute the root of the overlap vs. aspect ratio function using 
    // Brent's method
    //
    // Initialize the function whose root is to be calculated
    T g = max_aspect_ratio; 
    auto func = [&lambda, &R, &E0, &gamma, &delta](const T g) -> T
    {
        return delta - jkrOverlapFromAspectRatio<T>(lambda, R, E0, gamma, g); 
    };

    // Identify an initial bracket
    std::pair<T, T> bracket;
    int i = 0;
    T bracket_dx = init_bracket_dx;
    bool found_bracket = false;  
    while (i < n_tries_bracket)
    {
        try
        { 
            bracket = findBracket<T>(
                func, min_aspect_ratio, max_aspect_ratio, bracket_dx
            );
            found_bracket = true; 
        }
        catch (const std::runtime_error& e)
        {
            bracket_dx /= 2.0;
            i++;  
        }
        if (found_bracket)
        {
            break; 
        }
    }
    // If a bracket could not be found ... 
    if (!found_bracket)
    {
        // First calculate the function using the finest increment
        int n = static_cast<int>((max_aspect_ratio - min_aspect_ratio) / bracket_dx);  
        Array<T, Dynamic, 1> xmesh = Array<T, Dynamic, 1>::LinSpaced(
            n, min_aspect_ratio, max_aspect_ratio
        ); 
        Array<T, Dynamic, 1> fmesh(n); 
        for (int j = 0; j < n; ++j)
            fmesh(j) = func(xmesh(j));

        // If there are branches along the function due to infinities
        // or NaN's, as expected, refine the range of x-values
        if (fmesh.isNaN().any() || fmesh.isInf().any())
        {
            int max_idx = 0; 
            for (int j = 0; j < n; ++j)
            {
                if (isnan(fmesh(j)) || isinf(fmesh(j)))
                {
                    max_idx = j; 
                }
            }
            T new_xmin = xmesh(max_idx + 1); 
            xmesh = Array<T, Dynamic, 1>::LinSpaced(n, new_xmin, max_aspect_ratio); 
            for (int j = 0; j < n; ++j)
                fmesh(j) = func(xmesh(j));
        }

        // Look for the root of the function
        Index root_idx; 
        fmesh.abs().minCoeff(&root_idx);
        g = xmesh(root_idx);  
    }
    // Otherwise, use Brent's method 
    else 
    {
        T xmin = bracket.first; 
        T xmax = bracket.second; 
        auto result = brent<T>(
            func, xmin, xmax, brent_tol, brent_max_iter, verbose
        ); 
        g = (result.first + result.second) / 2;
    }

    return g;  
}

/**
 * Solve for the JKR contact area and force for an ellipsoidal surface JKR
 * contact, according to the model given by Giudici et al., J. Phys. D (2025).
 *
 * @param Rx Larger equivalent principal radius of curvature at the contact
 *           point.  
 * @param Ry Smaller equivalent principal radius of curvature at the contact
 *           point.
 * @param E0 Elastic modulus. 
 * @param gamma Surface adhesion energy density.
 * @param delta Overlap distance.
 * @param min_aspect_ratio Minimum aspect ratio.
 * @param max_aspect_ratio Maximum aspect ratio for anisotropic contacts.  
 * @param brent_tol Tolerance for Brent's method. 
 * @param brent_max_iter Maximum number of iterations for Brent's method.
 * @param init_bracket_dx Increment for bracket initialization. 
 * @param n_tries_bracket Number of attempts for bracket initialization.
 * @param imag_tol Tolerance for determining whether a root for the the JKR
 *                 contact radius polynomial is real.
 * @param aberth_tol Tolerance for Aberth-Ehrlich method.
 * @param verbose If true, print intermittent output to stdout.  
 */
template <typename T, int N = 100>
std::tuple<T, T, T> jkrContactAreaAndForceEllipsoidToSurface(const T Rx,
                                                             const T Ry, 
                                                             const T E0, 
                                                             const T gamma, 
                                                             const T delta, 
                                                             const T min_aspect_ratio = 0.001, 
                                                             const T max_aspect_ratio = 0.999,
                                                             const T brent_tol = 1e-8,
                                                             const int brent_max_iter = 1000,
                                                             const T init_bracket_dx = 1e-3,
                                                             const int n_tries_bracket = 5,
                                                             const T imag_tol = 1e-20, 
                                                             const T aberth_tol = 1e-20, 
                                                             const bool verbose = false)
{
    // Calculate the aspect ratio 
    T g = jkrAspectRatioFromOverlap<T>(
        Rx, Ry, E0, gamma, delta, min_aspect_ratio, max_aspect_ratio, 
        brent_tol, brent_max_iter, init_bracket_dx, n_tries_bracket, verbose
    );

    // If the aspect ratio is 1, use jkrContactRadius(), as the elliptic 
    // integrals are ill-defined
    T force, c;
    if (g == 1)     // Rx == Ry
    {
        // Use the equivalent radii of curvature (which are roughly equal)
        auto result = jkrContactRadius<T, N>(
            delta, Rx, E0, gamma, imag_tol, aberth_tol
        );
        c = result.second;
        T c3 = c * c * c;
        T force1 = (4 * E0 * c3) / (3 * Rx); 
        T force2 = 4 * sqrt(boost::math::constants::pi<T>() * c3 * gamma * E0);
        force = force1 - force2;    // Repulsive force minus attractive force  
    }
    else    // Otherwise ... 
    {
        // Calculate the eccentricity
        T e = sqrt(1 - g * g); 
        T lambda = sqrt(Ry / Rx); 
        T R_ = sqrt(Rx * Ry); 

        // Calculate the elliptic integrals (Eqns. 6, 7, 8) 
        auto ell = jkrEllipticIntegrals<T>(e); 
        T Ke = std::get<0>(ell); 
        T Ee = std::get<1>(ell); 
        T De = std::get<2>(ell);
        T Be = std::get<3>(ell); 
        T Ce = std::get<4>(ell); 

        // Calculate the non-dimensional parameters alpha and beta (Eqns. 10 and 11)
        auto ab = jkrAlphaBeta<T>(g, lambda, Be, Ce, De);
        T alpha = ab.first;  
        T beta = ab.second; 

        // Calculate the equivalent contact radius (Eqns. 17 and 18)
        c = jkrContactRadiusElliptical<T>(g, lambda, R, gamma, E0, Ke, Be, De, alpha, beta); 

        // Calculate the force magnitude (Eqn. 15)
        force = jkrContactForceElliptical<T>(g, R, E0, alpha, beta, c); 
    }

    return std::make_tuple(force, c, g); 
}

/**
 * Solve for the JKR contact area and force for an ellipsoidal surface JKR
 * contact, according to the approximate relations given by Li and Popov, 
 * J. Phys. D (2020).
 *
 * @param Rx Larger equivalent principal radius of curvature at the contact
 *           point.  
 * @param Ry Smaller equivalent principal radius of curvature at the contact
 *           point.
 * @param E0 Elastic modulus. 
 * @param gamma Surface adhesion energy density.
 * @param delta Overlap distance.
 * @param imag_tol Tolerance for determining whether a root for the the JKR
 *                 contact radius polynomial is real.
 * @param aberth_tol Tolerance for Aberth-Ehrlich method.
 * @param verbose If true, print intermittent output to stdout.  
 */
template <typename T, int N = 100>
std::pair<T, T> jkrContactAreaAndForceEllipsoidToSurfaceApprox(const T Rx,
                                                               const T Ry, 
                                                               const T E0, 
                                                               const T gamma, 
                                                               const T delta, 
                                                               const T imag_tol = 1e-20, 
                                                               const T aberth_tol = 1e-20, 
                                                               const bool verbose = false)
{
    typedef boost::multiprecision::number<boost::multiprecision::mpfr_float_backend<N> >  RealType;
    typedef boost::multiprecision::number<boost::multiprecision::mpc_complex_backend<N> > ComplexType;

    // Non-dimensionalize the overlap distance 
    const T prefactor1 = 4 * E0 / (9 * boost::math::constants::pi<T>() * gamma);
    const T R = sqrt(Rx * Ry); 
    T delta_nd = delta * pow(prefactor1 / sqrt(R), static_cast<T>(2) / static_cast<T>(3));

    // Calculate the dimensionless contact radius
    //
    // Set up the quartic equation ... 
    Matrix<RealType, Dynamic, 1> coefs(5);
    RealType delta_nd_ = static_cast<RealType>(delta_nd);
    RealType ratio = static_cast<RealType>(Rx / Ry);    // Should be >= 1
    RealType A = static_cast<RealType>(0.05548) * pow(ratio, static_cast<RealType>(0.3103));
    A += static_cast<RealType>(0.9323); 
    RealType B = static_cast<RealType>(0.00658) * pow(ratio, static_cast<RealType>(0.5779)); 
    B -= static_cast<RealType>(0.9439);  
    coefs << delta_nd_ * delta_nd_, -B * B, -2 * A * delta_nd_, 0, A * A;

    // Solve for the roots of the polynomial with the specified precision
    HighPrecisionPolynomial<N> p(coefs);
    Matrix<ComplexType, Dynamic, 1> roots = p.solveAberth(static_cast<RealType>(aberth_tol));

    // Find the largest real root
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
    T c_nd = static_cast<T>(std::max_element(roots_real.begin(), roots_real.end()));

    // Dimensionalize the contact radius 
    T c = c_nd * cbrt(R * R / prefactor);

    // Calculate the dimensionless force
    T c1 = 0.1467 * pow(ratio, 0.4872) + 0.7893; 
    T c2 = -0.1188 * pow(ratio, 0.474) - 1.23; 
    T alpha = 0.1741 * pow(ratio, 0.2446) + 1.281;  
    T force_nd = c1 * c_nd * c_nd * c_nd + c2 * pow(c_nd, alpha); 

    // Dimensionalize the force 
    T force = 3 * boost::math::constants::pi<T>() * R * gamma * force_nd;

    return std::make_pair(force, c); 
}

/**
 * Solve for the JKR contact area and force for an elliptical JKR contact, 
 * according to the model given by Giudici et al., J. Phys. D (2025). 
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
 * @param gamma Surface adhesion energy density.
 * @param max_overlap If non-negative, cap the overlap distance at this 
 *                    maximum value.
 * @param calibrate_endpoint_radii If true, calibrate the principal radii of
 *                                 curvature so that its minimum value is R. 
 * @param min_aspect_ratio Minimum aspect ratio of the contact area. 
 * @param project_tol Tolerance for ellipsoid projection. 
 * @param project_max_iter Maximum number of iterations for ellipsoid projection.
 * @param brent_tol Tolerance for Brent's method. 
 * @param brent_max_iter Maximum number of iterations for Brent's method.
 * @param init_bracket_dx Increment for bracket initialization. 
 * @param n_tries_bracket Number of attempts for bracket initialization. 
 * @param imag_tol Tolerance for determining whether a root for the the JKR
 *                 contact radius polynomial is real.
 * @param aberth_tol Tolerance for Aberth-Ehrlich method.
 * @param verbose If true, print intermittent output to stdout.  
 * @returns Estimated JKR force, as well as the contact area dimensions, in
 *          terms of the equivalent radius and the aspect ratio.  
 */
template <typename T, int N = 100>
std::tuple<T, T, T> jkrContactAreaAndForceEllipsoid(const Ref<const Matrix<T, 3, 1> >& r1, 
                                                    const Ref<const Matrix<T, 3, 1> >& n1,
                                                    const T half_l1,  
                                                    const Ref<const Matrix<T, 3, 1> >& r2, 
                                                    const Ref<const Matrix<T, 3, 1> >& n2,
                                                    const T half_l2, const T R,
                                                    const Ref<const Matrix<T, 3, 1> >& d12,
                                                    const T s, const T t, const T E0,
                                                    const T gamma, const T max_overlap = -1,
                                                    const bool calibrate_endpoint_radii = true, 
                                                    const T min_aspect_ratio = 0.01,
                                                    const T max_aspect_ratio = 0.999, 
                                                    const T project_tol = 1e-6,
                                                    const int project_max_iter = 100,
                                                    const T brent_tol = 1e-8, 
                                                    const int brent_max_iter = 1000,
                                                    const T init_bracket_dx = 1e-3,
                                                    const int n_tries_bracket = 5,
                                                    const T imag_tol = 1e-20, 
                                                    const T aberth_tol = 1e-20, 
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
        n1, half_l1, R, d12n, s, project_tol, project_max_iter
    ); 
    std::pair<T, T> radii2 = projectAndGetPrincipalRadiiOfCurvature<T>(
        n2, half_l2, R, -d12n, t, project_tol, project_max_iter
    );
    T Rx1 = radii1.first; 
    T Ry1 = radii1.second; 
    T Rx2 = radii2.first; 
    T Ry2 = radii2.second;

    // If desired, calibrate the radii of curvature so that they match that 
    // of a sphere at the endpoints of the ellipsoid 
    if (calibrate_endpoint_radii)
    {
        T factor1 = (half_l1 + R) / R;   // = R / (R * R / (half_l1 + R))
        T factor2 = (half_l2 + R) / R;   // = R / (R * R / (half_l2 + R))
        Rx1 *= factor1; 
        Ry1 *= factor1; 
        Rx2 *= factor2; 
        Ry2 *= factor2; 
    }

    // First calculate B and A, with the added assumption that B > A 
    T sum = 0.5 * (1.0 / Rx1 + 1.0 / Ry1 + 1.0 / Rx2 + 1.0 / Ry2);
    T phi = acosSafe<T>(n1.dot(n2));
    if (phi > boost::math::constants::half_pi<T>())
        phi = boost::math::constants::pi<T>() - phi; 
    T delta1 = (1.0 / Rx1 - 1.0 / Ry1); 
    T delta2 = (1.0 / Rx2 - 1.0 / Ry2); 
    T diff = 0.5 * sqrt(
        delta1 * delta1 + delta2 * delta2 + 2 * delta1 * delta2 * cos(2 * phi)
    );
    T B = 0.5 * (sum + diff);
    T A = sum - B;

    // Calculate the equivalent radii of curvature (A < B, so Rx > Ry)
    T Rx = 1.0 / (2 * A); 
    T Ry = 1.0 / (2 * B);

    // Solve the equivalent surface contact problem 
    return jkrContactAreaAndForceEllipsoidToSurface<T>(
        Rx, Ry, E0, gamma, delta, min_aspect_ratio, max_aspect_ratio, 
        brent_tol, brent_max_iter, init_bracket_dx, n_tries_bracket, imag_tol,
        aberth_tol, verbose
    ); 
}

/**
 * Solve for the estimated JKR contact area for an elliptical contact area,
 * according to the model given by Giudici et al., J. Phys. D (2025). 
 *
 * This is a version of the previous function that takes the equivalent
 * principal radii of curvature at the contact point and the overlap distance. 
 *
 * @param Rx Larger equivalent principal radius of curvature at the contact
 *           point.  
 * @param Ry Smaller equivalent principal radius of curvature at the contact
 *           point. 
 * @param delta Overlap distance. 
 * @param E0 Elastic modulus. 
 * @param gamma Surface adhesion energy density.
 * @param max_overlap If non-negative, cap the overlap distance at this 
 *                    maximum value.
 * @param min_aspect_ratio Minimum aspect ratio.
 * @param max_aspect_ratio Maximum aspect ratio for anisotropic contacts.  
 * @param brent_tol Tolerance for Brent's method. 
 * @param brent_max_iter Maximum number of iterations for Brent's method.
 * @param init_bracket_dx Increment for bracket initialization. 
 * @param n_tries_bracket Number of attempts for bracket initialization. 
 * @param imag_tol Tolerance for determining whether a root for the the JKR
 *                 contact radius polynomial is real.
 * @param aberth_tol Tolerance for Aberth-Ehrlich method.
 * @param verbose If true, print intermittent output to stdout.  
 * @returns Estimated JKR force, as well as the contact area dimensions, in
 *          terms of the equivalent radius and the aspect ratio.  
 */
template <typename T, int N = 100>
std::tuple<T, T, T> jkrContactAreaAndForceEllipsoid(const T Rx, const T Ry,
                                                    const T delta, const T E0,
                                                    const T gamma, const T max_overlap = -1,
                                                    const T min_aspect_ratio = 0.01,
                                                    const T max_aspect_ratio = 0.999,
                                                    const T brent_tol = 1e-8,
                                                    const int brent_max_iter = 1000,
                                                    const T init_bracket_dx = 1e-3,
                                                    const int n_tries_bracket = 5,
                                                    const T imag_tol = 1e-20, 
                                                    const T aberth_tol = 1e-20, 
                                                    const bool verbose = false)
{
    // Cap the overlap at the maximum value if given
    T delta_ = delta;  
    if (max_overlap >= 0 && delta > max_overlap)
        delta_ = max_overlap;

    // Solve the equivalent surface contact problem 
    return jkrContactAreaAndForceEllipsoidToSurface<T>(
        Rx, Ry, E0, gamma, delta_, min_aspect_ratio, max_aspect_ratio, 
        brent_tol, brent_max_iter, init_bracket_dx, n_tries_bracket, imag_tol,
        aberth_tol, verbose
    ); 
}

/**
 * Calculate the equilibrium cell-cell distance for the given JKR surface 
 * energy density. 
 *
 * @param R Cell radius (including the EPS). 
 * @param E0 Elastic modulus.
 * @param gamma Surface adhesion energy density.
 * @param dinit Initial cell-cell distance (from centerline to centerline). 
 * @param dmin Minimum cell-cell distance (from centerline to centerline),
 *             which determines the maximum overlap distance.   
 * @param increment_overlap Increment for finite differences approximation.
 * @param newton_tol Tolerance for the Newton-Raphson method.  
 * @param newton_max_iter Maximum number of iterations for the Newton-Raphson
 *                        method.
 * @param imag_tol Tolerance for determining whether a root for the the JKR
 *                 contact radius polynomial is real.
 * @param aberth_tol Tolerance for Aberth-Ehrlich method.
 * @param verbose If true, print intermittent output to stdout.  
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
        T Req = R / 2; 
        T radius = jkrContactRadius<T, N>(x, Req, E0, gamma, imag_tol, aberth_tol).second;
        T prefactor1 = static_cast<T>(4) / static_cast<T>(3) * E0 / Req;
        T prefactor2 = 4 * sqrt(boost::math::constants::pi<T>() * gamma * E0); 
        T f_hertz = prefactor1 * radius * radius * radius; 
        T f_jkr = prefactor2 * pow(radius, 1.5); 
        return f_hertz - f_jkr;
    };

    // Find the equilibrium distance using the Newton-Raphson method
    auto result = newtonRaphson<T>(
        func, overlap, increment_overlap, newton_tol, max_iter, verbose
    );
    T delta = min(result.first, max_overlap); 

    return 2 * R - delta; 
}

/**
 * Calculate the surface adhesion energy density for which the desired JKR
 * equilibrium cell-cell distance is achieved.
 *
 * @param R Cell radius (including the EPS). 
 * @param E0 Elastic modulus.
 * @param deq_target Target equilibrium cell-cell distance. 
 * @param init_gamma Initial value for the surface adhesion energy density.
 * @param log_increment_gamma Increment for finite differences approximation 
 *                            w.r.t log10(gamma) during gradient descent. 
 * @param increment_overlap Increment for finite differences approximation
 *                          w.r.t overlap in jkrEquilibriumDistance().
 * @param outer_newton_tol Tolerance for Newton-Raphson method in the 
 *                         function proper. 
 * @param inner_newton_tol Tolerance for Newton-Raphson method in each inner
 *                         call to jkrEquilibriumDistance().
 * @param outer_max_iter Maximum number of iterations for Newton-Raphson
 *                       method in the function proper. 
 * @param inner_max_iter Maximum number of iterations for Newton-Raphson
 *                       method in each inner call to jkrEquilibriumDistance(). 
 * @param imag_tol Tolerance for determining whether a root for the the JKR
 *                 contact radius polynomial is real.
 * @param aberth_tol Tolerance for Aberth-Ehrlich method.
 * @param verbose If true, print iteration details to stdout.
 * @returns Optimal surface energy density for the desired equilibrium 
 *          cell-cell distance.  
 */
template <typename T, int N = 100>
T jkrOptimalSurfaceEnergyDensity(const T R, const T Rcell, const T E0,
                                 const T deq_target, const T init_gamma = 10.0,
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
    T log_gamma = log10(init_gamma); 

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
        func, log_gamma, log_increment_gamma, outer_newton_tol, outer_max_iter,
        verbose
    ); 

    return pow(10.0, result.first);
}

#endif
