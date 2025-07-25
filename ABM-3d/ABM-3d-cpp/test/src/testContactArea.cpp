/**
 * Test module for contact area functions in `jkr.hpp`.
 *
 * Authors:
 *     Kee-Myoung Nam
 *
 * Last updated:
 *     6/4/2025
 */
#include <iostream>
#include <cmath>
#include <functional>
#include <Eigen/Dense>
#include <boost/math/constants/constants.hpp>
#include <boost/math/special_functions/ellint_1.hpp>
#include <boost/math/special_functions/ellint_2.hpp>
#include <boost/multiprecision/mpfr.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "../../include/ellipsoid.hpp"
#include "../../include/jkr.hpp"

using namespace Eigen; 

// Use high-precision type for testing 
typedef boost::multiprecision::number<boost::multiprecision::mpfr_float_backend<100> > T; 

using std::abs; 
using boost::multiprecision::abs; 
using std::sqrt; 
using boost::multiprecision::sqrt;

/* ------------------------------------------------------------------- //
 *                             TEST MODULES                            //
 * ------------------------------------------------------------------- */
/**
 * Test module for hertzContactArea(). 
 */
TEST_CASE("Tests for Hertzian contact area calculation", "[hertzContactArea()]")
{
    // Use slightly larger tolerances, since these calculations are based
    // on tables 
    const double B_over_A_tol = 1e-3;
    const double area_tol = 1e-6;

    // Case 1: Example on page 37 of Barber
    //
    // In this case, the total force is prescribed (20 newtons), so we must 
    // back-calculate the rigid-body approach
    T Rx1 = 1e-2;    // 10 mm
    T Ry1 = 1e-2;    // 10 mm
    T Rx2 = 1e+9;    // Large value
    T Ry2 = 0.025;   // 25 mm
    
    // Calculate the contact area dimensions, as stipulated in the example
    // (Eq. 3.32 and 3.33)
    T A = 50.0;      // 50 inverse meters
    T B = 70.0;      // 70 inverse meters
    T e = 0.6011;    // Eccentricity calculated using Maple
    T E = 115.4e+9;  // Elastic modulus of 115.4 GPa
    T Ke = boost::math::ellint_1<T>(e); 
    T Ee = boost::math::ellint_2<T>(e);  
    T a = pow(3 * 20 * (Ke - Ee) / (boost::math::constants::two_pi<T>() * e * e * E * A), 1.0 / 3.0); 
    T b = a * sqrt(1 - e * e); 
    
    // Calculate the maximum contact pressure from the total force and 
    // contact dimensions (Eq. 3.34)
    T p0 = 3 * 20 / (boost::math::constants::two_pi<T>() * a * b); 

    // Calculate the rigid-body approach from the total force (Eq. 3.27)
    T delta = p0 * b * Ke / E;

    // Now, calculate the contact area via hertzContactArea()
    Matrix<T, 3, 1> n1, n2;    // The two orientations should not matter
    n1 << 1, 0, 0; 
    n2 << 1, 0, 0;
    int ntable = 1000; 
    Matrix<T, Dynamic, 4> ellip_table = getEllipticIntegralTable<T>(ntable); 
    T area = hertzContactArea<T>(delta, Rx1, Ry1, Rx2, Ry2, n1, n2, ellip_table);

    // First check that the elliptic integral table closely approximates the
    // value of B / A
    Index minidx; 
    (ellip_table.col(0) - e * Matrix<T, Dynamic, 1>::Ones(ntable)).array().abs().minCoeff(&minidx);
    REQUIRE_THAT(
        static_cast<double>(ellip_table(minidx, 3)),
        Catch::Matchers::WithinAbs(static_cast<double>(B / A), B_over_A_tol)
    );

    // Then check the contact area 
    REQUIRE_THAT(
        static_cast<double>(area),
        Catch::Matchers::WithinAbs(
            static_cast<double>(boost::math::constants::pi<T>() * a * b), area_tol
        )
    );
}

/**
 * Test module for jkrContactRadius(). 
 */
TEST_CASE("Tests for JKR contact radius calculations", "[jkrContactRadius()]")
{
    typedef boost::multiprecision::number<boost::multiprecision::mpc_complex_backend<100> > ComplexType;
    const T tol = 1e-15;  

    T delta = 0.1;
    T R = 0.8; 
    T E = 3900.0; 
    T gamma = 100.0;
    T imag_tol = 1e-20;
    T aberth_tol = 1e-20; 
    std::pair<T, T> radii = jkrContactRadius<T, 100>(delta, R, E, gamma, imag_tol, aberth_tol);
    T r1 = radii.first; 
    T r2 = radii.second;

    // Check that both radii are positive
    REQUIRE(r1 > tol);
    REQUIRE(r2 > tol);

    // Check that the larger radius is greater than the Hertzian expectation
    REQUIRE(r2 - sqrt(R * delta) > tol);

    // Check that the polynomial evaluates to zero at both computed radii
    T c0 = R * R * delta * delta; 
    T c1 = -4 * boost::math::constants::pi<T>() * gamma * R * R / E;
    T c2 = -2 * R * delta; 
    T c4 = 1.0;
    Matrix<T, Dynamic, 1> coefs(5); 
    coefs << c0, c1, c2, 0.0, c4; 
    HighPrecisionPolynomial<100> p(coefs);
    ComplexType z1(r1, 0.0);
    ComplexType z2(r2, 0.0);  
    REQUIRE(abs(p.eval(z1)) < tol);
    REQUIRE(abs(p.eval(z2)) < tol);

    // Compare against the analytical formula derived by Parteli et al. (2014)
    T P = -c2 * c2 / 12.0 - c0; 
    T Q = -c2 * c2 * c2 / 108.0 + c0 * c2 / 3.0 - c1 * c1 / 8.0; 
    T U = pow(-Q / 2.0 + sqrt(Q * Q / 4.0 + P * P * P / 27.0), 1. / 3.);
    T s = -5 * c2 / 6.0; 
    if (P != 0)
        s += (U - P / (3 * U));
    else 
        s -= pow(Q, 1. / 3.);
    T w = sqrt(c2 + 2 * s); 
    T lambda = c1 / (2 * w);
    T a1 = 0.5 * (w + sqrt(w * w - 4 * (c2 + s + lambda)));
    T a2 = 0.5 * (w - sqrt(w * w - 4 * (c2 + s + lambda))); 
    REQUIRE_THAT(
        static_cast<double>(r2),
        Catch::Matchers::WithinAbs(static_cast<double>(a1), static_cast<double>(tol))
    );
    REQUIRE_THAT(
        static_cast<double>(r1),
        Catch::Matchers::WithinAbs(static_cast<double>(a2), static_cast<double>(tol))
    );
    ComplexType y1(a1, 0.0);
    ComplexType y2(a2, 0.0);  
    REQUIRE(abs(p.eval(y1)) < tol);
    REQUIRE(abs(p.eval(y2)) < tol);
}
