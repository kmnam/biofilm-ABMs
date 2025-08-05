/**
 * Test module for JKR contact force calculations. 
 *
 * Authors:
 *     Kee-Myoung Nam
 *
 * Last updated:
 *     8/4/2025
 */
#include <iostream>
#include <fstream>
#include <string>
#include <iomanip>
#include <utility>
#include <Eigen/Dense>
#include <boost/multiprecision/mpfr.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "../../include/jkr.hpp"

using boost::multiprecision::sqrt;
using boost::multiprecision::pow; 

typedef boost::multiprecision::number<boost::multiprecision::mpfr_float_backend<100> > PreciseType; 

TEST_CASE("Calculate JKR radii in the no-adhesion limit", "[jkrContactRadius()]")
{
    const PreciseType R = 0.8;
    const PreciseType Rcell = 0.5;  
    const PreciseType E0 = 3900.0; 
    Matrix<PreciseType, Dynamic, 1> delta = Matrix<PreciseType, Dynamic, 1>::LinSpaced(
        100, 0.0, 2 * (R - Rcell)
    );
    const double tol = 1e-8;

    // For each overlap distance ...  
    for (int i = 0; i < delta.size(); ++i)
    {
        // Calculate the JKR contact radius in the absence of adhesion 
        auto result = jkrContactRadius<PreciseType>(delta(i), R, E0, 0.0); 
        PreciseType radius = result.second;

        // Check that the radius matches the Hertzian expectation
        REQUIRE_THAT(
            static_cast<double>(radius),
            Catch::Matchers::WithinAbs(static_cast<double>(sqrt(R * delta(i))), tol)
        );  
    } 
}

TEST_CASE("Calculate JKR radii in the presence of adhesion", "[jkrContactRadius()]")
{
    const PreciseType R = 0.8;
    const PreciseType Rcell = 0.5;  
    const PreciseType E0 = 3900.0;
    Matrix<PreciseType, Dynamic, 1> delta = Matrix<PreciseType, Dynamic, 1>::LinSpaced(
        100, 0.0, 2 * (R - Rcell)
    );  
    Matrix<PreciseType, Dynamic, 1> gamma = Matrix<PreciseType, Dynamic, 1>::LinSpaced(
        41, 10.0, 100.0
    );
    const double tol = 1e-20;  
    for (int i = 0; i < gamma.size(); ++i)
    {
        for (int j = 0; j < delta.size(); ++j)
        {
            // Calculate the JKR contact radius 
            auto result = jkrContactRadius<PreciseType>(delta(j), R, E0, gamma(i)); 
            PreciseType radius = result.second;

            // Check that the JKR contact radius exceeds the Hertzian expectation
            REQUIRE(radius >= sqrt(R * delta(j))); 

            // Recover the overlap distance from the contact radius 
            PreciseType overlap = radius * radius / R;
            overlap -= 2 * sqrt(boost::math::constants::pi<PreciseType>() * gamma(i) * radius / E0);
            REQUIRE_THAT(
                static_cast<double>(overlap), 
                Catch::Matchers::WithinAbs(static_cast<double>(delta(j)), tol)
            );
        }
    } 
}

TEST_CASE("Calculate JKR equilibrium distance", "[jkrEquilibriumDistance()]")
{
    const PreciseType R = 0.8;
    const PreciseType Rcell = 0.5;  
    const PreciseType E0 = 3900.0;
    Matrix<PreciseType, Dynamic, 1> gamma = Matrix<PreciseType, Dynamic, 1>::LinSpaced(
        41, 10.0, 100.0
    );
    const double tol = 1e-8; 
    for (int i = 0; i < gamma.size(); ++i)
    {
        // Calculate the JKR equilibrium distance 
        PreciseType deq = jkrEquilibriumDistance<PreciseType>(
            R, E0, gamma(i), 0.98 * 2 * R, 2 * Rcell 
        );

        // Check that the contact radius at this equilibrium distance satisfies
        // the expected relation 
        auto result = jkrContactRadius<PreciseType>(2 * R - deq, R, E0, gamma(i));
        PreciseType req = result.second; 
        REQUIRE_THAT(
            static_cast<double>(req), 
            Catch::Matchers::WithinAbs(
                static_cast<double>(
                    pow(
                        9 * boost::math::constants::pi<PreciseType>() * gamma(i) * R * R / E0,
                        static_cast<PreciseType>(1.) / static_cast<PreciseType>(3.)
                    )
                ),
                tol
            )
        ); 
    } 
}
