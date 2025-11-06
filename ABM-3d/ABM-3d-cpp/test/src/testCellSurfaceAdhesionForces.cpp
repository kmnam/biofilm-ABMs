/**
 * Test module for `cellSurfaceAdhesionForces()`.
 *
 * Authors:
 *     Kee-Myoung Nam
 *
 * Last updated:
 *     3/25/2025
 */
#include <iostream>
#include <cmath>
#include <functional>
#include <Eigen/Dense>
#include <boost/math/constants/constants.hpp>
#include <boost/multiprecision/mpfr.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "../../include/indices.hpp"
#include "../../include/integrals.hpp"
#include "../../include/mechanics.hpp"

using namespace Eigen; 

// Use high-precision type for testing 
typedef boost::multiprecision::number<boost::multiprecision::mpfr_float_backend<100> > T; 

using std::sin; 
using boost::multiprecision::sin; 
using std::sqrt; 
using boost::multiprecision::sqrt;
using std::pow; 
using boost::multiprecision::pow; 

/* ------------------------------------------------------------------- //
 *                           HELPER FUNCTIONS                          //
 * ------------------------------------------------------------------- */
/**
 * Evaluate the cell-surface repulsive force for a cell in contact with
 * the surface by numerical differentiation.
 */
Array<T, 2, 1> cellSurfaceAdhesionForcesFiniteDiff(const T rz, const T nz,
                                                   const T half_l, const T R,
                                                   const T sigma0,
                                                   const T delta = 1e-8)
{
    // Calculate derivative with respect to rz  
    T area_prz = areaIntegral1<T>(rz + delta, nz, R, half_l, (R - rz - delta) / nz);
    T area_mrz = areaIntegral1<T>(rz - delta, nz, R, half_l, (R - rz + delta) / nz);
    T force_rz = -sigma0 * (area_prz - area_mrz) / (2 * delta);  

    // Calculate derivative with respect to nz
    T area_pnz = areaIntegral1<T>(rz, nz + delta, R, half_l, (R - rz) / (nz + delta)); 
    T area_mnz = areaIntegral1<T>(rz, nz - delta, R, half_l, (R - rz) / (nz - delta)); 
    T force_nz = -sigma0 * (area_pnz - area_mnz) / (2 * delta); 

    Array<T, 2, 1> forces; 
    forces << force_rz, force_nz; 
    return forces;
}

/* ------------------------------------------------------------------- //
 *                             TEST MODULES                            //
 * ------------------------------------------------------------------- */
/**
 * A series of tests for cellSurfaceAdhesionForces().
 */
TEST_CASE("Tests for cell-surface repulsion forces", "[cellSurfaceAdhesionForces()]")
{
    const T R = 0.8;
    const T sigma0 = 100.0;
    const T nz_threshold = 1e-8;
    const T delta = 1e-50;
    const double tol = 1e-45;   // Since sigma0 is on the order of 100
    T rz, nz; 
    Array<T, 10, 1> angles = Array<T, 10, 1>::Zero();
    for (int i = 0; i < angles.size(); ++i)
        angles(i) = boost::math::constants::half_pi<T>() * (i + 1) / 10.0; 

    // For each angle ... 
    for (int j = 0; j < angles.size(); ++j)
    {
        // Define the z-orientation
        nz = sin(angles(j));

        // Case 1: Assume the cell has a maximum overlap of 0.2 * R
        T half_l = 0.5;
        T max_overlap = 0.2 * R;
        rz = R + half_l * nz - max_overlap;

        // Prepare input arrays and compute forces via cellSurfaceAdhesionForces()
        Array<T, Dynamic, Dynamic> cells(1, __ncols_required);
        cells << 0, 0, 0, rz, cos(angles(j)), 0, nz, 0, 0, 0, 0, 0, 0,   // Coordinates
                 half_l * 2, half_l,     // Length and half-length
                 0, 0,                   // Birth time and growth rate
                 0, 0,                   // Ambient viscosity and friction coefficient 
                 sigma0,                 // Cell-surface adhesion energy density
                 0;                      // Group ID
        Array<T, Dynamic, 1> ss(1);  
        ss << (R - rz) / nz;
        Array<int, Dynamic, 1> assume_2d = Array<int, Dynamic, 1>::Zero(1); 
        assume_2d(0) = (nz < nz_threshold); 
        Array<T, Dynamic, 6> forces1 = cellSurfaceAdhesionForces<T>(
            cells, 1e-6, 0, ss, R, assume_2d, false
        );

        // Compute forces via finite differences 
        Array<T, 2, 1> forces2 = cellSurfaceAdhesionForcesFiniteDiff(
            rz, nz, half_l, R, sigma0, delta 
        ); 
        REQUIRE_THAT(
            static_cast<double>(forces1(0, 0)), Catch::Matchers::WithinAbs(0.0, tol)
        );
        REQUIRE_THAT(
            static_cast<double>(forces1(0, 1)), Catch::Matchers::WithinAbs(0.0, tol)
        ); 
        REQUIRE_THAT(
            static_cast<double>(forces1(0, 2)),
            Catch::Matchers::WithinAbs(static_cast<double>(forces2(0)), tol)
        );
        REQUIRE_THAT(
            static_cast<double>(forces1(0, 3)), Catch::Matchers::WithinAbs(0.0, tol)
        ); 
        REQUIRE_THAT(
            static_cast<double>(forces1(0, 4)), Catch::Matchers::WithinAbs(0.0, tol)
        ); 
        REQUIRE_THAT(
            static_cast<double>(forces1(0, 5)),
            Catch::Matchers::WithinAbs(static_cast<double>(forces2(1)), tol)
        ); 

        // Case 2: Assume the cell does not contact the surface
        max_overlap = -0.1 * R; 
        rz = R + half_l * nz - max_overlap;

        // Prepare input arrays and compute forces via cellSurfaceAdhesionForces()
        cells << 0, 0, 0, rz, cos(angles(j)), 0, nz, 0, 0, 0, 0, 0, 0,   // Coordinates
                 half_l * 2, half_l,     // Length and half-length
                 0, 0,                   // Birth time and growth rate
                 0, 0,                   // Ambient viscosity and friction coefficient 
                 sigma0,                 // Cell-surface adhesion energy density
                 0;                      // Group ID
        ss << (R - rz) / nz;
        assume_2d(0) = (nz < nz_threshold); 
        forces1 = cellSurfaceAdhesionForces<T>(cells, 1e-6, 0, ss, R, assume_2d, false);

        // Compute forces via finite differences 
        forces2 = cellSurfaceAdhesionForcesFiniteDiff(rz, nz, half_l, R, sigma0, delta);
        REQUIRE_THAT(
            static_cast<double>(forces1(0, 0)), Catch::Matchers::WithinAbs(0.0, tol)
        );
        REQUIRE_THAT(
            static_cast<double>(forces1(0, 1)), Catch::Matchers::WithinAbs(0.0, tol)
        ); 
        REQUIRE_THAT(
            static_cast<double>(forces1(0, 2)),
            Catch::Matchers::WithinAbs(static_cast<double>(forces2(0)), tol)
        );
        REQUIRE_THAT(
            static_cast<double>(forces1(0, 3)), Catch::Matchers::WithinAbs(0.0, tol)
        ); 
        REQUIRE_THAT(
            static_cast<double>(forces1(0, 4)), Catch::Matchers::WithinAbs(0.0, tol)
        ); 
        REQUIRE_THAT(
            static_cast<double>(forces1(0, 5)),
            Catch::Matchers::WithinAbs(static_cast<double>(forces2(1)), tol)
        );
        REQUIRE_THAT(
            static_cast<double>(forces1(0, 2)), Catch::Matchers::WithinAbs(0.0, tol)
        ); 
        REQUIRE_THAT(
            static_cast<double>(forces1(0, 5)), Catch::Matchers::WithinAbs(0.0, tol)
        );  
    }
}

