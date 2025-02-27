/**
 * Test module for `cellSurfaceRepulsionForces()`.
 *
 * Authors:
 *     Kee-Myoung Nam
 *
 * Last updated:
 *     2/26/2025
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

typedef double T; 

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
Array<T, 2, 1> cellSurfaceRepulsionForcesFiniteDiff(const T rz, const T nz,
                                                    const T half_l, const T R,
                                                    const T E0, const T delta = 1e-8)
{
    // Calculate derivative with respect to rz
    T int1_prz = integral1<T>(rz + delta, nz, R, half_l, 2.0, (R - rz - delta) / nz);
    T int2_prz = integral1<T>(rz + delta, nz, R, half_l, 1.5, (R - rz - delta) / nz);
    T total_prz = E0 * sqrt(R) * (
        pow(R, -0.5) * (1 - nz * nz) * int1_prz + (4.0 / 3.0) * nz * nz * int2_prz
    );
    T int1_mrz = integral1<T>(rz - delta, nz, R, half_l, 2.0, (R - rz + delta) / nz); 
    T int2_mrz = integral1<T>(rz - delta, nz, R, half_l, 1.5, (R - rz + delta) / nz);
    T total_mrz = E0 * sqrt(R) * (
        pow(R, -0.5) * (1 - nz * nz) * int1_mrz + (4.0 / 3.0) * nz * nz * int2_mrz
    );
    T force_rz = (total_prz - total_mrz) / (2 * delta);

    // Calculate derivative with respect to nz
    T int1_pnz = integral1<T>(rz, nz + delta, R, half_l, 2.0, (R - rz) / (nz + delta)); 
    T int2_pnz = integral1<T>(rz, nz + delta, R, half_l, 1.5, (R - rz) / (nz + delta)); 
    T total_pnz = E0 * sqrt(R) * (
        pow(R, -0.5) * (1 - (nz + delta) * (nz + delta)) * int1_pnz +
        (4.0 / 3.0) * (nz + delta) * (nz + delta) * int2_pnz
    );
    T int1_mnz = integral1<T>(rz, nz - delta, R, half_l, 2.0, (R - rz) / (nz - delta)); 
    T int2_mnz = integral1<T>(rz, nz - delta, R, half_l, 1.5, (R - rz) / (nz - delta)); 
    T total_mnz = E0 * sqrt(R) * (
        pow(R, -0.5) * (1 - (nz - delta) * (nz - delta)) * int1_mnz +
        (4.0 / 3.0) * (nz - delta) * (nz - delta) * int2_mnz
    );
    T force_nz = (total_pnz - total_mnz) / (2 * delta); 

    Array<T, 2, 1> forces; 
    forces << force_rz, force_nz; 
    return forces;
}

/* ------------------------------------------------------------------- //
 *                             TEST MODULES                            //
 * ------------------------------------------------------------------- */
/**
 * A series of tests for cellSurfaceRepulsionForces().
 */
TEST_CASE("Tests for cell-surface repulsion forces", "[cellSurfaceRepulsionForces()]")
{
    const T R = 0.8;
    const T E0 = 3900.0;
    const T nz_threshold = 1e-8;
    const T delta = 1e-8;
    const T tol = 1e-5;   // Since E0 is on the order of 1e+3 
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

        // Prepare input arrays and compute forces via cellSurfaceRepulsionForces()
        Array<T, Dynamic, Dynamic> cells(1, __ncols_required);
        cells << 0, 0, 0, rz, cos(angles(j)), 0, nz, 0, 0, 0, 0, 0, 0,
                 half_l * 2, half_l, 0, 0, 0, 0, 0, 0, 0;
        Array<T, Dynamic, 1> ss(1);  
        ss << (R - rz) / nz;
        Array<T, Dynamic, 2> forces1 = cellSurfaceRepulsionForces<T>(
            cells, 1e-6, 0, ss, R, E0, nz_threshold
        );

        // Compute forces via finite differences 
        Array<T, 2, 1> forces2 = cellSurfaceRepulsionForcesFiniteDiff(
            rz, nz, half_l, R, E0, delta 
        ); 
        REQUIRE_THAT(forces1(0, 0), Catch::Matchers::WithinAbs(forces2(0), tol));
        REQUIRE_THAT(forces1(0, 1), Catch::Matchers::WithinAbs(forces2(1), tol)); 

        // Case 2: Assume the cell does not contact the surface
        max_overlap = -0.1 * R; 
        rz = R + half_l * nz - max_overlap;

        // Prepare input arrays and compute forces via cellSurfaceRepulsionForces()
        cells << 0, 0, 0, rz, cos(angles(j)), 0, nz, 0, 0, 0, 0, 0, 0,
                 half_l * 2, half_l, 0, 0, 0, 0, 0, 0, 0;
        ss << (R - rz) / nz;
        forces1 = cellSurfaceRepulsionForces<T>(cells, 1e-6, 0, ss, R, E0, nz_threshold);

        // Compute forces via finite differences 
        forces2 = cellSurfaceRepulsionForcesFiniteDiff(rz, nz, half_l, R, E0, delta); 
        REQUIRE_THAT(forces1(0, 0), Catch::Matchers::WithinAbs(forces2(0), delta));
        REQUIRE_THAT(forces1(0, 1), Catch::Matchers::WithinAbs(forces2(1), delta));
        REQUIRE_THAT(forces1(0, 0), Catch::Matchers::WithinAbs(0.0, delta)); 
        REQUIRE_THAT(forces1(0, 1), Catch::Matchers::WithinAbs(0.0, delta));  
    }
}

