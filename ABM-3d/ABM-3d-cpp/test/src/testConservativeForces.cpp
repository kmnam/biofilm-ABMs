/**
 * Test module for `getConservativeForces()`.
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
Array<T, 2, 1> getTotalCellSurfaceForcesFiniteDiff(const T rz, const T nz,
                                                   const T half_l, const T R,
                                                   const T E0, const T sigma0,
                                                   const T delta)
{
    T four("4"), three("3"); 
    T four_thirds = four / three; 

    // Compute the partial derivatives of the cell-surface repulsion energy
    T int1_prz = integral1<T>(rz + delta, nz, R, half_l, 2.0, (R - rz - delta) / nz);
    T int2_prz = integral1<T>(rz + delta, nz, R, half_l, 1.5, (R - rz - delta) / nz); 
    T depth_prz = pow(R, -0.5) * (1 - nz * nz) * int1_prz + four_thirds * nz * nz * int2_prz;
    T int1_mrz = integral1<T>(rz - delta, nz, R, half_l, 2.0, (R - rz + delta) / nz);
    T int2_mrz = integral1<T>(rz - delta, nz, R, half_l, 1.5, (R - rz + delta) / nz); 
    T depth_mrz = pow(R, -0.5) * (1 - nz * nz) * int1_mrz + four_thirds * nz * nz * int2_mrz;
    T int1_pnz = integral1<T>(rz, nz + delta, R, half_l, 2.0, (R - rz) / (nz + delta));
    T int2_pnz = integral1<T>(rz, nz + delta, R, half_l, 1.5, (R - rz) / (nz + delta));  
    T depth_pnz = pow(R, -0.5) * (1 - (nz + delta) * (nz + delta)) * int1_pnz + four_thirds * (nz + delta) * (nz + delta) * int2_pnz; 
    T int1_mnz = integral1<T>(rz, nz - delta, R, half_l, 2.0, (R - rz) / (nz - delta));
    T int2_mnz = integral1<T>(rz, nz - delta, R, half_l, 1.5, (R - rz) / (nz - delta));
    T depth_mnz = pow(R, -0.5) * (1 - (nz - delta) * (nz - delta)) * int1_mnz + four_thirds * (nz - delta) * (nz - delta) * int2_mnz;
    T frep_rz = E0 * sqrt(R) * (depth_prz - depth_mrz) / (2 * delta);
    T frep_nz = E0 * sqrt(R) * (depth_pnz - depth_mnz) / (2 * delta);

    // Compute the partial derivatives of the cell-surface adhesion energy 
    T area_prz = areaIntegral1<T>(rz + delta, nz, R, half_l, (R - rz - delta) / nz);
    T area_mrz = areaIntegral1<T>(rz - delta, nz, R, half_l, (R - rz + delta) / nz);
    T area_pnz = areaIntegral1<T>(rz, nz + delta, R, half_l, (R - rz) / (nz + delta));
    T area_mnz = areaIntegral1<T>(rz, nz - delta, R, half_l, (R - rz) / (nz - delta));
    T fadh_rz = -sigma0 * (area_prz - area_mrz) / (2 * delta); 
    T fadh_nz = -sigma0 * (area_pnz - area_mnz) / (2 * delta); 

    Array<T, 2, 1> forces; 
    forces << -frep_rz - fadh_rz, -frep_nz - fadh_nz;
    return forces;  
}

/* ------------------------------------------------------------------- //
 *                             TEST MODULES                            //
 * ------------------------------------------------------------------- */
/**
 * A series of tests for `getConservativeForces()`, for one cell. 
 */
TEST_CASE("Tests for conservative forces for a single cell", "[getConservativeForces()]")
{
    const T R = 0.8;
    const T Req = 0.4; 
    const T Rcell = 0.5; 
    const T E0 = 3900.0;
    const T Ecell = 3900000.0; 
    const T sigma0 = 100.0;
    const T nz_threshold = 1e-8;
    Array<T, 3, 1> repulsion_prefactors; 
    repulsion_prefactors << (4. / 3.) * E0 * sqrt(Req),
                            (4. / 3.) * Ecell * sqrt(Rcell / 2), 
                            (4. / 3.) * E0 * sqrt(Req) * pow(2 * (R - Rcell), 1.5);
    std::unordered_map<std::string, T> adhesion_params; 
    std::unique_ptr<JKRData<T> > jkr_data;  
    const T delta = 1e-50;
    const double tol = 1e-12;   // Subtraction leads to precision loss  
    T rz, nz; 
    
    // Generate 11 possible orientations for the second cell, so that the 
    // angle between the two orientation vectors ranges between 0 and pi/2 
    Array<T, Dynamic, 1> angles = Array<T, Dynamic, 1>::LinSpaced(11, 0.0, 1.0);
    angles *= boost::math::constants::half_pi<T>(); 

    // For each angle ... 
    for (int j = 0; j < angles.size(); ++j)
    {
        // Define the z-orientation
        nz = sin(angles(j));

        // Case 1: Assume the cell has a maximum overlap of 0.2 * R
        T half_l = 0.5;
        T max_overlap = 0.2 * R;
        rz = R + half_l * nz - max_overlap;

        // Prepare input arrays and compute forces via getConservativeForces()
        Array<T, Dynamic, Dynamic> cells(1, __ncols_required);
        cells << 0, 0, 0, rz, cos(angles(j)), 0, nz, 0, 0, 0, 0, 0, 0,     // Coordinates
                 half_l * 2, half_l,     // Length and half-length
                 0, 0,                   // Birth time and growth rate
                 0, 0,                   // Ambient viscosity and friction coefficient 
                 sigma0,                 // Cell-surface adhesion energy density
                 0;                      // Group ID
        Array<T, Dynamic, 7> neighbors(0, 7); 
        Array<int, Dynamic, 1> assume_2d = Array<int, Dynamic, 1>::Zero(1);
        assume_2d(0) = (nz < nz_threshold);  
        Array<T, Dynamic, 6> forces = getConservativeForces<T>(
            cells, neighbors, 5e-7, 0, R, Rcell, E0, repulsion_prefactors, 
            assume_2d, nz_threshold, AdhesionMode::NONE, adhesion_params,
            jkr_data, -1, FrictionMode::NONE, -1, false, false
        );

        // Check that the forces in the x- and y-directions are zero 
        REQUIRE_THAT(static_cast<double>(forces(0)), Catch::Matchers::WithinAbs(0.0, tol));
        REQUIRE_THAT(static_cast<double>(forces(1)), Catch::Matchers::WithinAbs(0.0, tol));

        // Check that the torques in the x- and y-directions are zero 
        REQUIRE_THAT(static_cast<double>(forces(3)), Catch::Matchers::WithinAbs(0.0, tol)); 
        REQUIRE_THAT(static_cast<double>(forces(4)), Catch::Matchers::WithinAbs(0.0, tol));

        // Check that the forces and torques in the z-direction match the 
        // target values obtained via finite differences
        Array<T, 2, 1> targets; 
        if (nz > 0)
        {
            targets = getTotalCellSurfaceForcesFiniteDiff(
                rz, nz, half_l, R, E0, sigma0, delta
            );
        }
        else 
        {
            targets(0) = 2 * E0 * (R - rz) * 2 * half_l;
            targets(0) -= 0.5 * sigma0 * sqrt(R) * 2 * half_l / sqrt(R - rz);
            targets(1) = 0.0;
        }
        REQUIRE_THAT(
            static_cast<double>(forces(2)),
            Catch::Matchers::WithinAbs(static_cast<double>(targets(0)), tol)
        );
        REQUIRE_THAT(
            static_cast<double>(forces(5)),
            Catch::Matchers::WithinAbs(static_cast<double>(targets(1)), tol)
        ); 
    }
}

