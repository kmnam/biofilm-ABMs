/**
 * Test module for `getVelocities()`.
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
 *                             TEST MODULES                            //
 * ------------------------------------------------------------------- */
/**
 * A series of tests for getVelocities().
 */
TEST_CASE("Tests for velocity calculations", "[getVelocities()]")
{
    const T R = 0.8;
    const T Rcell = 0.5; 
    const T E0 = 3900.0;
    const T Ecell = 3900000.0; 
    const T sigma0 = 100.0;
    const T eta0 = 0.072; 
    const T eta1 = 720.0;
    const T nz_threshold = 1e-8;
    Array<T, 3, 1> cell_cell_prefactors;  
    cell_cell_prefactors << 2.5 * E0 * sqrt(R),
                            2.5 * E0 * sqrt(R) * pow(2 * R - 2 * Rcell, 1.5),
                            2.5 * Ecell * sqrt(Rcell);
    std::unordered_map<std::string, T> adhesion_params;  
    const T delta = 1e-50;
    const double tol = 1e-12;   // Subtraction leads to precision loss  
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

        // Case 1a: Compute the velocity vector via getVelocities() with
        // zero noise 
        Array<T, Dynamic, Dynamic> cells(1, __ncols_required);
        cells << 0, 0, 0, rz, cos(angles(j)), 0, nz, 0, 0, 0, 0, 0, 0,
                 half_l * 2, half_l, 0, 0, eta0, eta1, eta1, sigma0, 0;
        Array<T, Dynamic, 7> neighbors(0, 7); 
        Array<int, Dynamic, 1> to_adhere(0);
        Array<int, Dynamic, 1> assume_2d = Array<int, Dynamic, 1>::Zero(1);
        assume_2d(0) = (cells(0, __colidx_nz) < nz_threshold); 
        Array<T, Dynamic, 6> noise = Array<T, Dynamic, 6>::Zero(1, 6);
        Array<T, Dynamic, 6> velocities = getVelocities<T>(
            cells, neighbors, to_adhere, 1e-7, 0, R, Rcell, cell_cell_prefactors,
            E0, assume_2d, noise, AdhesionMode::NONE, adhesion_params, false,
            false
        );

        // Compute the viscosity force matrix
        Matrix<T, 6, 6> vmatrix = compositeViscosityForceMatrix<T>(
            rz, nz, 2 * half_l, half_l, (R - rz) / nz, eta0, eta1, R, 0, 
            1e-7, 0
        ).matrix();

        // Compute the conservative force vector
        Matrix<T, Dynamic, 6> cforces = getConservativeForces<T>(
            cells, neighbors, to_adhere, 1e-7, 0, R, Rcell, cell_cell_prefactors,
            E0, assume_2d, AdhesionMode::NONE, adhesion_params, false, false
        ).matrix();

        // Check that the velocity vector solves the desired linear equation
        Matrix<T, 6, 1> prod = vmatrix * velocities.row(0).matrix().transpose();
        for (int k = 0; k < 3; ++k)
        {
            REQUIRE_THAT(
                static_cast<double>(prod(k)),
                Catch::Matchers::WithinAbs(static_cast<double>(cforces(0, k)), tol)
            );
        }
        T lambda = (prod(5) - cforces(5)) / (2 * nz);   // Get the Lagrange multiplier
        REQUIRE_THAT(
            static_cast<double>(prod(3)), 
            Catch::Matchers::WithinAbs(
                static_cast<double>(cforces(0, 3) + 2 * lambda * cells(__colidx_nx)),
                tol
            )
        );
        REQUIRE_THAT(
            static_cast<double>(prod(4)), 
            Catch::Matchers::WithinAbs(
                static_cast<double>(cforces(0, 4) + 2 * lambda * cells(__colidx_ny)),
                tol
            )
        );
        REQUIRE_THAT(
            static_cast<double>(prod(5)),   // This should be satisfied trivially  
            Catch::Matchers::WithinAbs(
                static_cast<double>(cforces(0, 5) + 2 * lambda * cells(__colidx_nz)),
                tol
            )
        );

        // Case 1b: Add a small amount of noise and re-do the calculation 
        for (int k = 0; k < 6; ++k)
            noise(k) = k * 1e-6; 
        velocities = getVelocities<T>(
            cells, neighbors, to_adhere, 1e-7, 0, R, Rcell, cell_cell_prefactors,
            E0, assume_2d, noise, AdhesionMode::NONE, adhesion_params, false,
            false
        );

        // Check that the velocity vector solves the desired linear equation
        prod = vmatrix * velocities.row(0).matrix().transpose();
        for (int k = 0; k < 3; ++k)
        {
            REQUIRE_THAT(
                static_cast<double>(prod(k)),
                Catch::Matchers::WithinAbs(static_cast<double>(cforces(0, k) + noise(k)), tol)
            );
        }
        lambda = (prod(5) - cforces(5) - noise(5)) / (2 * nz);   // Get the Lagrange multiplier
        REQUIRE_THAT(
            static_cast<double>(prod(3)), 
            Catch::Matchers::WithinAbs(
                static_cast<double>(cforces(0, 3) + noise(3) + 2 * lambda * cells(__colidx_nx)),
                tol
            )
        );
        REQUIRE_THAT(
            static_cast<double>(prod(4)), 
            Catch::Matchers::WithinAbs(
                static_cast<double>(cforces(0, 4) + noise(4) + 2 * lambda * cells(__colidx_ny)),
                tol
            )
        );
        REQUIRE_THAT(
            static_cast<double>(prod(5)),   // This should be satisfied trivially  
            Catch::Matchers::WithinAbs(
                static_cast<double>(cforces(0, 5) + noise(5) + 2 * lambda * cells(__colidx_nz)),
                tol
            )
        ); 

    }
}

