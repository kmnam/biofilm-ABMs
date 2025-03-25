/**
 * Test module for `compositeViscosityForceMatrix()`. 
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
 *
 * @param rz
 * @param nz
 * @param dr
 * @param dn
 * @param half_l
 * @param R
 * @param eta0
 * @param delta
 */
Array<T, 6, 1> getAmbientViscosityForces(const Ref<const Matrix<T, 3, 1> >& dr,
                                         const Ref<const Matrix<T, 3, 1> >& dn,
                                         const T l, const T eta0)
{
    Array<T, 6, 1> forces; 
    forces.head(3) = eta0 * l * dr;
    forces.tail(3) = eta0 * l * l * l * dn / 12; 
    return forces;  
}

/**
 *
 * @param rz
 * @param nz
 * @param dr
 * @param dn
 * @param half_l
 * @param R
 * @param eta1
 * @param delta
 */
Array<T, 6, 1> getCellSurfaceFrictionForces(const T rz, const T nz,
                                            const Ref<const Matrix<T, 3, 1> >& dr,
                                            const Ref<const Matrix<T, 3, 1> >& dn,
                                            const T half_l, const T R, const T eta1)
{
    T ss = (R - rz) / nz; 
    T int1 = areaIntegral1<T>(rz, nz, R, half_l, ss);
    T int2 = areaIntegral2<T>(rz, nz, R, half_l, ss);
    T int3 = areaIntegral3<T>(rz, nz, R, half_l, ss);
    Array<T, 3, 1> dr_xy, dn_xy; 
    dr_xy << dr(0), dr(1), 0.0; 
    dn_xy << dn(0), dn(1), 0.0;  
    Array<T, 6, 1> forces; 
    forces.head(3) = (eta1 / R) * (dr_xy * int1 + dn_xy * int2);
    forces.tail(3) = (eta1 / R) * (dr_xy * int2 + dn_xy * int3);
    return forces;  
}

/* ------------------------------------------------------------------- //
 *                             TEST MODULES                            //
 * ------------------------------------------------------------------- */
/**
 * A series of tests for compositeViscosityForceMatrix(). 
 */
TEST_CASE("Tests for viscosity forces", "[compositeViscosityForceMatrix()]")
{
    const T R = 0.8;
    const T Rcell = 0.5; 
    const T E0 = 3900.0;
    const T Ecell = 3900000.0; 
    const T sigma0 = 100.0;
    const T eta0 = 0.072; 
    const T eta1 = 720.0;
    const T nz_threshold = 1e-8;
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

        // Compute the viscosity force matrix
        Matrix<T, 6, 6> vmatrix = compositeViscosityForceMatrix<T>(
            rz, nz, 2 * half_l, half_l, (R - rz) / nz, eta0, eta1, R, 0, 1e-6, 0
        ).matrix();

        // Compute the viscosity and cell-surface friction forces for a given 
        // velocity vector 
        Matrix<T, 6, 1> velocities; 
        velocities << 3.0, 2.0, 1.0, 0.5, 0.8, 0.1;
        Matrix<T, 6, 1> prod = vmatrix * velocities;
        Array<T, 6, 1> forces1 = getAmbientViscosityForces(
            velocities.head(3), velocities.tail(3), 2 * half_l, eta0
        );
        Array<T, 6, 1> forces2 = getCellSurfaceFrictionForces(
            rz, nz, velocities.head(3), velocities.tail(3), half_l, R, eta1
        ); 

        // Check that the forces match 
        REQUIRE_THAT(
            static_cast<double>(prod(0)),
            Catch::Matchers::WithinAbs(static_cast<double>(forces1(0) + forces2(0)), tol)
        );
        REQUIRE_THAT(
            static_cast<double>(prod(1)),
            Catch::Matchers::WithinAbs(static_cast<double>(forces1(1) + forces2(1)), tol)
        );
        REQUIRE_THAT(
            static_cast<double>(prod(2)),
            Catch::Matchers::WithinAbs(static_cast<double>(forces1(2)), tol)
        );
        REQUIRE_THAT(
            static_cast<double>(prod(3)),
            Catch::Matchers::WithinAbs(static_cast<double>(forces1(3) + forces2(3)), tol)
        );
        REQUIRE_THAT(
            static_cast<double>(prod(4)),
            Catch::Matchers::WithinAbs(static_cast<double>(forces1(4) + forces2(4)), tol)
        );
        REQUIRE_THAT(
            static_cast<double>(prod(5)),
            Catch::Matchers::WithinAbs(static_cast<double>(forces1(5)), tol)
        ); 
    }
}

