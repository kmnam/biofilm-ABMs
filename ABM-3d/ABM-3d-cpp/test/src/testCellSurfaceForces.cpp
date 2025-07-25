/**
 * Test module for functions in `cellSurfaceForces.hpp`.
 *
 * Authors:
 *     Kee-Myoung Nam
 *
 * Last updated:
 *     4/15/2025
 */
#include <iostream>
#include <cmath>
#include <functional>
#include <Eigen/Dense>
#include <boost/math/constants/constants.hpp>
#include <boost/multiprecision/mpfr.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "../../include/integrals.hpp"
#include "../../include/cellSurfaceForces.hpp"

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
 */
Array<T, 6, 1> cellSurfaceRepulsionForceHorizontal(const T rz, const T half_l,
                                                   const T R, const T E0,
                                                   const T delta)
{
    if (R - rz < 0)
        return Array<T, 6, 1>::Zero(); 

    T int_prz = pow(R, -0.5) * (R - rz - delta) * (R - rz - delta) * 2 * half_l;
    T int_mrz = pow(R, -0.5) * (R - rz + delta) * (R - rz + delta) * 2 * half_l;  
    T force_rz = E0 * sqrt(R) * (int_prz - int_mrz) / (2 * delta); 
    
    Array<T, 6, 1> forces; 
    forces << 0, 0, -force_rz, 0, 0, 0;
    return forces; 
} 

/**
 *
 */
Array<T, 6, 1> cellSurfaceAdhesionForceHorizontal(const T rz, const T half_l,
                                                  const T R, const T sigma0,
                                                  const T delta)
{
    if (R - rz < 0)
        return Array<T, 6, 1>::Zero(); 

    T int_prz = sqrt(R) * sqrt(R - rz - delta) * 2 * half_l; 
    T int_mrz = sqrt(R) * sqrt(R - rz + delta) * 2 * half_l; 
    T force_rz = -sigma0 * (int_prz - int_mrz) / (2 * delta); 
    
    Array<T, 6, 1> forces; 
    forces << 0, 0, -force_rz, 0, 0, 0;
    return forces; 
}

/**
 * Evaluate the cell-surface repulsive force for a cell in contact with
 * the surface by numerical differentiation.
 *
 * The first three coordinates correspond to the force vector, and the
 * latter three to the moment vector.
 *
 * This function assumes that the z-orientation is positive, nz > 0.  
 */
Array<T, 6, 1> cellSurfaceRepulsionForceFiniteDiff(const T rz,
                                                   const Ref<const Matrix<T, 3, 1> >& n,
                                                   const T half_l, const T R,
                                                   const T E0, const T delta = 1e-8)
{
    T four("4"), three("3"); 
    T four_thirds = four / three; 

    // Calculate the force vector, whose only nonzero coordinate is the 
    // derivative with respect to rz
    T nz = n(2); 
    T int1_prz = integral1<T>(rz + delta, nz, R, half_l, 2.0, (R - rz - delta) / nz);
    T int2_prz = integral1<T>(rz + delta, nz, R, half_l, 1.5, (R - rz - delta) / nz);
    T total_prz = E0 * sqrt(R) * (
        pow(R, -0.5) * (1 - nz * nz) * int1_prz + four_thirds * nz * nz * int2_prz
    );
    T int1_mrz = integral1<T>(rz - delta, nz, R, half_l, 2.0, (R - rz + delta) / nz); 
    T int2_mrz = integral1<T>(rz - delta, nz, R, half_l, 1.5, (R - rz + delta) / nz);
    T total_mrz = E0 * sqrt(R) * (
        pow(R, -0.5) * (1 - nz * nz) * int1_mrz + four_thirds * nz * nz * int2_mrz
    );
    T force_rz = -(total_prz - total_mrz) / (2 * delta);

    // Calculate the cross product of the orientation vector with z
    Matrix<T, 3, 1> z; 
    z << 0, 0, 1; 
    Matrix<T, 3, 1> cross = n.cross(z); 

    // Calculate the moment vector 
    T int3_prz = integral2<T>(rz + delta, nz, R, half_l, 2.0, (R - rz - delta) / nz); 
    T int4_prz = integral2<T>(rz + delta, nz, R, half_l, 1.5, (R - rz - delta) / nz); 
    total_prz = E0 * sqrt(R) * (
        pow(R, -0.5) * (1 - nz * nz) * int3_prz + four_thirds * nz * nz * int4_prz
    );
    T int3_mrz = integral2<T>(rz - delta, nz, R, half_l, 2.0, (R - rz + delta) / nz); 
    T int4_mrz = integral2<T>(rz - delta, nz, R, half_l, 1.5, (R - rz + delta) / nz); 
    total_mrz = E0 * sqrt(R) * (
        pow(R, -0.5) * (1 - nz * nz) * int3_mrz + four_thirds * nz * nz * int4_mrz
    );
    Matrix<T, 3, 1> moment = -cross * (total_prz - total_mrz) / (2 * delta); 

    Array<T, 6, 1> forces; 
    forces << 0, 0, force_rz, moment(0), moment(1), moment(2); 
    return forces;
}

/**
 * Evaluate the cell-surface adhesive force for a cell in contact with
 * the surface by numerical differentiation.
 *
 * The first three coordinates correspond to the force vector, and the
 * latter three to the moment vector.
 *
 * This function assumes that the z-orientation is positive, nz > 0.  
 */
Array<T, 6, 1> cellSurfaceAdhesionForceFiniteDiff(const T rz,
                                                  const Ref<const Matrix<T, 3, 1> >& n,
                                                  const T half_l, const T R,
                                                  const T sigma0,
                                                  const T delta = 1e-8)
{
    // Calculate the force vector, whose only nonzero coordinate is the 
    // derivative with respect to rz
    T nz = n(2); 
    T area_prz = areaIntegral1<T>(rz + delta, nz, R, half_l, (R - rz - delta) / nz);
    T area_mrz = areaIntegral1<T>(rz - delta, nz, R, half_l, (R - rz + delta) / nz);
    T force_rz = sigma0 * (area_prz - area_mrz) / (2 * delta);  

    // Calculate the cross product of the orientation vector with z
    Matrix<T, 3, 1> z; 
    z << 0, 0, 1; 
    Matrix<T, 3, 1> cross = n.cross(z); 

    // Calculate the moment vector 
    area_prz = areaIntegral2<T>(rz + delta, nz, R, half_l, (R - rz - delta) / nz); 
    area_mrz = areaIntegral2<T>(rz - delta, nz, R, half_l, (R - rz + delta) / nz);
    Matrix<T, 3, 1> moment = sigma0 * cross * (area_prz - area_mrz) / (2 * delta);  

    Array<T, 6, 1> forces; 
    forces << 0, 0, force_rz, moment(0), moment(1), moment(2); 
    return forces;
}

/**
 * Evaluate the cell-surface friction force for a horizontal cell in contact
 * with the surface.
 */
Array<T, 6, 1> cellSurfaceFrictionForceHorizontal(const T rz,
                                                  const Ref<const Matrix<T, 3, 1> >& n,  
                                                  const Ref<const Matrix<T, 3, 1> >& dr,
                                                  const Ref<const Matrix<T, 3, 1> >& omega,
                                                  const T half_l, const T R,
                                                  const T eta, const T delta)
{
    Array<T, 6, 1> forces = Array<T, 6, 1>::Zero();
    if (rz >= R)
        return forces;

    // Calculate the cell-surface contact area density
    T area_density = sqrt(R) * sqrt(R - rz);

    // Calculate the forces through finite differences
    Matrix<T, 3, 1> dn = omega.cross(n); 
    Matrix<T, 3, 1> drxy, dnxy;
    drxy << dr(0), dr(1), 0; 
    dnxy << dn(0), dn(1), 0;
    T int1 = area_density * 2 * half_l; 
    T int2 = 0.0; 
    T int3 = area_density * pow(2 * half_l, 3) / 12.0;
    T int4 = 0.0;  
    for (int i = 0; i < 3; ++i)
    {
        Matrix<T, 3, 1> dx = Matrix<T, 3, 1>::Zero(); 
        dx(i) = delta;
        T int_p = (
            (drxy + dx).dot(drxy + dx) * int1 + 2 * (drxy + dx).dot(dnxy) * int2 
            + dnxy.dot(dnxy) * int3
        );
        T int_m = (
            (drxy - dx).dot(drxy - dx) * int1 + 2 * (drxy - dx).dot(dnxy) * int2
            + dnxy.dot(dnxy) * int3
        );
        forces(i) = -eta / (2 * R) * (int_p - int_m) / (2 * delta);  
    }

    // Calculate the moments through finite differences
    Matrix<T, 3, 1> moment_precross;  
    for (int i = 0; i < 3; ++i)
    {
        Matrix<T, 3, 1> dx = Matrix<T, 3, 1>::Zero(); 
        dx(i) = delta;
        T int_p = (
            (drxy + dx).dot(drxy + dx) * int2 + 2 * (drxy + dx).dot(dnxy) * int3 
            + dnxy.dot(dnxy) * int4
        );
        T int_m = (
            (drxy - dx).dot(drxy - dx) * int2 + 2 * (drxy - dx).dot(dnxy) * int3
            + dnxy.dot(dnxy) * int4
        );
        moment_precross(i) = -eta / (2 * R) * (int_p - int_m) / (2 * delta);  
    }
    forces(Eigen::seq(3, 5)) = n.cross(moment_precross); 
    
    return forces;  
}

/**
 *
 */
T areaIntegral4(const T rz, const T nz, const T R, const T half_l, const T ss)
{
    T overlap1 = pow(phi<T>(rz, nz, R, -half_l), 1.5); 
    T overlap2 = pow(phi<T>(rz, nz, R, half_l), 1.5);  
    T term1a = 0.0; 
    if (ss > half_l)
        term1a = (pow(-half_l, 3) * overlap1 - pow(half_l, 3) * overlap2) / (1.5 * nz);
    else if (ss > -half_l)
        term1a = (pow(-half_l, 3) * overlap1) / (1.5 * nz);
    T term1b = 3 * integral3<T>(rz, nz, R, half_l, 1.5, ss) / (1.5 * nz);  
    T term1 = pow(R, 0.5) * (1 - nz * nz) * (term1a + term1b);
    T term2 = 0.0; 
    if (ss > -half_l && ss <= half_l)
        term2 = boost::math::constants::pi<T>() * R * nz * nz * (pow(ss, 4) - pow(-half_l, 4)) / 4;
    
    return term1 + term2;  
}

/**
 * Evaluate the cell-surface friction force for a cell in contact with the
 * surface.
 *
 * This function assumes that the cell is *not* horizontal (nz > 0).
 */
Array<T, 6, 1> cellSurfaceFrictionForceFiniteDiff(const T rz,
                                                  const Ref<const Matrix<T, 3, 1> >& n,  
                                                  const Ref<const Matrix<T, 3, 1> >& dr, 
                                                  const Ref<const Matrix<T, 3, 1> >& omega,  
                                                  const T half_l, const T R,
                                                  const T eta, const T delta)
{
    Array<T, 6, 1> forces = Array<T, 6, 1>::Zero();

    // Check if the maximum overlap between the cell and the surface is 
    // greater than zero  
    if (phi<T>(rz, n(2), R, -half_l) <= 0)
        return forces;

    // Calculate the forces through finite differences
    Matrix<T, 3, 1> dn = omega.cross(n); 
    Matrix<T, 3, 1> drxy, dnxy;
    drxy << dr(0), dr(1), 0; 
    dnxy << dn(0), dn(1), 0;
    T int1 = areaIntegral1<T>(rz, n(2), R, half_l, (R - rz) / n(2));
    T int2 = areaIntegral2<T>(rz, n(2), R, half_l, (R - rz) / n(2)); 
    T int3 = areaIntegral3<T>(rz, n(2), R, half_l, (R - rz) / n(2)); 
    T int4 = areaIntegral4(rz, n(2), R, half_l, (R - rz) / n(2));
    for (int i = 0; i < 3; ++i)
    {
        Matrix<T, 3, 1> dx = Matrix<T, 3, 1>::Zero(); 
        dx(i) = delta;
        T int_p = (
            (drxy + dx).dot(drxy + dx) * int1 + 2 * (drxy + dx).dot(dnxy) * int2 
            + dnxy.dot(dnxy) * int3
        );
        T int_m = (
            (drxy - dx).dot(drxy - dx) * int1 + 2 * (drxy - dx).dot(dnxy) * int2
            + dnxy.dot(dnxy) * int3
        );
        forces(i) = -eta / (2 * R) * (int_p - int_m) / (2 * delta);  
    }

    // Calculate the moments through finite differences
    Matrix<T, 3, 1> moment_precross;  
    for (int i = 0; i < 3; ++i)
    {
        Matrix<T, 3, 1> dx = Matrix<T, 3, 1>::Zero(); 
        dx(i) = delta;
        T int_p = (
            (drxy + dx).dot(drxy + dx) * int2 + 2 * (drxy + dx).dot(dnxy) * int3 
            + dnxy.dot(dnxy) * int4
        );
        T int_m = (
            (drxy - dx).dot(drxy - dx) * int2 + 2 * (drxy - dx).dot(dnxy) * int3
            + dnxy.dot(dnxy) * int4
        );
        moment_precross(i) = -eta / (2 * R) * (int_p - int_m) / (2 * delta);  
    }
    forces(Eigen::seq(3, 5)) = n.cross(moment_precross); 
    
    return forces;  
}

/* ------------------------------------------------------------------- //
 *                             TEST MODULES                            //
 * ------------------------------------------------------------------- */
/**
 * A series of tests for cellSurfaceRepulsionForce().
 */
TEST_CASE("Tests for cell-surface repulsion forces", "[cellSurfaceRepulsionForces()]")
{
    const T R = 0.8;
    const T E0 = 3900.0;
    const T delta = 1e-50;
    const double tol = 1e-8;   // Since one set of calculations is done with doubles
    std::vector<double> r {0, 0, 0};
    std::vector<double> n {0, 0, 0};
    T rz, nz, max_overlap;
    Matrix<T, 3, 1> n2; 
    T half_l = 0.5;
    std::vector<double> forces1; 
    Array<T, 6, 1> forces2;  

    // --------------------------------------------------------------- //
    //                      HORIZONTAL TEST CASES                      //
    // --------------------------------------------------------------- //
    nz = 0.0;
    n[0] = 1.0; 
    n[2] = 0.0;

    // Case 1: Assume the cell has a maximum overlap of 0.04 * R
    max_overlap = 0.04 * R;
    rz = R + half_l * nz - max_overlap;
    r[2] = static_cast<double>(rz); 

    // Compute forces via cellSurfaceRepulsionForce()
    forces1 = cellSurfaceRepulsionForce(
        r, n, static_cast<double>(half_l), static_cast<double>(R),
        static_cast<double>(E0)
    ); 

    // Compute forces via finite differences 
    forces2 = cellSurfaceRepulsionForceHorizontal(rz, half_l, R, E0, delta);
    REQUIRE_THAT(
        forces1[2],
        Catch::Matchers::WithinAbs(static_cast<double>(forces2(2)), tol)
    );
    for (int i = 3; i < 6; ++i)
        REQUIRE_THAT(forces1[i], Catch::Matchers::WithinAbs(0.0, tol)); 

    // Case 2: Assume the cell has a maximum overlap of 0.2 * R
    max_overlap = 0.2 * R;
    rz = R + half_l * nz - max_overlap;
    r[2] = static_cast<double>(rz); 

    // Compute forces via cellSurfaceRepulsionForce()
    forces1 = cellSurfaceRepulsionForce(
        r, n, static_cast<double>(half_l), static_cast<double>(R),
        static_cast<double>(E0)
    ); 

    // Compute forces via finite differences 
    forces2 = cellSurfaceRepulsionForceHorizontal(rz, half_l, R, E0, delta); 
    REQUIRE_THAT(
        forces1[2],
        Catch::Matchers::WithinAbs(static_cast<double>(forces2(2)), tol)
    );
    for (int i = 3; i < 6; ++i)
        REQUIRE_THAT(forces1[i], Catch::Matchers::WithinAbs(0.0, tol)); 

    // Case 3: Assume the cell does not contact the surface
    max_overlap = -0.1 * R; 
    rz = R + half_l * nz - max_overlap;
    r[2] = static_cast<double>(rz); 

    // Compute forces via cellSurfaceRepulsionForce()
    forces1 = cellSurfaceRepulsionForce(
        r, n, static_cast<double>(half_l), static_cast<double>(R),
        static_cast<double>(E0)
    );

    // Compute forces via finite differences 
    forces2 = cellSurfaceRepulsionForceHorizontal(rz, half_l, R, E0, delta);
    REQUIRE_THAT(
        forces1[2],
        Catch::Matchers::WithinAbs(static_cast<double>(forces2(2)), tol)
    );
    REQUIRE_THAT(forces1[2], Catch::Matchers::WithinAbs(0.0, tol));
    for (int i = 3; i < 6; ++i)
        REQUIRE_THAT(forces1[i], Catch::Matchers::WithinAbs(0.0, tol)); 

    // --------------------------------------------------------------- //
    //                          SKEW TEST CASES                        //
    // --------------------------------------------------------------- //
    Array<T, 10, 1> angles = Array<T, 10, 1>::Zero();
    for (int i = 0; i < angles.size(); ++i)
        angles(i) = boost::math::constants::half_pi<T>() * (i + 1) / 10.0; 

    // For each angle ... 
    for (int i = 0; i < angles.size(); ++i)
    {
        // Define the z-orientation
        nz = sin(angles(i));
        n[2] = static_cast<double>(nz); 
        n[0] = sqrt(1.0 - n[2] * n[2]);
        n2 << sqrt(1.0 - nz * nz), 0.0, nz; 

        // Case 1: Assume the cell has a maximum overlap of 0.04 * R
        max_overlap = 0.04 * R;
        rz = R + half_l * nz - max_overlap;
        r[2] = static_cast<double>(rz); 

        // Compute forces via cellSurfaceRepulsionForce()
        forces1 = cellSurfaceRepulsionForce(
            r, n, static_cast<double>(half_l), static_cast<double>(R),
            static_cast<double>(E0)
        ); 

        // Compute forces via finite differences 
        forces2 = cellSurfaceRepulsionForceFiniteDiff(rz, n2, half_l, R, E0, delta);
        for (int j = 0; j < 6; ++j)
            REQUIRE_THAT(
                forces1[j],
                Catch::Matchers::WithinAbs(static_cast<double>(forces2(j)), tol)
            );

        // Case 2: Assume the cell has a maximum overlap of 0.2 * R
        max_overlap = 0.2 * R;
        rz = R + half_l * nz - max_overlap;
        r[2] = static_cast<double>(rz); 

        // Compute forces via cellSurfaceRepulsionForce()
        forces1 = cellSurfaceRepulsionForce(
            r, n, static_cast<double>(half_l), static_cast<double>(R),
            static_cast<double>(E0)
        ); 

        // Compute forces via finite differences 
        forces2 = cellSurfaceRepulsionForceFiniteDiff(rz, n2, half_l, R, E0, delta); 
        for (int j = 0; j < 6; ++j)
            REQUIRE_THAT(
                forces1[j],
                Catch::Matchers::WithinAbs(static_cast<double>(forces2(j)), tol)
            );

        // Case 3: Assume the cell does not contact the surface
        max_overlap = -0.1 * R; 
        rz = R + half_l * nz - max_overlap;
        r[2] = static_cast<double>(rz); 

        // Compute forces via cellSurfaceRepulsionForce()
        forces1 = cellSurfaceRepulsionForce(
            r, n, static_cast<double>(half_l), static_cast<double>(R),
            static_cast<double>(E0)
        ); 

        // Compute forces via finite differences 
        forces2 = cellSurfaceRepulsionForceFiniteDiff(rz, n2, half_l, R, E0, delta); 
        for (int j = 0; j < 6; ++j)
        {
            REQUIRE_THAT(
                forces1[j],
                Catch::Matchers::WithinAbs(static_cast<double>(forces2(j)), tol)
            );
            REQUIRE_THAT(forces1[j], Catch::Matchers::WithinAbs(0.0, tol));
        } 
    }
}

/**
 * A series of tests for cellSurfaceAdhesionForce().
 */
TEST_CASE("Tests for cell-surface adhesion forces", "[cellSurfaceAdhesionForces()]")
{
    const T R = 0.8;
    const T sigma0 = 100.0;
    const T delta = 1e-50;
    const double tol = 1e-8;   // Since one set of calculations is done with doubles
    std::vector<double> r {0, 0, 0};
    std::vector<double> n {0, 0, 0};
    T rz, nz, max_overlap;
    Matrix<T, 3, 1> n2; 
    T half_l = 0.5;
    std::vector<double> forces1; 
    Array<T, 6, 1> forces2;  

    // --------------------------------------------------------------- //
    //                      HORIZONTAL TEST CASES                      //
    // --------------------------------------------------------------- //
    nz = 0.0;
    n[0] = 1.0; 
    n[2] = 0.0;

    // Case 1: Assume the cell has a maximum overlap of 0.04 * R
    max_overlap = 0.04 * R;
    rz = R + half_l * nz - max_overlap;
    r[2] = static_cast<double>(rz); 

    // Compute forces via cellSurfaceAdhesionForce()
    forces1 = cellSurfaceAdhesionForce(
        r, n, static_cast<double>(half_l), static_cast<double>(R),
        static_cast<double>(sigma0)
    ); 

    // Compute forces via finite differences 
    forces2 = cellSurfaceAdhesionForceHorizontal(rz, half_l, R, sigma0, delta);
    REQUIRE_THAT(
        forces1[2],
        Catch::Matchers::WithinAbs(static_cast<double>(forces2(2)), tol)
    );
    for (int i = 3; i < 6; ++i)
        REQUIRE_THAT(forces1[i], Catch::Matchers::WithinAbs(0.0, tol)); 

    // Case 2: Assume the cell has a maximum overlap of 0.2 * R
    max_overlap = 0.2 * R;
    rz = R + half_l * nz - max_overlap;
    r[2] = static_cast<double>(rz); 

    // Compute forces via cellSurfaceAdhesionForce()
    forces1 = cellSurfaceAdhesionForce(
        r, n, static_cast<double>(half_l), static_cast<double>(R),
        static_cast<double>(sigma0)
    ); 

    // Compute forces via finite differences 
    forces2 = cellSurfaceAdhesionForceHorizontal(rz, half_l, R, sigma0, delta);
    REQUIRE_THAT(
        forces1[2],
        Catch::Matchers::WithinAbs(static_cast<double>(forces2(2)), tol)
    );
    for (int i = 3; i < 6; ++i)
        REQUIRE_THAT(forces1[i], Catch::Matchers::WithinAbs(0.0, tol)); 

    // Case 3: Assume the cell does not contact the surface
    max_overlap = -0.1 * R; 
    rz = R + half_l * nz - max_overlap;
    r[2] = static_cast<double>(rz); 

    // Compute forces via cellSurfaceAdhesionForce()
    forces1 = cellSurfaceAdhesionForce(
        r, n, static_cast<double>(half_l), static_cast<double>(R),
        static_cast<double>(sigma0)
    ); 

    // Compute forces via finite differences 
    forces2 = cellSurfaceAdhesionForceHorizontal(rz, half_l, R, sigma0, delta);
    REQUIRE_THAT(
        forces1[2],
        Catch::Matchers::WithinAbs(static_cast<double>(forces2(2)), tol)
    );
    REQUIRE_THAT(forces1[2], Catch::Matchers::WithinAbs(0.0, tol));
    for (int i = 3; i < 6; ++i)
        REQUIRE_THAT(forces1[i], Catch::Matchers::WithinAbs(0.0, tol)); 

    // --------------------------------------------------------------- //
    //                          SKEW TEST CASES                        //
    // --------------------------------------------------------------- //
    Array<T, 10, 1> angles = Array<T, 10, 1>::Zero();
    for (int i = 0; i < angles.size(); ++i)
        angles(i) = boost::math::constants::half_pi<T>() * (i + 1) / 10.0; 

    // For each angle ... 
    for (int i = 0; i < angles.size(); ++i)
    {
        // Define the z-orientation
        nz = sin(angles(i));
        n[2] = static_cast<double>(nz); 
        n[0] = sqrt(1.0 - n[2] * n[2]);
        n2 << sqrt(1.0 - nz * nz), 0.0, nz; 

        // Case 1: Assume the cell has a maximum overlap of 0.04 * R
        max_overlap = 0.04 * R;
        rz = R + half_l * nz - max_overlap;
        r[2] = static_cast<double>(rz); 

        // Compute forces via cellSurfaceAdhesionForce()
        forces1 = cellSurfaceAdhesionForce(
            r, n, static_cast<double>(half_l), static_cast<double>(R),
            static_cast<double>(sigma0)
        ); 

        // Compute forces via finite differences 
        forces2 = cellSurfaceAdhesionForceFiniteDiff(rz, n2, half_l, R, sigma0, delta);
        for (int j = 0; j < 6; ++j)
            REQUIRE_THAT(
                forces1[j],
                Catch::Matchers::WithinAbs(static_cast<double>(forces2(j)), tol)
            );

        // Case 2: Assume the cell has a maximum overlap of 0.2 * R
        max_overlap = 0.2 * R;
        rz = R + half_l * nz - max_overlap;
        r[2] = static_cast<double>(rz); 

        // Compute forces via cellSurfaceAdhesionForce()
        forces1 = cellSurfaceAdhesionForce(
            r, n, static_cast<double>(half_l), static_cast<double>(R),
            static_cast<double>(sigma0)
        ); 

        // Compute forces via finite differences 
        forces2 = cellSurfaceAdhesionForceFiniteDiff(rz, n2, half_l, R, sigma0, delta); 
        for (int j = 0; j < 6; ++j)
            REQUIRE_THAT(
                forces1[j],
                Catch::Matchers::WithinAbs(static_cast<double>(forces2(j)), tol)
            );

        // Case 3: Assume the cell does not contact the surface
        max_overlap = -0.1 * R; 
        rz = R + half_l * nz - max_overlap;
        r[2] = static_cast<double>(rz); 

        // Compute forces via cellSurfaceAdhesionForce()
        forces1 = cellSurfaceAdhesionForce(
            r, n, static_cast<double>(half_l), static_cast<double>(R),
            static_cast<double>(sigma0)
        ); 

        // Compute forces via finite differences 
        forces2 = cellSurfaceAdhesionForceFiniteDiff(rz, n2, half_l, R, sigma0, delta); 
        for (int j = 0; j < 6; ++j)
        {
            REQUIRE_THAT(
                forces1[j],
                Catch::Matchers::WithinAbs(static_cast<double>(forces2(j)), tol)
            );
            REQUIRE_THAT(forces1[j], Catch::Matchers::WithinAbs(0.0, tol));
        } 
    }
}

/**
 * A series of tests for cellSurfaceFrictionForce().
 */
TEST_CASE("Tests for cell-surface friction forces", "[cellSurfaceFrictionForces()]")
{
    const T R = 0.8;
    const T eta = 720.0;
    const T delta = 1e-50;
    const double tol = 1e-8;   // Since one set of calculations is done with doubles
    std::vector<double> r {0, 0, 0};
    std::vector<double> n {0, 0, 0};
    std::vector<double> dr {0, 0, 0}; 
    std::vector<double> omega {0, 0, 0}; 
    T rz, nz, max_overlap;
    Matrix<T, 3, 1> n2, dr2, omega2; 
    T half_l = 0.5;
    std::vector<double> forces1; 
    Array<T, 6, 1> forces2;  

    // --------------------------------------------------------------- //
    //                      HORIZONTAL TEST CASES                      //
    // --------------------------------------------------------------- //
    nz = 0.0;
    n[0] = 1.0; 
    n[2] = 0.0;
    n2 << 1.0, 0.0, 0.0; 

    // Case 1: Assume the cell has a maximum overlap of 0.04 * R
    max_overlap = 0.04 * R;
    rz = R + half_l * nz - max_overlap;
    r[2] = static_cast<double>(rz);

    // Case 1a: Assume the cell is moving in the x- and y-directions
    dr[0] = 1.0; 
    dr[1] = 1.0;
    dr[2] = 0.0;
    omega[0] = 0.0;
    omega[1] = 0.0; 
    omega[2] = 1.0;
    dr2 << 1.0, 1.0, 0.0; 
    omega2 << 0.0, 0.0, 1.0;

    // Compute forces via cellSurfaceFrictionForce()
    forces1 = cellSurfaceFrictionForce(
        r, n, static_cast<double>(half_l), dr, omega, static_cast<double>(R),
        static_cast<double>(eta)
    ); 

    // Compute forces via finite differences 
    forces2 = cellSurfaceFrictionForceHorizontal(
        rz, n2, dr2, omega2, half_l, R, eta, delta
    );
    for (int i = 0; i < 6; ++i) 
        REQUIRE_THAT(
            forces1[i],
            Catch::Matchers::WithinAbs(static_cast<double>(forces2(i)), tol)
        );

    // Case 1b: Assume the cell is moving in all three directions 
    dr[0] = 0.2; 
    dr[1] = 0.3;
    dr[2] = 0.5;
    omega[0] = 0.1; 
    omega[1] = 0.2;  
    omega[2] = 0.3;
    dr2 << 0.2, 0.3, 0.5;
    omega2 << 0.1, 0.2, 0.3;

    // Compute forces via cellSurfaceFrictionForce()
    forces1 = cellSurfaceFrictionForce(
        r, n, static_cast<double>(half_l), dr, omega, static_cast<double>(R),
        static_cast<double>(eta)
    ); 

    // Compute forces via finite differences 
    forces2 = cellSurfaceFrictionForceHorizontal(
        rz, n2, dr2, omega2, half_l, R, eta, delta
    );
    for (int i = 0; i < 6; ++i) 
        REQUIRE_THAT(
            forces1[i],
            Catch::Matchers::WithinAbs(static_cast<double>(forces2(i)), tol)
        );

    // Case 2: Assume the cell has a maximum overlap of 0.2 * R
    max_overlap = 0.2 * R;
    rz = R + half_l * nz - max_overlap;
    r[2] = static_cast<double>(rz);

    // Case 2a: Assume the cell is moving in the x- and y-directions
    dr[0] = 1.0; 
    dr[1] = 1.0;
    dr[2] = 0.0;
    omega[0] = 0.0;
    omega[1] = 0.0; 
    omega[2] = 1.0;
    dr2 << 1.0, 1.0, 0.0; 
    omega2 << 0.0, 0.0, 1.0;

    // Compute forces via cellSurfaceFrictionForce()
    forces1 = cellSurfaceFrictionForce(
        r, n, static_cast<double>(half_l), dr, omega, static_cast<double>(R),
        static_cast<double>(eta)
    ); 

    // Compute forces via finite differences 
    forces2 = cellSurfaceFrictionForceHorizontal(
        rz, n2, dr2, omega2, half_l, R, eta, delta
    );
    for (int i = 0; i < 6; ++i) 
        REQUIRE_THAT(
            forces1[i],
            Catch::Matchers::WithinAbs(static_cast<double>(forces2(i)), tol)
        );

    // Case 2b: Assume the cell is moving in all three directions 
    dr[0] = 0.2; 
    dr[1] = 0.3;
    dr[2] = 0.5;
    omega[0] = 0.1; 
    omega[1] = 0.2;  
    omega[2] = 0.3;
    dr2 << 0.2, 0.3, 0.5;
    omega2 << 0.1, 0.2, 0.3;

    // Compute forces via cellSurfaceFrictionForce()
    forces1 = cellSurfaceFrictionForce(
        r, n, static_cast<double>(half_l), dr, omega, static_cast<double>(R),
        static_cast<double>(eta)
    ); 

    // Compute forces via finite differences 
    forces2 = cellSurfaceFrictionForceHorizontal(
        rz, n2, dr2, omega2, half_l, R, eta, delta
    );
    for (int i = 0; i < 6; ++i) 
        REQUIRE_THAT(
            forces1[i],
            Catch::Matchers::WithinAbs(static_cast<double>(forces2(i)), tol)
        );

    // Case 3: Assume the cell does not contact the surface
    max_overlap = -0.1 * R; 
    rz = R + half_l * nz - max_overlap;
    r[2] = static_cast<double>(rz); 

    // Case 3a: Assume the cell is moving in the x- and y-directions
    dr[0] = 1.0; 
    dr[1] = 1.0;
    dr[2] = 0.0;
    omega[0] = 0.0;
    omega[1] = 0.0; 
    omega[2] = 1.0;
    dr2 << 1.0, 1.0, 0.0; 
    omega2 << 0.0, 0.0, 1.0;

    // Compute forces via cellSurfaceFrictionForce()
    forces1 = cellSurfaceFrictionForce(
        r, n, static_cast<double>(half_l), dr, omega, static_cast<double>(R),
        static_cast<double>(eta)
    ); 

    // Compute forces via finite differences 
    forces2 = cellSurfaceFrictionForceHorizontal(
        rz, n2, dr2, omega2, half_l, R, eta, delta
    );
    for (int i = 0; i < 6; ++i)
    { 
        REQUIRE_THAT(
            forces1[i],
            Catch::Matchers::WithinAbs(static_cast<double>(forces2(i)), tol)
        );
        REQUIRE_THAT(forces1[i], Catch::Matchers::WithinAbs(0.0, tol));
    }

    // Case 3b: Assume the cell is moving in all three directions 
    dr[0] = 0.2; 
    dr[1] = 0.3;
    dr[2] = 0.5;
    omega[0] = 0.1; 
    omega[1] = 0.2;  
    omega[2] = 0.3;
    dr2 << 0.2, 0.3, 0.5;
    omega2 << 0.1, 0.2, 0.3;

    // Compute forces via cellSurfaceFrictionForce()
    forces1 = cellSurfaceFrictionForce(
        r, n, static_cast<double>(half_l), dr, omega, static_cast<double>(R),
        static_cast<double>(eta)
    ); 

    // Compute forces via finite differences 
    forces2 = cellSurfaceFrictionForceHorizontal(
        rz, n2, dr2, omega2, half_l, R, eta, delta
    );
    for (int i = 0; i < 6; ++i) 
        REQUIRE_THAT(
            forces1[i],
            Catch::Matchers::WithinAbs(static_cast<double>(forces2(i)), tol)
        );

    // --------------------------------------------------------------- //
    //                          SKEW TEST CASES                        //
    // --------------------------------------------------------------- //
    Array<T, 10, 1> angles = Array<T, 10, 1>::Zero();
    for (int i = 0; i < angles.size(); ++i)
        angles(i) = boost::math::constants::half_pi<T>() * (i + 1) / 10.0; 

    // For each angle ... 
    for (int i = 0; i < angles.size(); ++i)
    {
        // Define the z-orientation
        nz = sin(angles(i));
        n[2] = static_cast<double>(nz); 
        n[0] = sqrt(1.0 - n[2] * n[2]);
        n2 << sqrt(1.0 - nz * nz), 0.0, nz; 

        // Case 1: Assume the cell has a maximum overlap of 0.04 * R
        max_overlap = 0.04 * R;
        rz = R + half_l * nz - max_overlap;
        r[2] = static_cast<double>(rz); 

        // Case 1a: Assume the cell is moving in the x- and y-directions
        dr[0] = 1.0; 
        dr[1] = 1.0;
        dr[2] = 0.0;
        omega[0] = 0.0;
        omega[1] = 0.0; 
        omega[2] = 1.0;
        dr2 << 1.0, 1.0, 0.0; 
        omega2 << 0.0, 0.0, 1.0;

        // Compute forces via cellSurfaceFrictionForce()
        forces1 = cellSurfaceFrictionForce(
            r, n, static_cast<double>(half_l), dr, omega, static_cast<double>(R),
            static_cast<double>(eta)
        );

        // Compute forces via finite differences 
        forces2 = cellSurfaceFrictionForceFiniteDiff(
            rz, n2, dr2, omega2, half_l, R, eta, delta
        );
        for (int j = 0; j < 6; ++j)
            REQUIRE_THAT(
                forces1[j],
                Catch::Matchers::WithinAbs(static_cast<double>(forces2(j)), tol)
            );

        // Case 1b: Assume the cell is moving in all three directions 
        dr[0] = 0.2; 
        dr[1] = 0.3;
        dr[2] = 0.5;
        omega[0] = 0.1; 
        omega[1] = 0.2;  
        omega[2] = 0.3;
        dr2 << 0.2, 0.3, 0.5;
        omega2 << 0.1, 0.2, 0.3;

        // Compute forces via cellSurfaceFrictionForce()
        forces1 = cellSurfaceFrictionForce(
            r, n, static_cast<double>(half_l), dr, omega, static_cast<double>(R),
            static_cast<double>(eta)
        ); 

        // Compute forces via finite differences 
        forces2 = cellSurfaceFrictionForceFiniteDiff(
            rz, n2, dr2, omega2, half_l, R, eta, delta
        );
        for (int j = 0; j < 6; ++j) 
            REQUIRE_THAT(
                forces1[j],
                Catch::Matchers::WithinAbs(static_cast<double>(forces2(j)), tol)
            );

        // Case 2: Assume the cell has a maximum overlap of 0.2 * R
        max_overlap = 0.2 * R;
        rz = R + half_l * nz - max_overlap;
        r[2] = static_cast<double>(rz); 

        // Case 2a: Assume the cell is moving in the x- and y-directions
        dr[0] = 1.0; 
        dr[1] = 1.0;
        dr[2] = 0.0;
        omega[0] = 0.0;
        omega[1] = 0.0; 
        omega[2] = 1.0;
        dr2 << 1.0, 1.0, 0.0; 
        omega2 << 0.0, 0.0, 1.0;

        // Compute forces via cellSurfaceFrictionForce()
        forces1 = cellSurfaceFrictionForce(
            r, n, static_cast<double>(half_l), dr, omega, static_cast<double>(R),
            static_cast<double>(eta)
        );

        // Compute forces via finite differences 
        forces2 = cellSurfaceFrictionForceFiniteDiff(
            rz, n2, dr2, omega2, half_l, R, eta, delta
        );
        for (int j = 0; j < 6; ++j)
            REQUIRE_THAT(
                forces1[j],
                Catch::Matchers::WithinAbs(static_cast<double>(forces2(j)), tol)
            );

        // Case 2b: Assume the cell is moving in all three directions 
        dr[0] = 0.2; 
        dr[1] = 0.3;
        dr[2] = 0.5;
        omega[0] = 0.1; 
        omega[1] = 0.2;  
        omega[2] = 0.3;
        dr2 << 0.2, 0.3, 0.5;
        omega2 << 0.1, 0.2, 0.3;

        // Compute forces via cellSurfaceFrictionForce()
        forces1 = cellSurfaceFrictionForce(
            r, n, static_cast<double>(half_l), dr, omega, static_cast<double>(R),
            static_cast<double>(eta)
        ); 

        // Compute forces via finite differences 
        forces2 = cellSurfaceFrictionForceFiniteDiff(
            rz, n2, dr2, omega2, half_l, R, eta, delta
        );
        for (int j = 0; j < 6; ++j) 
            REQUIRE_THAT(
                forces1[j],
                Catch::Matchers::WithinAbs(static_cast<double>(forces2(j)), tol)
            );

        // Case 3: Assume the cell does not contact the surface
        max_overlap = -0.1 * R; 
        rz = R + half_l * nz - max_overlap;
        r[2] = static_cast<double>(rz); 

        // Case 3a: Assume the cell is moving in the x- and y-directions
        dr[0] = 1.0; 
        dr[1] = 1.0;
        dr[2] = 0.0;
        omega[0] = 0.0;
        omega[1] = 0.0; 
        omega[2] = 1.0;
        dr2 << 1.0, 1.0, 0.0; 
        omega2 << 0.0, 0.0, 1.0;

        // Compute forces via cellSurfaceFrictionForce()
        forces1 = cellSurfaceFrictionForce(
            r, n, static_cast<double>(half_l), dr, omega, static_cast<double>(R),
            static_cast<double>(eta)
        );

        // Compute forces via finite differences 
        forces2 = cellSurfaceFrictionForceFiniteDiff(
            rz, n2, dr2, omega2, half_l, R, eta, delta
        );
        for (int j = 0; j < 6; ++j)
            REQUIRE_THAT(
                forces1[j],
                Catch::Matchers::WithinAbs(static_cast<double>(forces2(j)), tol)
            );

        // Case 3b: Assume the cell is moving in all three directions 
        dr[0] = 0.2; 
        dr[1] = 0.3;
        dr[2] = 0.5;
        omega[0] = 0.1; 
        omega[1] = 0.2;  
        omega[2] = 0.3;
        dr2 << 0.2, 0.3, 0.5;
        omega2 << 0.1, 0.2, 0.3;

        // Compute forces via cellSurfaceFrictionForce()
        forces1 = cellSurfaceFrictionForce(
            r, n, static_cast<double>(half_l), dr, omega, static_cast<double>(R),
            static_cast<double>(eta)
        ); 

        // Compute forces via finite differences 
        forces2 = cellSurfaceFrictionForceFiniteDiff(
            rz, n2, dr2, omega2, half_l, R, eta, delta
        );
        for (int j = 0; j < 6; ++j) 
            REQUIRE_THAT(
                forces1[j],
                Catch::Matchers::WithinAbs(static_cast<double>(forces2(j)), tol)
            );
    }
}

