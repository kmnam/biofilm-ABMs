/**
 * Test module for the `cellCellRepulsiveForces()` and `cellCellRepulsiveForcesNewton()`
 * functions. 
 *
 * Authors:
 *     Kee-Myoung Nam
 *
 * Last updated:
 *     3/12/2025
 */
#include <cmath>
#include <Eigen/Dense>
#include <boost/math/constants/constants.hpp>
#include <boost/multiprecision/mpfr.hpp>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Segment_3.h>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "../../include/distances.hpp"
#include "../../include/adhesion.hpp"
#include "../../include/mechanics.hpp"
#include "../../include/indices.hpp"

using namespace Eigen;

using std::sin; 
using boost::multiprecision::sin; 
using std::cos; 
using boost::multiprecision::cos; 
using std::sqrt; 
using boost::multiprecision::sqrt;
using std::pow; 
using boost::multiprecision::pow; 

typedef double T; 
typedef CGAL::Exact_predicates_inexact_constructions_kernel K; 
typedef K::Segment_3 Segment_3;

/* ------------------------------------------------------------------ //
 *                           HELPER FUNCTIONS                         //
 * ------------------------------------------------------------------ */
/**
 * Returns the Hertzian potential energy between two cells.
 *
 * @param r1 Cell 1 center. 
 * @param n1 Cell 1 orientation. 
 * @param half_l1 Cell 1 half-length. 
 * @param r2 Cell 2 center. 
 * @param n2 Cell 2 orientation. 
 * @param half_l2 Cell 2 half-length. 
 * @param R Cell radius, including the EPS. 
 * @param Rcell Cell radius, excluding the EPS. 
 * @param E0 Elastic modulus of EPS. 
 * @param Ecell Elastic modulus of cell body.
 * @returns Hertzian potential energy.  
 */
T potentialHertz(const Ref<const Array<T, 2, 1> >& r1,
                 const Ref<const Array<T, 2, 1> >& n1, const T half_l1,
                 const Ref<const Array<T, 2, 1> >& r2, 
                 const Ref<const Array<T, 2, 1> >& n2, const T half_l2,
                 const T R, const T Rcell, const T E0, const T Ecell)
{
    // Compute the distance vector from cell 1 to cell 2 
    K kernel; 
    Segment_3 seg1 = generateSegment<T>(r1, n1, half_l1); 
    Segment_3 seg2 = generateSegment<T>(r2, n2, half_l2);
    auto result = distBetweenCells<T>(seg1, seg2, 0, r1, n1, half_l1, 1, r2, n2, half_l2, kernel);
    Matrix<T, 2, 1> d12 = std::get<0>(result); 
    T dist = d12.norm();
    return potentialHertz<T>(dist, R, Rcell, E0, Ecell);  
}

/**
 * Returns the generalized Hertzian forces and torques on the given two cells,
 * computed via finite difference approximation from the potential energy.  
 *
 * @param r1 Cell 1 center. 
 * @param n1 Cell 1 orientation. 
 * @param half_l1 Cell 1 half-length. 
 * @param r2 Cell 2 center. 
 * @param n2 Cell 2 orientation. 
 * @param half_l2 Cell 2 half-length. 
 * @param R Cell radius, including the EPS. 
 * @param Rcell Cell radius, excluding the EPS. 
 * @param E0 Elastic modulus of EPS. 
 * @param Ecell Elastic modulus of cell body.
 * @param delta Increment for finite difference approximation. 
 * @returns Array of generalized forces and torques on the two cells. 
 */
Array<T, 2, 4> cellCellRepulsiveForcesFiniteDiff(const Ref<const Array<T, 2, 1> >& r1,
                                                 const Ref<const Array<T, 2, 1> >& n1,
                                                 const T half_l1,
                                                 const Ref<const Array<T, 2, 1> >& r2, 
                                                 const Ref<const Array<T, 2, 1> >& n2,
                                                 const T half_l2, const T R,
                                                 const T Rcell, const T E0,
                                                 const T Ecell, const T delta)
{
    Array<T, 2, 4> dEdq = Array<T, 2, 4>::Zero();
    Array<T, 2, 1> dx, dy; 
    dx << delta, 0; 
    dy << 0, delta;  

    // Compute all eight finite differences ... 
    //
    // 1) Partial derivatives w.r.t r1
    dEdq(0, 0) = (
        potentialHertz(r1 + dx, n1, half_l1, r2, n2, half_l2, R, Rcell, E0, Ecell) -
        potentialHertz(r1 - dx, n1, half_l1, r2, n2, half_l2, R, Rcell, E0, Ecell)
    );
    dEdq(0, 1) = (
        potentialHertz(r1 + dy, n1, half_l1, r2, n2, half_l2, R, Rcell, E0, Ecell) -
        potentialHertz(r1 - dy, n1, half_l1, r2, n2, half_l2, R, Rcell, E0, Ecell)
    );

    // 2) Partial derivatives w.r.t n1
    dEdq(0, 2) = (
        potentialHertz(r1, n1 + dx, half_l1, r2, n2, half_l2, R, Rcell, E0, Ecell) -
        potentialHertz(r1, n1 - dx, half_l1, r2, n2, half_l2, R, Rcell, E0, Ecell)
    );
    dEdq(0, 3) = (
        potentialHertz(r1, n1 + dy, half_l1, r2, n2, half_l2, R, Rcell, E0, Ecell) -
        potentialHertz(r1, n1 - dy, half_l1, r2, n2, half_l2, R, Rcell, E0, Ecell)
    );
    
    // 3) Partial derivatives w.r.t r2
    dEdq(1, 0) = (
        potentialHertz(r1, n1, half_l1, r2 + dx, n2, half_l2, R, Rcell, E0, Ecell) -
        potentialHertz(r1, n1, half_l1, r2 - dx, n2, half_l2, R, Rcell, E0, Ecell)
    );
    dEdq(1, 1) = (
        potentialHertz(r1, n1, half_l1, r2 + dy, n2, half_l2, R, Rcell, E0, Ecell) -
        potentialHertz(r1, n1, half_l1, r2 - dy, n2, half_l2, R, Rcell, E0, Ecell)
    );

    // 4) Partial derivatives w.r.t n2
    dEdq(1, 2) = (
        potentialHertz(r1, n1, half_l1, r2, n2 + dx, half_l2, R, Rcell, E0, Ecell) -
        potentialHertz(r1, n1, half_l1, r2, n2 - dx, half_l2, R, Rcell, E0, Ecell)
    );
    dEdq(1, 3) = (
        potentialHertz(r1, n1, half_l1, r2, n2 + dy, half_l2, R, Rcell, E0, Ecell) -
        potentialHertz(r1, n1, half_l1, r2, n2 - dy, half_l2, R, Rcell, E0, Ecell)
    );

    // Normalize by double the increment and return 
    return dEdq / (2 * delta); 
}

/* ------------------------------------------------------------------ //
 *                       GENERIC TEST FUNCTIONS                       //
 * ------------------------------------------------------------------ */
/**
 * A generic test function for cellCellRepulsiveForces().
 *
 * @param r1 Cell 1 center. 
 * @param n1 Cell 1 orientation. 
 * @param half_l1 Cell 1 half-length. 
 * @param r2 Cell 2 center. 
 * @param n2 Cell 2 orientation. 
 * @param half_l2 Cell 2 half-length. 
 * @param R Cell radius, including the EPS. 
 * @param Rcell Cell radius, excluding the EPS. 
 * @param E0 Elastic modulus of EPS. 
 * @param Ecell Elastic modulus of cell body.
 * @param delta Increment for finite difference approximation. 
 * @param target_force_21 Pre-computed force vector on cell 1 due to cell 2. 
 */
void testCellCellRepulsiveForces(const Ref<const Array<T, 2, 1> >& r1,
                                 const Ref<const Array<T, 2, 1> >& n1,
                                 const T half_l1,
                                 const Ref<const Array<T, 2, 1> >& r2, 
                                 const Ref<const Array<T, 2, 1> >& n2,
                                 const T half_l2, const T R, const T Rcell,
                                 const T E0, const T Ecell, const T delta,
                                 const Ref<const Array<T, 2, 1> >& target_force_21)
{
    // Compute the distance vector from cell 1 to cell 2 
    K kernel; 
    Segment_3 seg1 = generateSegment<T>(r1, n1, half_l1); 
    Segment_3 seg2 = generateSegment<T>(r2, n2, half_l2);
    auto result = distBetweenCells<T>(seg1, seg2, 0, r1, n1, half_l1, 1, r2, n2, half_l2, kernel);
    Matrix<T, 2, 1> d12 = std::get<0>(result); 
    T s = std::get<1>(result); 
    T t = std::get<2>(result);
    T dist = d12.norm();
    T threshold;
    if (dist < R + Rcell)
        threshold = delta * Ecell; 
    else if (dist < 2 * R)
        threshold = delta * E0; 
    else 
        threshold = delta; 

    // Prepare the arrays to be passed into cellCellRepulsiveForces() 
    Array<T, Dynamic, Dynamic> cells(2, __ncols_required);
    cells << 0, r1(0), r1(1), n1(0), n1(1), 0, 0, 0, 0, 2 * half_l1, half_l1, 0, 1, 1, 1, 1, 1, 
             1, r2(0), r2(1), n2(0), n2(1), 0, 0, 0, 0, 2 * half_l2, half_l2, 0, 1, 1, 1, 1, 1;
    Array<T, Dynamic, 6> neighbors(1, 6);
    neighbors << 0, 1, d12(0), d12(1), s, t;
    Array<T, 3, 1> prefactors; 
    prefactors << 2.5 * E0 * sqrt(R), 
                  2.5 * E0 * sqrt(R) * pow(2 * (R - Rcell), 1.5), 
                  2.5 * Ecell * sqrt(R);  

    // Compute the forces via cellCellRepulsiveForces()
    //
    // Note that these forces are computed as formulated by You et al. (2018, 2019, 2021) 
    Array<T, 2, 4> forces1 = -cellCellRepulsiveForces<T>(
        cells, neighbors, 1e-6, 0, R, Rcell, prefactors
    );

    // Check that the force on cell 1 due to cell 2 is correct 
    REQUIRE_THAT(
        (forces1(0, Eigen::seq(0, 1)) - target_force_21.transpose()).matrix().norm(),
        Catch::Matchers::WithinAbs(0.0, threshold)
    );

    // Check that the torque on cell 1 due to cell 2, inferred from the given
    // force, is correct
    //
    // Following You et al., this torque should be (s * n1).cross(force).cross(n1)
    Matrix<T, 3, 1> u, v;
    u << n1(0), n1(1), 0; 
    v << target_force_21(0), target_force_21(1), 0;
    Array<T, 2, 1> target_torque_21 = (s * u).cross(v).cross(u)(Eigen::seq(0, 1)).array();
    REQUIRE_THAT(
        (forces1(0, Eigen::seq(2, 3)) - target_torque_21.transpose()).matrix().norm(),
        Catch::Matchers::WithinAbs(0.0, threshold)
    ); 

    // Check that the two force vectors are equal and opposite 
    REQUIRE_THAT(
        (forces1(0, Eigen::seq(0, 1)) + forces1(1, Eigen::seq(0, 1))).matrix().norm(),
        Catch::Matchers::WithinAbs(0.0, threshold)
    );

    // Check that the two torque vectors are zero, if either is expected 
    //
    // The torque on either cell should be zero if:
    // (1) the force is acting on the center of the cell and the force vector
    // is orthogonal to the cell orientation, or 
    // (2) the force is acting on either end of the cell and the force vector 
    // is parallel to the cell orientation
    Matrix<T, 2, 1> dnorm = d12 / dist;
    bool cond1 = (abs(s) < delta && abs(dnorm.dot(n1.matrix())) < delta); 
    bool cond2 = (half_l1 - abs(s) < delta && 1.0 - abs(dnorm.dot(n1.matrix())) < delta);
    bool cond3 = (abs(t) < delta && abs(dnorm.dot(n2.matrix())) < delta); 
    bool cond4 = (half_l2 - abs(t) < delta && 1.0 - abs(dnorm.dot(n2.matrix())) < delta);  
    if (cond1 || cond2)
        REQUIRE_THAT(forces1(0, Eigen::seq(2, 3)).matrix().norm(), Catch::Matchers::WithinAbs(0.0, threshold));
    if (cond3 || cond4)
        REQUIRE_THAT(forces1(1, Eigen::seq(2, 3)).matrix().norm(), Catch::Matchers::WithinAbs(0.0, threshold));         

    // Compute the forces via finite differences 
    Array<T, 2, 4> dEdq2 = cellCellRepulsiveForcesFiniteDiff(
        r1, n1, half_l1, r2, n2, half_l2, R, Rcell, E0, Ecell, delta
    );

    // Apply the unit vector constraint to each torque computed via finite 
    // differences 
    Array<T, 2, 4> forces2(-dEdq2); 
    T lambda1 = 0.5 * n1.matrix().dot(dEdq2(0, Eigen::seq(2, 3)).matrix()); 
    T lambda2 = 0.5 * n2.matrix().dot(dEdq2(1, Eigen::seq(2, 3)).matrix());
    forces2(0, Eigen::seq(2, 3)) += lambda1 * 2 * n1; 
    forces2(1, Eigen::seq(2, 3)) += lambda2 * 2 * n2;

    // Check that the constrained forces agree with each other
    REQUIRE_THAT((forces1 - forces2).abs().maxCoeff(), Catch::Matchers::WithinAbs(0.0, threshold)); 
}

/**
 * A generic test function for cellCellRepulsiveForcesNewton().
 *
 * This function compares the calculated forces against those obtained from 
 * cellCellRepulsiveForces(). 
 *
 * @param r1 Cell 1 center. 
 * @param n1 Cell 1 orientation. 
 * @param half_l1 Cell 1 half-length. 
 * @param r2 Cell 2 center. 
 * @param n2 Cell 2 orientation. 
 * @param half_l2 Cell 2 half-length. 
 * @param R Cell radius, including the EPS. 
 * @param Rcell Cell radius, excluding the EPS. 
 * @param E0 Elastic modulus of EPS. 
 * @param Ecell Elastic modulus of cell body.
 */
void testCellCellRepulsiveForcesNewton(const Ref<const Array<T, 2, 1> >& r1,
                                       const Ref<const Array<T, 2, 1> >& n1,
                                       const T half_l1,
                                       const Ref<const Array<T, 2, 1> >& r2, 
                                       const Ref<const Array<T, 2, 1> >& n2,
                                       const T half_l2, const T R, const T Rcell,
                                       const T E0, const T Ecell)
{
    // Compute the distance vector from cell 1 to cell 2 
    K kernel; 
    Segment_3 seg1 = generateSegment<T>(r1, n1, half_l1); 
    Segment_3 seg2 = generateSegment<T>(r2, n2, half_l2);
    auto result = distBetweenCells<T>(seg1, seg2, 0, r1, n1, half_l1, 1, r2, n2, half_l2, kernel);
    Matrix<T, 2, 1> d12 = std::get<0>(result); 
    T s = std::get<1>(result); 
    T t = std::get<2>(result);
    T dist = d12.norm();
    T threshold;
    if (dist < R + Rcell)
        threshold = 1e-8 * Ecell; 
    else if (dist < 2 * R)
        threshold = 1e-8 * E0; 
    else 
        threshold = 1e-8; 

    // Prepare the arrays to be passed into cellCellRepulsiveForces() 
    Array<T, Dynamic, Dynamic> cells(2, __ncols_required);
    cells << 0, r1(0), r1(1), n1(0), n1(1), 0, 0, 0, 0, 2 * half_l1, half_l1, 0, 1, 1, 1, 1, 1, 
             1, r2(0), r2(1), n2(0), n2(1), 0, 0, 0, 0, 2 * half_l2, half_l2, 0, 1, 1, 1, 1, 1;
    Array<T, Dynamic, 6> neighbors(1, 6);
    neighbors << 0, 1, d12(0), d12(1), s, t;
    Array<T, 3, 1> prefactors; 
    prefactors << 2.5 * E0 * sqrt(R), 
                  2.5 * E0 * sqrt(R) * pow(2 * (R - Rcell), 1.5), 
                  2.5 * Ecell * sqrt(R);  

    // Compute the forces via cellCellRepulsiveForcesNewton()
    Array<T, 2, 3> forces1 = cellCellRepulsiveForcesNewton<T>(
        cells, neighbors, 1e-6, 0, R, Rcell, prefactors
    );

    // Compute the forces via cellCellRepulsiveForces() 
    Array<T, 2, 4> forces2 = -cellCellRepulsiveForces<T>(
        cells, neighbors, 1e-6, 0, R, Rcell, prefactors
    );

    // Check that the force vectors match 
    REQUIRE_THAT(
        (forces1(0, Eigen::seq(0, 1)) - forces2(0, Eigen::seq(0, 1))).matrix().norm(), 
        Catch::Matchers::WithinAbs(0.0, threshold)
    );
    REQUIRE_THAT(
        (forces1(1, Eigen::seq(0, 1)) - forces2(1, Eigen::seq(0, 1))).matrix().norm(), 
        Catch::Matchers::WithinAbs(0.0, threshold)
    );

    // Calculate the torque vectors from the angular velocities and check that
    // they match the torque vectors calculated using cellCellRepulsiveForces() 
    Matrix<T, 3, 1> torque1, torque2, v1, v2, cross1, cross2; 
    torque1 << 0, 0, forces1(0, 2); 
    torque2 << 0, 0, forces1(1, 2);
    v1 << n1(0), n1(1), 0; 
    v2 << n2(0), n2(1), 0;
    cross1 = torque1.cross(v1.matrix()); 
    cross2 = torque2.cross(v2.matrix()); 
    REQUIRE_THAT(
        (cross1.head(2) - forces2(0, Eigen::seq(2, 3)).matrix().transpose()).norm(), 
        Catch::Matchers::WithinAbs(0.0, threshold)
    );
    REQUIRE_THAT(
        (cross2.head(2) - forces2(1, Eigen::seq(2, 3)).matrix().transpose()).norm(),
        Catch::Matchers::WithinAbs(0.0, threshold)
    );
}

/* ------------------------------------------------------------------ //
 *                            TEST MODULES                            //
 * ------------------------------------------------------------------ */
/**
 * A series of tests for cellCellRepulsiveForces() for skew cell-cell configurations. 
 */
TEST_CASE("Tests for cellCellRepulsiveForces(), skew cells", "[cellCellRepulsiveForces()]")
{
    const T R = 0.8;
    const T Rcell = 0.5; 
    const T E0 = 1.0;
    const T Ecell = 100.0;
    const T delta = 1e-8;  

    // r1 = (2, 1), n1 = (0.6, 0.8), l1 = 2
    // r2 = (0, 3), n2 = (0, 1), l2 = 1
    //
    // The shortest distance between the two cells is the vector (-2, 1.5),
    // which has norm 2.5 > 2 * R = 1.6, which means that the force between
    // these cells should be zero 
    Array<T, 2, 1> r1, n1, r2, n2; 
    r1 << 2, 1; 
    n1 << 0.6, 0.8; 
    r2 << 0, 3; 
    n2 << 0, 1;
    Array<T, 2, 1> target_force_21 = Array<T, 2, 1>::Zero();  
    testCellCellRepulsiveForces(
        r1, n1, 1, r2, n2, 0.5, R, Rcell, E0, Ecell, delta, target_force_21
    );

    // r1 = (0, 0), n1 = (1, 0), l1 = 1
    // r2 = (0.5 + 0.5 * cos(pi/6), 1 + 0.5 * sin(pi/6)), n2 = (cos(pi/6), sin(pi/6)), l2 = 1
    //
    // The shortest distance between the two cells is the vector (0, 1)
    r1(0) = 0; 
    r1(1) = 0; 
    n1(0) = 1; 
    n1(1) = 0; 
    r2(0) = 0.5 + 0.5 * cos(boost::math::constants::sixth_pi<T>()); 
    r2(1) = 1 + 0.5 * sin(boost::math::constants::sixth_pi<T>()); 
    n2(0) = cos(boost::math::constants::sixth_pi<T>()); 
    n2(1) = sin(boost::math::constants::sixth_pi<T>());
    Array<T, 2, 1> d12; 
    d12 << 0, 1;
    target_force_21 = -d12 * (
        2.5 * E0 * sqrt(R) * pow(2 * (R - Rcell), 1.5) +
        2.5 * Ecell * sqrt(Rcell) * pow(2 * Rcell - 1, 1.5)
    ); 
    testCellCellRepulsiveForces(
        r1, n1, 0.5, r2, n2, 0.5, R, Rcell, E0, Ecell, delta, target_force_21
    );

    // r1 = (0, 0), n1 = (1, 0), l1 = 1
    // r2 = (0.2 + 0.5 * cos(pi/6), 1 + 0.5 * sin(pi/6)), n2 = (cos(pi/6), sin(pi/6)), l2 = 1
    //
    // The shortest distance between the two cells is, again, (0, 1)
    r1(0) = 0; 
    r1(1) = 0; 
    n1(0) = 1; 
    n1(1) = 0; 
    r2(0) = 0.2 + 0.5 * cos(boost::math::constants::sixth_pi<T>()); 
    r2(1) = 1 + 0.5 * sin(boost::math::constants::sixth_pi<T>()); 
    n2(0) = cos(boost::math::constants::sixth_pi<T>()); 
    n2(1) = sin(boost::math::constants::sixth_pi<T>());
    testCellCellRepulsiveForces(
        r1, n1, 0.5, r2, n2, 0.5, R, Rcell, E0, Ecell, delta, target_force_21
    );

    // r1 = (0, 0), n1 = (1, 0), l1 = 1
    // r2 = (0.5 + 1.4 / sqrt(2), 1.4 / sqrt(2)), n2 = (1 / sqrt(2), -1 / sqrt(2)), l2 = 1
    //
    // The shortest distance between the two cells is (1.4 / sqrt(2), 1.4 / sqrt(2))
    //
    // The torque on cell 2 should be zero, since the force acts on the cell 
    // center (r2)
    r2(0) = 0.5 + 1.4 / sqrt(2.0); 
    r2(1) = 1.4 / sqrt(2.0); 
    n2(0) = 1.0 / sqrt(2.0); 
    n2(1) = -1.0 / sqrt(2.0);
    d12(0) = 1.4 / sqrt(2.0); 
    d12(1) = 1.4 / sqrt(2.0);
    T dist = d12.matrix().norm(); 
    Array<T, 2, 1> dnorm = d12 / dist; 
    target_force_21 = -dnorm * (2.5 * E0 * sqrt(R) * pow(2 * R - dist, 1.5));
    testCellCellRepulsiveForces(
        r1, n1, 0.5, r2, n2, 0.5, R, Rcell, E0, Ecell, delta, target_force_21
    ); 
}

/**
 * A series of tests for cellCellRepulsiveForces() for parallel cell-cell
 * configurations.
 */
TEST_CASE("Tests for cellCellRepulsiveForces(), parallel cells", "[cellCellRepulsiveForces()]")
{
    const T R = 0.8;
    const T Rcell = 0.5; 
    const T E0 = 1.0;
    const T Ecell = 100.0;
    const T delta = 1e-8;

    // r1 = (0, 0), n1 = (1, 0), l1 = 1
    // r2 = (2.4, 0), n2 = (1, 0), l2 = 1
    //
    // The shortest distance between the two cells is the vector (1.4, 0)
    // 
    // Since the cells are aligned, there should be zero torque
    Array<T, 2, 1> r1, n1, r2, n2; 
    r1 << 0, 0; 
    n1 << 1, 0;
    r2 << 2.4, 0;
    n2 << 1, 0;
    Array<T, 2, 1> d12;  
    d12 << 1.4, 0;
    T dist = d12.matrix().norm(); 
    Array<T, 2, 1> dnorm = d12 / dist; 
    Array<T, 2, 1> target_force_21 = -dnorm * (2.5 * E0 * sqrt(R) * pow(2 * R - dist, 1.5));
    testCellCellRepulsiveForces(
        r1, n1, 0.5, r2, n2, 0.5, R, Rcell, E0, Ecell, delta, target_force_21
    );

    // r1 = (0, 0), n1 = (1, 0), l1 = 1
    // r2 = (1 + 1.4 * cos(pi/6), 1.4 * sin(pi/6)), n2 = (1, 0), l2 = 1
    //
    // The shortest distance between the two cells is the vector
    // (1.4 * cos(pi/6), 1.4 * sin(pi/6)), which has norm 1.4
    r2(0) = 1.0 + 1.4 * cos(boost::math::constants::sixth_pi<T>());
    r2(1) = 1.4 * sin(boost::math::constants::sixth_pi<T>());  
    d12(0) = 1.4 * cos(boost::math::constants::sixth_pi<T>()); 
    d12(1) = 1.4 * sin(boost::math::constants::sixth_pi<T>()); 
    dist = d12.matrix().norm(); 
    dnorm = d12 / dist; 
    target_force_21 = -dnorm * (2.5 * E0 * sqrt(R) * pow(2 * R - dist, 1.5));
    testCellCellRepulsiveForces(
        r1, n1, 0.5, r2, n2, 0.5, R, Rcell, E0, Ecell, delta, target_force_21
    );

    // r1 = (0, 0), n1 = (1, 0), l1 = 1
    // r2 = (0, 1.5), n2 = (1, 0), l2 = 1
    //
    // The shortest distance between the two cells is the vector (0, 1.5)
    // 
    // Since the cells are aligned, there should be zero torque
    r2(0) = 0; 
    r2(1) = 1.5; 
    d12(0) = 0; 
    d12(1) = 1.5;
    dist = d12.matrix().norm(); 
    dnorm = d12 / dist; 
    target_force_21 = -dnorm * (2.5 * E0 * sqrt(R) * pow(2 * R - dist, 1.5));
    testCellCellRepulsiveForces(
        r1, n1, 0.5, r2, n2, 0.5, R, Rcell, E0, Ecell, delta, target_force_21
    );

    // r1 = (0, 0), n1 = (1, 0), l1 = 1
    // r2 = (0.2, 1.5), n2 = (1, 0), l2 = 1
    //
    // The shortest distance between the two cells is, again, the vector
    // (0, 1.5), but the cells are not aligned and there should be nonzero
    // torque
    r2(0) = 0.2; 
    testCellCellRepulsiveForces(
        r1, n1, 0.5, r2, n2, 0.5, R, Rcell, E0, Ecell, delta, target_force_21
    );

    // r1 = (0, 0), n1 = (1, 0), l1 = 2
    // r2 = (-0.3, 1.5), n2 = (1, 0), l2 = 1
    //
    // The shortest distance between the two cells is, again, the vector 
    // (0, 1.5), but cell 1 should experience a nonzero torque while cell 2
    // experiences zero torque
    r2(0) = -0.3; 
    testCellCellRepulsiveForces(
        r1, n1, 1, r2, n2, 0.5, R, Rcell, E0, Ecell, delta, target_force_21
    );
}

/**
 * A series of tests for cellCellRepulsiveForces() for perpendicular cell-cell
 * configurations. 
 */
TEST_CASE("Tests for cellCellRepulsiveForces(), perpendicular cells", "[cellCellRepulsiveForces()]") 
{
    const T R = 0.8;
    const T Rcell = 0.5; 
    const T E0 = 1.0;
    const T Ecell = 100.0;
    const T delta = 1e-8;

    // r1 = (0, 0), n1 = (1, 0), l1 = 1
    // r2 = (1.5, 0), n2 = (0, 1), l2 = 1
    //
    // The shortest distance between the two cells is the vector (1, 0)
    // 
    // Since the cells are aligned, there should be zero torque
    Array<T, 2, 1> r1, n1, r2, n2; 
    r1 << 0, 0; 
    n1 << 1, 0;
    r2 << 1.5, 0;
    n2 << 0, 1;
    Array<T, 2, 1> d12;  
    d12 << 1, 0;
    T dist = d12.matrix().norm(); 
    Array<T, 2, 1> dnorm = d12 / dist; 
    Array<T, 2, 1> target_force_21 = -dnorm * (
        2.5 * E0 * sqrt(R) * pow(2 * (R - Rcell), 1.5) +
        2.5 * Ecell * sqrt(Rcell) * pow(2 * Rcell - dist, 1.5)
    );
    testCellCellRepulsiveForces(
        r1, n1, 0.5, r2, n2, 0.5, R, Rcell, E0, Ecell, delta, target_force_21
    );

    // r1 = (0, 0), n1 = (1, 0), l1 = 1
    // r2 = (1.5, 0.2), n2 = (0, 1), l2 = 1
    //
    // The shortest distance between the two cells is, again, the vector
    // (1, 0), but cell 1 should experience zero torque while cell 2 experiences
    // a nonzero torque 
    r2(1) = 0.2;  
    testCellCellRepulsiveForces(
        r1, n1, 0.5, r2, n2, 0.5, R, Rcell, E0, Ecell, delta, target_force_21
    );
}

/**
 * A series of tests for cellCellRepulsiveForcesNewton() for skew cell-cell
 * configurations. 
 */
TEST_CASE("Tests for cellCellRepulsiveForcesNewton(), skew cells", "[cellCellRepulsiveForcesNewton()]")
{
    const T R = 0.8;
    const T Rcell = 0.5; 
    const T E0 = 1.0;
    const T Ecell = 100.0;

    // r1 = (2, 1), n1 = (0.6, 0.8), l1 = 2
    // r2 = (0, 3), n2 = (0, 1), l2 = 1
    //
    // The shortest distance between the two cells is the vector (-2, 1.5),
    // which has norm 2.5 > 2 * R = 1.6, which means that the force between
    // these cells should be zero 
    Array<T, 2, 1> r1, n1, r2, n2; 
    r1 << 2, 1; 
    n1 << 0.6, 0.8; 
    r2 << 0, 3; 
    n2 << 0, 1;
    testCellCellRepulsiveForcesNewton(r1, n1, 1, r2, n2, 0.5, R, Rcell, E0, Ecell);

    // r1 = (0, 0), n1 = (1, 0), l1 = 1
    // r2 = (0.5 + 0.5 * cos(pi/6), 1 + 0.5 * sin(pi/6)), n2 = (cos(pi/6), sin(pi/6)), l2 = 1
    //
    // The shortest distance between the two cells is the vector (0, 1)
    r1(0) = 0; 
    r1(1) = 0; 
    n1(0) = 1; 
    n1(1) = 0; 
    r2(0) = 0.5 + 0.5 * cos(boost::math::constants::sixth_pi<T>()); 
    r2(1) = 1 + 0.5 * sin(boost::math::constants::sixth_pi<T>()); 
    n2(0) = cos(boost::math::constants::sixth_pi<T>()); 
    n2(1) = sin(boost::math::constants::sixth_pi<T>());
    testCellCellRepulsiveForcesNewton(r1, n1, 0.5, r2, n2, 0.5, R, Rcell, E0, Ecell);

    // r1 = (0, 0), n1 = (1, 0), l1 = 1
    // r2 = (0.2 + 0.5 * cos(pi/6), 1 + 0.5 * sin(pi/6)), n2 = (cos(pi/6), sin(pi/6)), l2 = 1
    //
    // The shortest distance between the two cells is, again, (0, 1)
    r1(0) = 0; 
    r1(1) = 0; 
    n1(0) = 1; 
    n1(1) = 0; 
    r2(0) = 0.2 + 0.5 * cos(boost::math::constants::sixth_pi<T>()); 
    r2(1) = 1 + 0.5 * sin(boost::math::constants::sixth_pi<T>()); 
    n2(0) = cos(boost::math::constants::sixth_pi<T>()); 
    n2(1) = sin(boost::math::constants::sixth_pi<T>());
    testCellCellRepulsiveForcesNewton(r1, n1, 0.5, r2, n2, 0.5, R, Rcell, E0, Ecell);

    // r1 = (0, 0), n1 = (1, 0), l1 = 1
    // r2 = (0.5 + 1.4 / sqrt(2), 1.4 / sqrt(2)), n2 = (1 / sqrt(2), -1 / sqrt(2)), l2 = 1
    //
    // The shortest distance between the two cells is (1.4 / sqrt(2), 1.4 / sqrt(2))
    //
    // The torque on cell 2 should be zero, since the force acts on the cell 
    // center (r2)
    r2(0) = 0.5 + 1.4 / sqrt(2.0); 
    r2(1) = 1.4 / sqrt(2.0); 
    n2(0) = 1.0 / sqrt(2.0); 
    n2(1) = -1.0 / sqrt(2.0);
    testCellCellRepulsiveForcesNewton(r1, n1, 0.5, r2, n2, 0.5, R, Rcell, E0, Ecell);
}

/**
 * A series of tests for cellCellRepulsiveForcesNewton() for parallel cell-cell
 * configurations.
 */
TEST_CASE("Tests for cellCellRepulsiveForcesNewton(), parallel cells", "[cellCellRepulsiveForcesNewton()]")
{
    const T R = 0.8;
    const T Rcell = 0.5; 
    const T E0 = 1.0;
    const T Ecell = 100.0;

    // r1 = (0, 0), n1 = (1, 0), l1 = 1
    // r2 = (2.4, 0), n2 = (1, 0), l2 = 1
    //
    // The shortest distance between the two cells is the vector (1.4, 0)
    // 
    // Since the cells are aligned, there should be zero torque
    Array<T, 2, 1> r1, n1, r2, n2; 
    r1 << 0, 0; 
    n1 << 1, 0;
    r2 << 2.4, 0;
    n2 << 1, 0;
    testCellCellRepulsiveForcesNewton(r1, n1, 0.5, r2, n2, 0.5, R, Rcell, E0, Ecell);

    // r1 = (0, 0), n1 = (1, 0), l1 = 1
    // r2 = (1 + 1.4 * cos(pi/6), 1.4 * sin(pi/6)), n2 = (1, 0), l2 = 1
    //
    // The shortest distance between the two cells is the vector
    // (1.4 * cos(pi/6), 1.4 * sin(pi/6)), which has norm 1.4
    r2(0) = 1.0 + 1.4 * cos(boost::math::constants::sixth_pi<T>());
    r2(1) = 1.4 * sin(boost::math::constants::sixth_pi<T>());  
    testCellCellRepulsiveForcesNewton(r1, n1, 0.5, r2, n2, 0.5, R, Rcell, E0, Ecell);

    // r1 = (0, 0), n1 = (1, 0), l1 = 1
    // r2 = (0, 1.5), n2 = (1, 0), l2 = 1
    //
    // The shortest distance between the two cells is the vector (0, 1.5)
    // 
    // Since the cells are aligned, there should be zero torque
    r2(0) = 0; 
    r2(1) = 1.5; 
    testCellCellRepulsiveForcesNewton(r1, n1, 0.5, r2, n2, 0.5, R, Rcell, E0, Ecell);

    // r1 = (0, 0), n1 = (1, 0), l1 = 1
    // r2 = (0.2, 1.5), n2 = (1, 0), l2 = 1
    //
    // The shortest distance between the two cells is, again, the vector
    // (0, 1.5), but the cells are not aligned and there should be nonzero
    // torque
    r2(0) = 0.2; 
    testCellCellRepulsiveForcesNewton(r1, n1, 0.5, r2, n2, 0.5, R, Rcell, E0, Ecell);

    // r1 = (0, 0), n1 = (1, 0), l1 = 2
    // r2 = (-0.3, 1.5), n2 = (1, 0), l2 = 1
    //
    // The shortest distance between the two cells is, again, the vector 
    // (0, 1.5), but cell 1 should experience a nonzero torque while cell 2
    // experiences zero torque
    r2(0) = -0.3; 
    testCellCellRepulsiveForcesNewton(r1, n1, 1, r2, n2, 0.5, R, Rcell, E0, Ecell);
}

/**
 * A series of tests for cellCellRepulsiveForcesNewton() for perpendicular cell-cell
 * configurations. 
 */
TEST_CASE("Tests for cellCellRepulsiveForcesNewton(), perpendicular cells", "[cellCellRepulsiveForcesNewton()]") 
{
    const T R = 0.8;
    const T Rcell = 0.5; 
    const T E0 = 1.0;
    const T Ecell = 100.0;

    // r1 = (0, 0), n1 = (1, 0), l1 = 1
    // r2 = (1.5, 0), n2 = (0, 1), l2 = 1
    //
    // The shortest distance between the two cells is the vector (1, 0)
    // 
    // Since the cells are aligned, there should be zero torque
    Array<T, 2, 1> r1, n1, r2, n2; 
    r1 << 0, 0; 
    n1 << 1, 0;
    r2 << 1.5, 0;
    n2 << 0, 1;
    testCellCellRepulsiveForcesNewton(r1, n1, 0.5, r2, n2, 0.5, R, Rcell, E0, Ecell);

    // r1 = (0, 0), n1 = (1, 0), l1 = 1
    // r2 = (1.5, 0.2), n2 = (0, 1), l2 = 1
    //
    // The shortest distance between the two cells is, again, the vector
    // (1, 0), but cell 1 should experience zero torque while cell 2 experiences
    // a nonzero torque 
    r2(1) = 0.2;  
    testCellCellRepulsiveForcesNewton(r1, n1, 0.5, r2, n2, 0.5, R, Rcell, E0, Ecell);
}

