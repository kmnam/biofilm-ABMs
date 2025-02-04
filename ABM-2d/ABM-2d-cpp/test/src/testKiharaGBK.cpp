/**
 * Test module for the `forcesKiharaLagrange()`, `forcesKiharaNewton()`, 
 * `forcesGBKLagrange()`, and `forcesGBKNewton()` functions.  
 *
 * Authors:
 *     Kee-Myoung Nam
 *
 * Last updated:
 *     2/3/2025
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
#include "../../include/kiharaGBK.hpp"
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
 * Returns the generalized attractive forces and torques arising from the
 * Kihara potential on the given two cells, computed via finite difference
 * approximation from the potential.  
 *
 * @param r1 Cell 1 center. 
 * @param n1 Cell 1 orientation. 
 * @param half_l1 Cell 1 half-length. 
 * @param r2 Cell 2 center. 
 * @param n2 Cell 2 orientation. 
 * @param half_l2 Cell 2 half-length.
 * @param R Cell radius, including the EPS.
 * @param exp Exponent in Kihara potential. 
 * @param dmin Minimum distance at which the Kihara potential is nonzero. 
 * @param delta Increment for finite difference approximation. 
 * @returns Array of generalized forces and torques on the two cells. 
 */
Array<T, 2, 4> forcesKiharaFiniteDiff(const Ref<const Array<T, 2, 1> >& r1,
                                      const Ref<const Array<T, 2, 1> >& n1,
                                      const T half_l1,
                                      const Ref<const Array<T, 2, 1> >& r2, 
                                      const Ref<const Array<T, 2, 1> >& n2,
                                      const T half_l2, const T R, const T exp, 
                                      const T dmin, const T delta)
{
    K kernel;
    Segment_3 seg1 = generateSegment<T>(r1, n1, half_l1); 
    Segment_3 seg2 = generateSegment<T>(r2, n2, half_l2);  
    Array<T, 2, 4> dEdq = Array<T, 2, 4>::Zero();
    Array<T, 2, 1> dx, dy; 
    dx << delta, 0; 
    dy << 0, delta;

    // Compute all eight finite differences ... 
    //
    // 1) Partial derivatives w.r.t r1
    Segment_3 seg1_px = generateSegment<T>(r1 + dx, n1, half_l1);
    Segment_3 seg1_mx = generateSegment<T>(r1 - dx, n1, half_l1); 
    Segment_3 seg1_py = generateSegment<T>(r1 + dy, n1, half_l1); 
    Segment_3 seg1_my = generateSegment<T>(r1 - dy, n1, half_l1); 
    auto result = distBetweenCells<T>(
        seg1_px, seg2, 0, r1 + dx, n1, half_l1, 1, r2, n2, half_l2, kernel
    );
    T dist1 = std::get<0>(result).norm();
    result = distBetweenCells<T>(
        seg1_mx, seg2, 0, r1 - dx, n1, half_l1, 1, r2, n2, half_l2, kernel
    );
    T dist2 = std::get<0>(result).norm();
    result = distBetweenCells<T>(
        seg1_py, seg2, 0, r1 + dy, n1, half_l1, 1, r2, n2, half_l2, kernel
    );
    T dist3 = std::get<0>(result).norm();  
    result = distBetweenCells<T>(
        seg1_my, seg2, 0, r1 - dy, n1, half_l1, 1, r2, n2, half_l2, kernel
    );
    T dist4 = std::get<0>(result).norm();  
    dEdq(0, 0) = potentialKihara<T>(dist1, R, exp, dmin) - potentialKihara<T>(dist2, R, exp, dmin); 
    dEdq(0, 1) = potentialKihara<T>(dist3, R, exp, dmin) - potentialKihara<T>(dist4, R, exp, dmin); 

    // 2) Partial derivatives w.r.t n1
    seg1_px = generateSegment<T>(r1, n1 + dx, half_l1);
    seg1_mx = generateSegment<T>(r1, n1 - dx, half_l1); 
    seg1_py = generateSegment<T>(r1, n1 + dy, half_l1); 
    seg1_my = generateSegment<T>(r1, n1 - dy, half_l1);
    result = distBetweenCells<T>(
        seg1_px, seg2, 0, r1, n1 + dx, half_l1, 1, r2, n2, half_l2, kernel
    );
    dist1 = std::get<0>(result).norm();
    result = distBetweenCells<T>(
        seg1_mx, seg2, 0, r1, n1 - dx, half_l1, 1, r2, n2, half_l2, kernel
    );
    dist2 = std::get<0>(result).norm();
    result = distBetweenCells<T>(
        seg1_py, seg2, 0, r1, n1 + dy, half_l1, 1, r2, n2, half_l2, kernel
    );
    dist3 = std::get<0>(result).norm();  
    result = distBetweenCells<T>(
        seg1_my, seg2, 0, r1, n1 - dy, half_l1, 1, r2, n2, half_l2, kernel
    );
    dist4 = std::get<0>(result).norm(); 
    dEdq(0, 2) = potentialKihara<T>(dist1, R, exp, dmin) - potentialKihara<T>(dist2, R, exp, dmin); 
    dEdq(0, 3) = potentialKihara<T>(dist3, R, exp, dmin) - potentialKihara<T>(dist4, R, exp, dmin); 
    
    // 3) Partial derivatives w.r.t r2
    Segment_3 seg2_px = generateSegment<T>(r2 + dx, n2, half_l2);
    Segment_3 seg2_mx = generateSegment<T>(r2 - dx, n2, half_l2); 
    Segment_3 seg2_py = generateSegment<T>(r2 + dy, n2, half_l2); 
    Segment_3 seg2_my = generateSegment<T>(r2 - dy, n2, half_l2); 
    result = distBetweenCells<T>(
        seg1, seg2_px, 0, r1, n1, half_l1, 1, r2 + dx, n2, half_l2, kernel
    );
    dist1 = std::get<0>(result).norm();
    result = distBetweenCells<T>(
        seg1, seg2_mx, 0, r1, n1, half_l1, 1, r2 - dx, n2, half_l2, kernel
    );
    dist2 = std::get<0>(result).norm();
    result = distBetweenCells<T>(
        seg1, seg2_py, 0, r1, n1, half_l1, 1, r2 + dy, n2, half_l2, kernel
    );
    dist3 = std::get<0>(result).norm();  
    result = distBetweenCells<T>(
        seg1, seg2_my, 0, r1, n1, half_l1, 1, r2 - dy, n2, half_l2, kernel
    );
    dist4 = std::get<0>(result).norm(); 
    dEdq(1, 0) = potentialKihara<T>(dist1, R, exp, dmin) - potentialKihara<T>(dist2, R, exp, dmin); 
    dEdq(1, 1) = potentialKihara<T>(dist3, R, exp, dmin) - potentialKihara<T>(dist4, R, exp, dmin);

    // 4) Partial derivatives w.r.t n2
    seg2_px = generateSegment<T>(r2, n2 + dx, half_l2);
    seg2_mx = generateSegment<T>(r2, n2 - dx, half_l2); 
    seg2_py = generateSegment<T>(r2, n2 + dy, half_l2); 
    seg2_my = generateSegment<T>(r2, n2 - dy, half_l2); 
    result = distBetweenCells<T>(
        seg1, seg2_px, 0, r1, n1, half_l1, 1, r2, n2 + dx, half_l2, kernel
    );
    dist1 = std::get<0>(result).norm();
    result = distBetweenCells<T>(
        seg1, seg2_mx, 0, r1, n1, half_l1, 1, r2, n2 - dx, half_l2, kernel
    );
    dist2 = std::get<0>(result).norm();
    result = distBetweenCells<T>(
        seg1, seg2_py, 0, r1, n1, half_l1, 1, r2, n2 + dy, half_l2, kernel
    );
    dist3 = std::get<0>(result).norm();  
    result = distBetweenCells<T>(
        seg1, seg2_my, 0, r1, n1, half_l1, 1, r2, n2 - dy, half_l2, kernel
    );
    dist4 = std::get<0>(result).norm(); 
    dEdq(1, 2) = potentialKihara<T>(dist1, R, exp, dmin) - potentialKihara<T>(dist2, R, exp, dmin); 
    dEdq(1, 3) = potentialKihara<T>(dist3, R, exp, dmin) - potentialKihara<T>(dist4, R, exp, dmin);

    // Normalize by double the increment and return 
    return dEdq / (2 * delta); 
}

/* ------------------------------------------------------------------ //
 *                       GENERIC TEST FUNCTIONS                       //
 * ------------------------------------------------------------------ */
/**
 * A generic test function for forcesKiharaLagrange(). 
 *
 * @param r1 Cell 1 center. 
 * @param n1 Cell 1 orientation. 
 * @param half_l1 Cell 1 half-length. 
 * @param r2 Cell 2 center. 
 * @param n2 Cell 2 orientation. 
 * @param half_l2 Cell 2 half-length. 
 * @param R Cell radius, including the EPS. 
 * @param exp Exponent in Kihara potential. 
 * @param dmin Minimum distance at which the Kihara potential is nonzero. 
 * @param delta Increment for finite difference approximation. 
 * @param target_force_21 Pre-computed force vector on cell 1 due to cell 2. 
 * @param check_zero_torque If true, check that the torques on both cells are
 *                          zero.   
 */
void testForcesKiharaLagrange(const Ref<const Array<T, 2, 1> >& r1,
                              const Ref<const Array<T, 2, 1> >& n1, const T half_l1,
                              const Ref<const Array<T, 2, 1> >& r2, 
                              const Ref<const Array<T, 2, 1> >& n2, const T half_l2,
                              const T R, const T exp, const T dmin, const T delta, 
                              const Ref<const Array<T, 2, 1> >& target_force_21,
                              const bool check_zero_torque)
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

    // Compute the forces via forcesKiharaLagrange() *without* enforcing 
    // the orientation vector norm constraint 
    Array<T, 2, 4> forces1_unconstrained = -forcesKiharaLagrange<T, 2>(
        n1.matrix(), n2.matrix(), d12, R, s, t, exp, dmin, false
    );

    // Do the same *while* enforcing the orientation vector norm constraint 
    Array<T, 2, 4> forces1_constrained = -forcesKiharaLagrange<T, 2>(
        n1.matrix(), n2.matrix(), d12, R, s, t, exp, dmin, true
    );

    // Check that the force on cell 1 due to cell 2 is correct 
    REQUIRE_THAT(
        (forces1_unconstrained(0, Eigen::seq(0, 1)) - target_force_21.transpose()).matrix().norm(),
        Catch::Matchers::WithinAbs(0.0, delta)
    );
    REQUIRE_THAT(
        (forces1_constrained(0, Eigen::seq(0, 1)) - target_force_21.transpose()).matrix().norm(), 
        Catch::Matchers::WithinAbs(0.0, delta)
    );

    // Check that the torques with and without the constraint differ by the 
    // Lagrange multiplier 
    T lambda1 = 0.5 * n1.matrix().dot(-forces1_unconstrained(0, Eigen::seq(2, 3)).matrix()); 
    T lambda2 = 0.5 * n2.matrix().dot(-forces1_unconstrained(1, Eigen::seq(2, 3)).matrix()); 
    REQUIRE_THAT(
        forces1_constrained(0, 2) - forces1_unconstrained(0, 2),
        Catch::Matchers::WithinAbs(2 * lambda1 * n1(0), delta)
    );
    REQUIRE_THAT(
        forces1_constrained(0, 3) - forces1_unconstrained(0, 3), 
        Catch::Matchers::WithinAbs(2 * lambda1 * n1(1), delta)
    );
    REQUIRE_THAT(
        forces1_constrained(1, 2) - forces1_unconstrained(1, 2), 
        Catch::Matchers::WithinAbs(2 * lambda2 * n2(0), delta) 
    ); 
    REQUIRE_THAT(
        forces1_constrained(1, 3) - forces1_unconstrained(1, 3), 
        Catch::Matchers::WithinAbs(2 * lambda2 * n2(1), delta)
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
        (forces1_constrained(0, Eigen::seq(2, 3)) - target_torque_21.transpose()).matrix().norm(),
        Catch::Matchers::WithinAbs(0.0, delta)
    ); 

    // Check that the two force vectors are equal and opposite
    REQUIRE_THAT(
        (forces1_unconstrained(0, Eigen::seq(0, 1)) + forces1_unconstrained(1, Eigen::seq(0, 1))).matrix().norm(),
        Catch::Matchers::WithinAbs(0.0, delta)
    );
    REQUIRE_THAT(
        (forces1_constrained(0, Eigen::seq(0, 1)) + forces1_constrained(1, Eigen::seq(0, 1))).matrix().norm(),
        Catch::Matchers::WithinAbs(0.0, delta)
    );

    // Check that the two torque vectors are zero, if that is expected 
    if (check_zero_torque)
    {
        REQUIRE_THAT(
            forces1_constrained(0, Eigen::seq(2, 3)).matrix().norm(),
            Catch::Matchers::WithinAbs(0.0, delta)
        );
        REQUIRE_THAT(
            forces1_constrained(1, Eigen::seq(2, 3)).matrix().norm(),
            Catch::Matchers::WithinAbs(0.0, delta)
        );         
    }

    // Compute the forces via finite differences and check against the 
    // unconstrained forces  
    Array<T, 2, 4> dEdq2 = forcesKiharaFiniteDiff(
        r1, n1, half_l1, r2, n2, half_l2, R, exp, dmin, delta
    );
    Array<T, 2, 4> forces2(-dEdq2);
    REQUIRE_THAT(
        (forces1_unconstrained - forces2).abs().maxCoeff(),
        Catch::Matchers::WithinAbs(0.0, delta)
    );         

    // Apply the unit vector constraint to each torque computed via finite 
    // differences 
    lambda1 = 0.5 * n1.matrix().dot(dEdq2(0, Eigen::seq(2, 3)).matrix()); 
    lambda2 = 0.5 * n2.matrix().dot(dEdq2(1, Eigen::seq(2, 3)).matrix());
    forces2(0, Eigen::seq(2, 3)) += lambda1 * 2 * n1; 
    forces2(1, Eigen::seq(2, 3)) += lambda2 * 2 * n2;

    // Check that the constrained forces agree with each other
    REQUIRE_THAT(
        (forces1_constrained - forces2).abs().maxCoeff(),
        Catch::Matchers::WithinAbs(0.0, delta)
    ); 
}

/**
 * A series of tests for forcesKiharaLagrange() for skew cell-cell configurations. 
 */
TEST_CASE("Tests for forcesKiharaLagrange(), skew cells", "[forcesKiharaLagrange()]") 
{
    const T R = 0.8;
    const T exp = 4; 
    const T dmin = 1.05;
    const T delta = 1e-7;  

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
    testForcesKiharaLagrange(
        r1, n1, 1, r2, n2, 0.5, R, exp, dmin, delta, target_force_21, true
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
    target_force_21 = d12 * exp * (pow(dmin, -exp - 1) - pow(2 * R, -exp - 1));
    testForcesKiharaLagrange(
        r1, n1, 0.5, r2, n2, 0.5, R, exp, dmin, delta, target_force_21, false
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
    testForcesKiharaLagrange(
        r1, n1, 0.5, r2, n2, 0.5, R, exp, dmin, delta, target_force_21, false
    );

    // TODO Cells that are further apart, in the repulsive-attractive regime 
}

/**
 * A series of tests for forcesKiharaLagrange() for parallel cell-cell configurations.
 */
TEST_CASE("Tests for forcesKiharaLagrange(), parallel cells", "[forcesKiharaLagrange()]")
{
    const T R = 0.8;
    const T exp = 4; 
    const T dmin = 1.05;
    const T delta = 1e-7;  

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
    Array<T, 2, 1> target_force_21 = dnorm * exp * (pow(dist, -exp - 1) - pow(2 * R, -exp - 1)); 
    testForcesKiharaLagrange(
        r1, n1, 0.5, r2, n2, 0.5, R, exp, dmin, delta, target_force_21, true
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
    target_force_21 = dnorm * exp * (pow(dist, -exp - 1) - pow(2 * R, -exp - 1)); 
    testForcesKiharaLagrange(
        r1, n1, 0.5, r2, n2, 0.5, R, exp, dmin, delta, target_force_21, false
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
    target_force_21 = dnorm * exp * (pow(dist, -exp - 1) - pow(2 * R, -exp - 1)); 
    testForcesKiharaLagrange(
        r1, n1, 0.5, r2, n2, 0.5, R, exp, dmin, delta, target_force_21, true
    );

    // r1 = (0, 0), n1 = (1, 0), l1 = 1
    // r2 = (0.2, 1.5), n2 = (1, 0), l2 = 1
    //
    // The shortest distance between the two cells is, again, the vector
    // (0, 1.5), but the cells are not aligned and there should be nonzero
    // torque
    r2(0) = 0.2; 
    testForcesKiharaLagrange(
        r1, n1, 0.5, r2, n2, 0.5, R, exp, dmin, delta, target_force_21, false
    );
}

/**
 * A series of tests for forcesKiharaLagrange() for perpendicular cell-cell
 * configurations. 
 */
TEST_CASE("Tests for forcesKiharaLagrange(), perpendicular cells", "[forcesKiharaLagrange()]") 
{
    const T R = 0.8;
    const T exp = 4; 
    const T dmin = 1.05;
    const T delta = 1e-7;  

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
    Array<T, 2, 1> target_force_21 = dnorm * exp * (pow(dmin, -exp - 1) - pow(2 * R, -exp - 1));
    testForcesKiharaLagrange(
        r1, n1, 0.5, r2, n2, 0.5, R, exp, dmin, delta, target_force_21, true
    );

    // r1 = (0, 0), n1 = (1, 0), l1 = 1
    // r2 = (1.5, 0.2), n2 = (0, 1), l2 = 1
    //
    // The shortest distance between the two cells is, again, the vector
    // (1, 0), but the cells are not aligned and there should be nonzero 
    // torque
    r2(1) = 0.2;  
    testForcesKiharaLagrange(
        r1, n1, 0.5, r2, n2, 0.5, R, exp, dmin, delta, target_force_21, false
    );
}

