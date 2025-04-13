/**
 * Test module for the `nearestCellBodyCoordToPoint()` and `distBetweenCells()`
 * functions. 
 *
 * Authors:
 *     Kee-Myoung Nam
 *
 * Last updated:
 *     4/13/2025
 */
#include <cmath>
#include <Eigen/Dense>
#include <boost/math/constants/constants.hpp>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Segment_3.h>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "../../include/distances.hpp"

using namespace Eigen;

typedef double T;
typedef CGAL::Exact_predicates_inexact_constructions_kernel K; 
typedef K::Segment_3 Segment_3;

using std::sin; 
using boost::multiprecision::sin; 
using std::cos; 
using boost::multiprecision::cos; 
using std::tan; 
using boost::multiprecision::tan;

/* ---------------------------------------------------------------- //
 *                         UTILITY FUNCTIONS                        //
 * ---------------------------------------------------------------- */
/**
 * A function that calculates the shortest distance vector between two 
 * infinite lines. 
 *
 * This was taken from Vega and Lago, Comput. Chem. (1994). 
 */
std::tuple<Matrix<T, 3, 1>, T, T> distBetweenLines(const Ref<const Matrix<T, 3, 1> >& r1,
                                                   const Ref<const Matrix<T, 3, 1> >& n1,
                                                   const Ref<const Matrix<T, 3, 1> >& r2,
                                                   const Ref<const Matrix<T, 3, 1> >& n2)
{
    Matrix<T, 3, 1> r12 = r2 - r1; 
    T s = (r12.dot(n1) - n1.dot(n2) * r12.dot(n2)) / (1.0 - n1.dot(n2) * n1.dot(n2));
    T t = (-r12.dot(n2) + n1.dot(n2) * r12.dot(n1)) / (1.0 - n1.dot(n2) * n1.dot(n2));
    Matrix<T, 3, 1> d = r12 + t * n2 - s * n1;

    return std::make_tuple(d, s, t);  
}

/* ---------------------------------------------------------------- //
 *                          TEST FUNCTIONS                          //
 * ---------------------------------------------------------------- */
/**
 * A generic test function for nearestCellBodyCoordToPoint().
 *
 * @param rx x-coordinate of cell center. 
 * @param ry y-coordinate of cell center.
 * @param rz z-coordinate of cell center. 
 * @param nx x-coordinate of cell orientation. 
 * @param ny y-coordinate of cell orientation.
 * @param nz z-coordinate of cell orientation.
 * @param half_l Cell half-length.
 * @param qx x-coordinate of query point. 
 * @param qy y-coordinate of query point. 
 * @param qz z-coordinate of query point.
 * @param target Pre-computed nearest centerline coordinate. 
 */
void testNearestCellBodyCoordToPoint(const T rx, const T ry, const T rz,
                                     const T nx, const T ny, const T nz,
                                     const T half_l, const T qx, const T qy,
                                     const T qz, const T target)
{
    Matrix<T, 3, 1> r, n, q; 
    r << rx, ry, rz; 
    n << nx, ny, nz; 
    q << qx, qy, qz; 
    T s = nearestCellBodyCoordToPoint<T>(r, n, half_l, q); 
    REQUIRE_THAT(s, Catch::Matchers::WithinAbs(target, 1e-8)); 
}

/**
 * A generic test function for distBetweenCells().
 *
 * @param r1x x-coordinate of cell 1 center. 
 * @param r1y y-coordinate of cell 1 center.
 * @param r1z z-coordinate of cell 1 center. 
 * @param n1x x-coordinate of cell 1 orientation. 
 * @param n1y y-coordinate of cell 1 orientation.
 * @param n1z z-coordinate of cell 1 orientation.
 * @param half_l1 Half-length of cell 1.
 * @param r2x x-coordinate of cell 2 center. 
 * @param r2y y-coordinate of cell 2 center.
 * @param r2z z-coordinate of cell 2 center. 
 * @param n2x x-coordinate of cell 2 orientation. 
 * @param n2y y-coordinate of cell 2 orientation.
 * @param n2z z-coordinate of cell 2 orientation. 
 * @param half_l2 Half-length of cell 2.
 * @param target_dx x-coordinate of pre-computed distance vector from cell 1 to
 *                  cell 2.
 * @param target_dy y-coordinate of pre-computed distance vector from cell 1 to
 *                  cell 2.
 * @param target_dz z-coordinate of pre-computed distance vector from cell 1 to
 *                  cell 2.
 * @param target_s Pre-computed centerline coordinate along cell 1.
 * @param target_t Pre-computed centerline coordinate along cell 2.  
 */
void testDistBetweenCells(const T r1x, const T r1y, const T r1z, const T n1x,
                          const T n1y, const T n1z, const T half_l1,
                          const T r2x, const T r2y, const T r2z, const T n2x,
                          const T n2y, const T n2z, const T half_l2,
                          const T target_dx, const T target_dy, const T target_dz,
                          const T target_s, const T target_t)
{
    K kernel; 
    Matrix<T, 3, 1> r1, n1, r2, n2; 
    r1 << r1x, r1y, r1z; 
    n1 << n1x, n1y, n1z; 
    r2 << r2x, r2y, r2z; 
    n2 << n2x, n2y, n2z;
    Segment_3 seg1 = generateSegment<T>(r1, n1, half_l1); 
    Segment_3 seg2 = generateSegment<T>(r2, n2, half_l2); 
    auto result = distBetweenCells<T>(
        seg1, seg2, 0, r1, n1, half_l1, 1, r2, n2, half_l2, kernel    // Use IDs 0 and 1
    );
    REQUIRE_THAT(std::get<0>(result)(0), Catch::Matchers::WithinAbs(target_dx, 1e-8)); 
    REQUIRE_THAT(std::get<0>(result)(1), Catch::Matchers::WithinAbs(target_dy, 1e-8));
    REQUIRE_THAT(std::get<0>(result)(2), Catch::Matchers::WithinAbs(target_dz, 1e-8)); 
    REQUIRE_THAT(std::get<1>(result), Catch::Matchers::WithinAbs(target_s, 1e-8)); 
    REQUIRE_THAT(std::get<2>(result), Catch::Matchers::WithinAbs(target_t, 1e-8)); 
}

/* ---------------------------------------------------------------- //
 *                           TEST MODULES                           //
 * ---------------------------------------------------------------- */
/**
 * A series of tests for nearestCellBodyCoordToPoint(). 
 */
TEST_CASE("Tests for nearestCellBodyCoordToPoint()", "[nearestCellBodyCoordToPoint()]")
{
    /* -------------------------------------------------------------- //
     *                     HORIZONTAL TEST CASES                      //
     * -------------------------------------------------------------- */
    // r = (2, 1, 0), n = (0.6, 0.8, 0), l = 2, q = (0, 3, 0)
    //
    // The nearest cell-body coordinate should be 0.4
    testNearestCellBodyCoordToPoint(2, 1, 0, 0.6, 0.8, 0, 1, 0, 3, 0, 0.4); 

    // r = (6, 2, 2), n = (12 / 13, 5 / 13, 0), l = 1.5, q = (7, 1, 2)
    //
    // The nearest cell-body coordinate should be ~0.5384615385
    testNearestCellBodyCoordToPoint(
        6, 2, 2, 12.0 / 13.0, 5.0 / 13.0, 0, 0.75, 7, 1, 2, 0.5384615385
    ); 

    // The same configuration as the previous, except with l = 1.0
    //
    // The nearest cell-body coordinate should be 0.5
    testNearestCellBodyCoordToPoint(
        6, 2, 2, 12.0 / 13.0, 5.0 / 13.0, 0, 0.5, 7, 1, 2, 0.5
    );
}

/**
 * A series of tests for distBetweenCells() for skew cell-cell configurations. 
 */
TEST_CASE("Tests for distBetweenCells(), skew cells", "[distBetweenCells()]")
{
    T target_dx, target_dy, target_dz, target_s, target_t; 

    // -------------------------------------------------------------- //
    //                     HORIZONTAL TEST CASES                      //
    // -------------------------------------------------------------- //
    // r1 = (2, 1, 0), n1 = (0.6, 0.8, 0), l1 = 2
    // r2 = (0, 3, 0), n2 = (0, 1, 0), l2 = 1
    //
    // The shortest distance between the two cells should be the vector
    // from r1 = (2, 1, 0) to r2 - 0.5 * n2 = (0, 2.5, 0), which is (-2, 1.5, 0)
    testDistBetweenCells(
        2, 1, 0, 0.6, 0.8, 0, 1,    // Cell 1
        0, 3, 0, 0, 1, 0, 0.5,      // Cell 2
        -2, 1.5, 0, 0, -0.5
    );

    // r1 = (4, 4, 2), n1 = (5 / 13, 12 / 13, 0), l1 = 5.4
    // r2 = (5, 3, 2), n1 = (0, 1, 0), l2 = 1
    //
    // The shortest distance between the two cells is achieved at t = 0.5
    // and s = (13 / 5) - (12 / 13) * (6.4 - 3 - 0.5) ~= -0.07692307692
    target_s = -0.07692307692; 
    target_t = 0.5; 
    target_dx = (5 + target_t * 0) - (4 + target_s * 5.0 / 13.0);
    target_dy = (3 + target_t * 1) - (4 + target_s * 12.0 / 13.0);
    target_dz = 0; 
    testDistBetweenCells(
        4, 4, 2, 5.0 / 13.0, 12.0 / 13.0, 0, 2.7,    // Cell 1
        5, 3, 2, 0, 1, 0, 0.5,                       // Cell 2
        target_dx, target_dy, target_dz, target_s, target_t
    );

    // The same configuration as the previous, except with l2 = 10.0
    //
    // The two cells should now intersect at (5, 6.4), which correspond
    // to cell-body coordinates of s = 13 / 5 = 2.6 and t = 3.4
    testDistBetweenCells(
        4, 4, 2, 5.0 / 13.0, 12.0 / 13.0, 0, 2.7,    // Cell 1
        5, 3, 2, 0, 1, 0, 5.0,                       // Cell 2
        0.0, 0.0, 0.0, 2.6, 3.4
    );

    // The same configuration as the previous, except with l1 = 2.0
    //
    // The shortest distance between the two cells is achieved at s = 1.0
    // and t = 3.4 - (6.4 - 4 - 12/13) ~= 1.9230769231
    target_s = 1.0; 
    target_t = 1.9230769231;
    target_dx = (5 + target_t * 0) - (4 + target_s * 5.0 / 13.0); 
    target_dy = (3 + target_t * 1) - (4 + target_s * 12.0 / 13.0);
    target_dz = 0;
    testDistBetweenCells(
        4, 4, 2, 5.0 / 13.0, 12.0 / 13.0, 0, 1.0,    // Cell 1
        5, 3, 2, 0, 1, 0, 5.0,                       // Cell 2 
        target_dx, target_dy, target_dz, target_s, target_t
    );

    // r1 = (4, 4, 5), n1 = (5 / 13, 12 / 13, 0), l1 = 1.2
    // r2 = (6, 2, 5), n1 = (0, 1, 0), l2 = 1
    //
    // The shortest distance between the two cells is achieved at the two
    // endpoints, s = -0.6 and t = 0.5
    target_s = -0.6; 
    target_t = 0.5;
    target_dx = (6 + target_t * 0) - (4 + target_s * 5.0 / 13.0); 
    target_dy = (2 + target_t * 1) - (4 + target_s * 12.0 / 13.0);
    target_dz = 0; 
    testDistBetweenCells(
        4, 4, 5, 5.0 / 13.0, 12.0 / 13.0, 0, 0.6,    // Cell 1
        6, 2, 5, 0, 1, 0, 0.5,                       // Cell 2
        target_dx, target_dy, target_dz, target_s, target_t
    );

    // -------------------------------------------------------------- //
    //                         3-D TEST CASES                         //
    // -------------------------------------------------------------- //
    // r1 = (0, 0, 0), n1 = (1, 0, 0), l1 = 1
    // r2 = (0, 1.4, 0), n2 = (0, 0, 1), l2 = 1
    //
    // The shortest distance between the two cells is achieved at their
    // centers, s = t = 0, along the vector (0, 1.4, 0)
    target_s = 0; 
    target_t = 0; 
    target_dx = 1.4; 
    target_dy = 0; 
    target_dz = 0; 
    testDistBetweenCells(
        0, 0, 0, 1, 0, 0, 0.5,      // Cell 1
        0, 1.4, 0, 0, 0, 1, 0.5,    // Cell 2
        0, 1.4, 0, 0, 0
    );

    // A skew configuration in which cell 2 runs along the y-axis and cell 1
    // runs diagonally "on top" of cell 2, within the xz-plane
    //
    // r1 = (0, 0, l1 * sin(30) / 2), n1 = (cos(30), 0, sin(30)), l1 = 1
    // r2 = (0.75 * l1 / cos(30) - l1 * cos(30) / 2, 0, 0), n2 = (0, 1, 0), l2 = 1
    //
    // The shortest distance between the two cells is achieved at:
    //
    // s = 0.25 * l1 = 0.25 and t = 0,
    //
    // along the vector (0.75 / cos(30) - 0.75 * cos(30), 0, -0.75 * sin(30)),
    // which is equal to (0.75 * tan(30) * sin(30), 0, -0.75 * sin(30))
    T half_l1 = 0.5; 
    T half_l2 = 0.5; 
    T l1 = 2 * half_l1; 
    T l2 = 2 * half_l2; 
    T theta = boost::math::constants::sixth_pi<T>(); 
    target_s = 0.25 * l1;
    target_t = 0; 
    target_dx = 0.75 * tan(theta) * sin(theta); 
    target_dy = 0; 
    target_dz = -0.75 * sin(theta); 
    testDistBetweenCells(
        0, 0, half_l1 * sin(theta), cos(theta), 0, sin(theta), 0.5,           // Cell 1
        0.75 * l1 / cos(theta) - half_l1 * cos(theta), 0, 0, 0, 1, 0, 0.5,    // Cell 2
        target_dx, target_dy, target_dz, target_s, target_t
    );

    // The same configuration but with cell 2 translated along the y-direction
    //
    // r1 = (0, 0, l1 * sin(30) / 2), n1 = (cos(30), 0, sin(30)), l1 = 1
    // r2 = (0.75 * l1 / cos(30) - l1 * cos(30) / 2, l2 / 4, 0), n2 = (0, 1, 0), l2 = 1
    //
    // The shortest distance between the two cells is achieved at:
    //
    // s = 0.25 * l1 = 0.25 and t = -0.25 * l2 = -0.25,
    //
    // along the vector (0.75 / cos(30) - 0.75 * cos(30), 0, -0.75 * sin(30)),
    // which is equal to (0.75 * tan(30) * sin(30), 0, -0.75 * sin(30))
    target_s = 0.25 * l1;
    target_t = -0.25 * l2; 
    target_dx = 0.75 * tan(theta) * sin(theta); 
    target_dy = 0; 
    target_dz = -0.75 * sin(theta); 
    testDistBetweenCells(
        0, 0, half_l1 * sin(theta), cos(theta), 0, sin(theta), 0.5,                // Cell 1
        0.75 * l1 / cos(theta) - half_l1 * cos(theta), l2 / 4, 0, 0, 1, 0, 0.5,    // Cell 2
        target_dx, target_dy, target_dz, target_s, target_t
    ); 

    // The same configuration but with cell 2 recentered to r(1) = 0, but 
    // with the orientation perturbed slightly within the xy-plane 
    //
    // r1 = (0, 0, l1 * sin(30) / 2), n1 = (cos(30), 0, sin(30)), l1 = 1
    // r2 = (0.75 * l1 / cos(30) - l1 * cos(30) / 2, 0, 0), n2 = (-sin(5), cos(5), 0), l2 = 1
    //
    // We can compute the shortest distance between the two cells as the 
    // shortest distance between the two extended lines 
    T eps = 5 * boost::math::constants::pi<T>() / 180;
    Matrix<T, 3, 1> r1, n1, r2, n2; 
    r1 << 0, 0, half_l1 * sin(theta);
    n1 << cos(theta), 0, sin(theta);
    r2 << 0.75 * l1 / cos(theta) - half_l1 * cos(theta), 0, 0;
    n2 << -sin(eps), cos(eps), 0; 
    auto target = distBetweenLines(r1, n1, r2, n2); 
    target_s = std::get<1>(target); 
    target_t = std::get<2>(target); 
    target_dx = std::get<0>(target)(0);
    target_dy = std::get<0>(target)(1); 
    target_dz = std::get<0>(target)(2); 
    testDistBetweenCells(
        0, 0, half_l1 * sin(theta), cos(theta), 0, sin(theta), 0.5,                          // Cell 1
        0.75 * l1 / cos(theta) - half_l1 * cos(theta), 0, 0, -sin(eps), cos(eps), 0, 0.5,    // Cell 2
        target_dx, target_dy, target_dz, target_s, target_t
    ); 
}

/**
 * A series of tests for distBetweenCells() for parallel and nearly parallel
 * cell-cell configurations. 
 */
TEST_CASE("Tests for distBetweenCells(), parallel cells", "[distBetweenCells()]")
{
    /* -------------------------------------------------------------- //
     *                     HORIZONTAL TEST CASES                      //
     * -------------------------------------------------------------- */
    // r1 = (0, 0, 0), n1 = (1, 0, 0), l1 = 2
    // r2 = (-2, 2, 0), n2 = (1, 0, 0), l2 = 1.5
    //
    // The shortest distance between the two cells is achieved at the two 
    // endpoints, s = -1 and t = 0.75
    T target_s = -1; 
    T target_t = 0.75;
    T target_dx = (-2 + target_t * 1) - (0 + target_s * 1);
    T target_dy = 2;
    T target_dz = 0;
    testDistBetweenCells(
        0, 0, 0, 1, 0, 0, 1.0,      // Cell 1
        -2, 2, 0, 1, 0, 0, 0.75,    // Cell 2
        target_dx, target_dy, target_dz, target_s, target_t
    );

    // r1 = (0, 0, 2), n1 = (1, 0, 0), l1 = 2
    // r2 = (-0.8, 2, 2), n2 = (1, 0, 0), l2 = 3
    //
    // The shortest distance between the two cells is achieved halfway between
    // the two vectors from (-1, 0, 2) to (-1, 2, 2) and from (0.7, 0, 2) to
    // (0.7, 2, 2), which is the vector from (-0.15, 0, 2) to (-0.15, 2, 2)
    //
    // This is achieved at the cell-body coordinates s = -0.15 and t = 0.65
    target_s = -0.15; 
    target_t = 0.65;
    target_dx = 0;
    target_dy = 2;
    target_dz = 0;
    testDistBetweenCells(
        0, 0, 2, 1, 0, 0, 1.0,       // Cell 1
        -0.8, 2, 2, 1, 0, 0, 1.5,    // Cell 2
        target_dx, target_dy, target_dz, target_s, target_t
    );

    // The same configuration as the previous, but with the orientation of 
    // cell 2 reversed
    //
    // The shortest distance between the two cells again runs from (-0.15, 0, 2)
    // to (-0.15, 2, 2), which is achieved at the cell-body coordinates s = -0.15
    // and t = -0.65
    testDistBetweenCells(
        0, 0, 2, 1, 0, 0, 1.0,       // Cell 1
        -0.8, 2, 2, -1, 0, 0, 1.5,   // Cell 2
        target_dx, target_dy, target_dz, target_s, -target_t
    );

    // r1 = (0, 0, 5), n1 = (1, 0, 0), l1 = 2
    // r2 = (0.1, 2, 5), n2 = (1, 0, 0), l2 = 3
    //
    // The shortest distance between the two cells is achieved halfway between
    // the two vectors from (-1, 0, 5) to (-1, 2, 5) and from (1, 0, 5) to
    // (1, 2, 5), which is the vector from (0, 0, 5) to (0, 2, 5)
    //
    // This is achieved at the cell-body coordinates s = 0 and t = -0.1
    target_s = 0; 
    target_t = -0.1; 
    testDistBetweenCells(
        0, 0, 5, 1, 0, 0, 1.0,      // Cell 1
        0.1, 2, 5, 1, 0, 0, 1.5,    // Cell 2
        target_dx, target_dy, target_dz, target_s, target_t
    );

    // The same configuration as the previous, except with l2 = 0.6
    //
    // The shortest distance between the two cells is now achieved halfway
    // between the two vectors from (-0.2, 0, 5) to (-0.2, 2, 5) and from
    // (0.4, 0, 5) to (0.4, 2, 5), which is the vector from (0.1, 0, 5) to
    // (0.1, 2, 5)
    //
    // This is achieved at the cell-body coordinates s = 0.1 and t = 0
    target_s = 0.1;
    target_t = 0;
    testDistBetweenCells(
        0, 0, 5, 1, 0, 0, 1.0,     // Cell 1
        0.1, 2, 5, 1, 0, 0, 0.6,   // Cell 2
        target_dx, target_dy, target_dz, target_s, target_t
    );

    // r1 = (0, 0, 10), n1 = (1, 0, 0), l1 = 2
    // r2 = (0.5, 2, 10), n2 = (1, 0, 0), l2 = 2.2
    //
    // The shortest distance between the two cells is achieved halfway between
    // the two vectors from (-0.6, 0, 10) to (-0.6, 2, 10) and from (1, 0, 10)
    // to (1, 2, 10), which is the vector from (0.2, 0, 10) to (0.2, 2, 10)
    //
    // This is achieved at the cell-body coordinates s = 0.2 and t = -0.3
    target_s = 0.2; 
    target_t = -0.3; 
    testDistBetweenCells(
        0, 0, 10, 1, 0, 0, 1.0,      // Cell 1
        0.5, 2, 10, 1, 0, 0, 1.1,    // Cell 2
        target_dx, target_dy, target_dz, target_s, target_t
    );

    // The same configuration as the previous, but with the orientation of 
    // cell 2 reversed 
    //
    // The shortest distance between the two cells again runs from (0.2, 0)
    // to (0.2, 2), which is achieved at s = 0.2 and t = 0.3
    testDistBetweenCells(
        0, 0, 10, 1, 0, 0, 1.0,       // Cell 1
        0.5, 2, 10, -1, 0, 0, 1.1,    // Cell 2
        target_dx, target_dy, target_dz, target_s, -target_t
    );

    // The same configuration as the previous, but with the two cells switched
    testDistBetweenCells(
        0.5, 2, 10, 1, 0, 0, 1.1,     // Cell 1
        0, 0, 10, 1, 0, 0, 1.0,       // Cell 2
        -target_dx, -target_dy, target_dz, target_t, target_s
    );

    // r1 = (0, 0, 7), n1 = (1, 0, 0), l1 = 2
    // r2 = (5, 2, 7), n1 = (1, 0, 0), l2 = 3
    //
    // The shortest distance between the two cells is achieved at the two 
    // endpoints, s = 1 and t = -1.5
    target_s = 1;
    target_t = -1.5; 
    target_dx = (5 + target_t * 1) - (0 + target_s * 1); 
    target_dy = 2;
    target_dz = 0;  
    testDistBetweenCells(
        0, 0, 7, 1, 0, 0, 1.0,      // Cell 1
        5, 2, 7, 1, 0, 0, 1.5,      // Cell 2
        target_dx, target_dy, target_dz, target_s, target_t
    );

    // The same configuration as the previous, but with the orientation of 
    // cell 1 reversed
    //
    // The shortest distance between the two cells is again achieved at the
    // two endpoints, s = -1 and t = -1.5
    testDistBetweenCells(
        0, 0, 7, -1, 0, 0, 1.0,    // Cell 1
        5, 2, 7, 1, 0, 0, 1.5,     // Cell 2
        target_dx, target_dy, target_dz, -target_s, target_t
    );

    // r1 = (0, 0, 0.5), n1 = (1, 0, 0), l1 = 2
    // r2 = (5, 0, 0.5), n1 = (1, 0, 0), l2 = 3
    //
    // The shortest distance between the two cells is achieved at the two
    // endpoints, s = 1 and t = -1.5
    target_s = 1;
    target_t = -1.5; 
    target_dx = (5 + target_t * 1) - (0 + target_s * 1); 
    target_dy = 0;
    target_dz = 0;  
    testDistBetweenCells(
        0, 0, 0.5, 1, 0, 0, 1.0,    // Cell 1
        5, 0, 0.5, 1, 0, 0, 1.5,    // Cell 2
        target_dx, target_dy, target_dz, target_s, target_t
    );
}

