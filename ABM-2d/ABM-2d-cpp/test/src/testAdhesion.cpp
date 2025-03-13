/**
 * Test module for the `forcesJKRLagrange()`, `forceJKRNewton()`,
 * `forcesKiharaLagrange()`, `forceKiharaNewton()`, `forcesGBKLagrange()`,
 * and `forcesGBKNewton()` functions.  
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
 * simplified JKR potential on the given two cells, computed via finite
 * difference approximation from the potential.  
 *
 * @param r1 Cell 1 center. 
 * @param n1 Cell 1 orientation. 
 * @param half_l1 Cell 1 half-length. 
 * @param r2 Cell 2 center. 
 * @param n2 Cell 2 orientation. 
 * @param half_l2 Cell 2 half-length.
 * @param R Cell radius, including the EPS.
 * @param dmin Minimum distance at which the potential is nonzero. 
 * @param delta Increment for finite difference approximation. 
 * @returns Array of generalized forces and torques on the two cells. 
 */
Array<T, 2, 4> forcesJKRFiniteDiff(const Ref<const Array<T, 2, 1> >& r1,
                                   const Ref<const Array<T, 2, 1> >& n1,
                                   const T half_l1,
                                   const Ref<const Array<T, 2, 1> >& r2, 
                                   const Ref<const Array<T, 2, 1> >& n2,
                                   const T half_l2, const T R, const T dmin,
                                   const T delta)
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
    dEdq(0, 0) = potentialJKR<T>(dist1, R, dmin) - potentialJKR<T>(dist2, R, dmin); 
    dEdq(0, 1) = potentialJKR<T>(dist3, R, dmin) - potentialJKR<T>(dist4, R, dmin); 

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
    dEdq(0, 2) = potentialJKR<T>(dist1, R, dmin) - potentialJKR<T>(dist2, R, dmin); 
    dEdq(0, 3) = potentialJKR<T>(dist3, R, dmin) - potentialJKR<T>(dist4, R, dmin); 
    
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
    dEdq(1, 0) = potentialJKR<T>(dist1, R, dmin) - potentialJKR<T>(dist2, R, dmin); 
    dEdq(1, 1) = potentialJKR<T>(dist3, R, dmin) - potentialJKR<T>(dist4, R, dmin);

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
    dEdq(1, 2) = potentialJKR<T>(dist1, R, dmin) - potentialJKR<T>(dist2, R, dmin); 
    dEdq(1, 3) = potentialJKR<T>(dist3, R, dmin) - potentialJKR<T>(dist4, R, dmin);

    // Normalize by double the increment and return 
    return dEdq / (2 * delta); 
}

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
 * @param dmin Minimum distance at which the potential is nonzero. 
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

/**
 * Returns the partial derivatives of the first anisotropy parameter in the
 * Gay-Berne-Kihara potential on the given two cells, computed via finite 
 * difference approximation. 
 *
 * @param n1 Cell 1 orientation. 
 * @param half_l1 Cell 1 half-length.
 * @param n2 Cell 2 orientation. 
 * @param half_l2 Cell 2 half-length. 
 * @param Rcell Cell radius, excluding the EPS. 
 * @param exp Exponent.
 * @param delta Increment for finite difference approximation. 
 * @returns Array of partial derivatives of the first anisotropy parameter. 
 */
Array<T, 2, 4> anisotropyParamGBK1FiniteDiff(const Ref<const Matrix<T, 2, 1> >& n1,
                                             const T half_l1, 
                                             const Ref<const Matrix<T, 2, 1> >& n2,
                                             const T half_l2, const T Rcell,
                                             const T exp, const T delta)
{
    Array<T, 2, 4> dedq = Array<T, 2, 4>::Zero();
    Matrix<T, 2, 1> dx, dy; 
    dx << delta, 0; 
    dy << 0, delta;

    // Compute all four nonzero finite differences with respect to the 
    // orientational coordinates ... 
    //
    // 1) Partial derivatives w.r.t n1
    dedq(0, 2) = (
        anisotropyParamGBK1<T, 2>(n1 + dx, half_l1, n2, half_l2, Rcell, exp) -
        anisotropyParamGBK1<T, 2>(n1 - dx, half_l1, n2, half_l2, Rcell, exp)
    ); 
    dedq(0, 3) = (
        anisotropyParamGBK1<T, 2>(n1 + dy, half_l1, n2, half_l2, Rcell, exp) -
        anisotropyParamGBK1<T, 2>(n1 - dy, half_l1, n2, half_l2, Rcell, exp)
    ); 

    // 2) Partial derivatives w.r.t n2
    dedq(1, 2) = (
        anisotropyParamGBK1<T, 2>(n1, half_l1, n2 + dx, half_l2, Rcell, exp) -
        anisotropyParamGBK1<T, 2>(n1, half_l1, n2 - dx, half_l2, Rcell, exp)
    ); 
    dedq(1, 3) = (
        anisotropyParamGBK1<T, 2>(n1, half_l1, n2 + dy, half_l2, Rcell, exp) -
        anisotropyParamGBK1<T, 2>(n1, half_l1, n2 - dy, half_l2, Rcell, exp)
    );

    return dedq / (2 * delta);  
} 

/**
 * Returns the generalized attractive forces and torques arising from the
 * Gay-Berne-Kihara potential on the given two cells, computed via finite
 * difference approximation from the potential.  
 *
 * @param r1 Cell 1 center. 
 * @param n1 Cell 1 orientation. 
 * @param half_l1 Cell 1 half-length. 
 * @param r2 Cell 2 center. 
 * @param n2 Cell 2 orientation. 
 * @param half_l2 Cell 2 half-length.
 * @param R Cell radius, including the EPS.
 * @param Rcell Cell radius, excluding the EPS. 
 * @param expd Exponent determining the distance dependence in the potential. 
 * @param dmin Minimum distance at which the potential is nonzero. 
 * @param delta Increment for finite difference approximation. 
 * @returns Array of generalized forces and torques on the two cells. 
 */
Array<T, 2, 4> forcesGBKFiniteDiff(const Ref<const Array<T, 2, 1> >& r1,
                                   const Ref<const Array<T, 2, 1> >& n1,
                                   const T half_l1,
                                   const Ref<const Array<T, 2, 1> >& r2, 
                                   const Ref<const Array<T, 2, 1> >& n2,
                                   const T half_l2, const T R, const T Rcell, 
                                   const T expd, const T dmin, const T delta)
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
    dEdq(0, 0) = (
        potentialGBK<T, 2>(
            (r1 + dx).matrix(), n1.matrix(), half_l1, r2.matrix(), n2.matrix(),
            half_l2, R, Rcell, dist1, expd, 1.0, dmin
        ) -
        potentialGBK<T, 2>(
            (r1 - dx).matrix(), n1.matrix(), half_l1, r2.matrix(), n2.matrix(),
            half_l2, R, Rcell, dist2, expd, 1.0, dmin
        )
    ); 
    dEdq(0, 1) = (
        potentialGBK<T, 2>(
            (r1 + dy).matrix(), n1.matrix(), half_l1, r2.matrix(), n2.matrix(),
            half_l2, R, Rcell, dist3, expd, 1.0, dmin
        ) -
        potentialGBK<T, 2>(
            (r1 - dy).matrix(), n1.matrix(), half_l1, r2.matrix(), n2.matrix(),
            half_l2, R, Rcell, dist4, expd, 1.0, dmin
        )
    );

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
    dEdq(0, 2) = (
        potentialGBK<T, 2>(
            r1.matrix(), (n1 + dx).matrix(), half_l1, r2.matrix(), n2.matrix(),
            half_l2, R, Rcell, dist1, expd, 1.0, dmin
        ) -
        potentialGBK<T, 2>(
            r1.matrix(), (n1 - dx).matrix(), half_l1, r2.matrix(), n2.matrix(),
            half_l2, R, Rcell, dist2, expd, 1.0, dmin
        )
    ); 
    dEdq(0, 3) = (
        potentialGBK<T, 2>(
            r1.matrix(), (n1 + dy).matrix(), half_l1, r2.matrix(), n2.matrix(), 
            half_l2, R, Rcell, dist3, expd, 1.0, dmin
        ) -
        potentialGBK<T, 2>(
            r1.matrix(), (n1 - dy).matrix(), half_l1, r2.matrix(), n2.matrix(),
            half_l2, R, Rcell, dist4, expd, 1.0, dmin
        )
    ); 
    
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
    dEdq(1, 0) = (
        potentialGBK<T, 2>(
            r1.matrix(), n1.matrix(), half_l1, (r2 + dx).matrix(), n2.matrix(),
            half_l2, R, Rcell, dist1, expd, 1.0, dmin
        ) -
        potentialGBK<T, 2>(
            r1.matrix(), n1.matrix(), half_l1, (r2 - dx).matrix(), n2.matrix(),
            half_l2, R, Rcell, dist2, expd, 1.0, dmin
        )
    ); 
    dEdq(1, 1) = (
        potentialGBK<T, 2>(
            r1.matrix(), n1.matrix(), half_l1, (r2 + dy).matrix(), n2.matrix(),
            half_l2, R, Rcell, dist3, expd, 1.0, dmin
        ) -
        potentialGBK<T, 2>(
            r1.matrix(), n1.matrix(), half_l1, (r2 - dy).matrix(), n2.matrix(),
            half_l2, R, Rcell, dist4, expd, 1.0, dmin
        )
    ); 

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
    dEdq(1, 2) = (
        potentialGBK<T, 2>(
            r1.matrix(), n1.matrix(), half_l1, r2.matrix(), (n2 + dx).matrix(),
            half_l2, R, Rcell, dist1, expd, 1.0, dmin
        ) -
        potentialGBK<T, 2>(
            r1.matrix(), n1.matrix(), half_l1, r2.matrix(), (n2 - dx).matrix(),
            half_l2, R, Rcell, dist2, expd, 1.0, dmin
        )
    ); 
    dEdq(1, 3) = (
        potentialGBK<T, 2>(
            r1.matrix(), n1.matrix(), half_l1, r2.matrix(), (n2 + dy).matrix(),
            half_l2, R, Rcell, dist3, expd, 1.0, dmin
        ) -
        potentialGBK<T, 2>(
            r1.matrix(), n1.matrix(), half_l1, r2.matrix(), (n2 - dy).matrix(),
            half_l2, R, Rcell, dist4, expd, 1.0, dmin
        )
    ); 

    // Normalize by double the increment and return 
    return dEdq / (2 * delta); 
}

/* ------------------------------------------------------------------ //
 *                       GENERIC TEST FUNCTIONS                       //
 * ------------------------------------------------------------------ */
/**
 * A generic test function for forcesJKRLagrange(). 
 *
 * @param r1 Cell 1 center. 
 * @param n1 Cell 1 orientation. 
 * @param half_l1 Cell 1 half-length. 
 * @param r2 Cell 2 center. 
 * @param n2 Cell 2 orientation. 
 * @param half_l2 Cell 2 half-length. 
 * @param R Cell radius, including the EPS. 
 * @param dmin Minimum distance at which the potential is nonzero. 
 * @param delta Increment for finite difference approximation. 
 * @param target_force_21 Pre-computed force vector on cell 1 due to cell 2. 
 */
void testForcesJKRLagrange(const Ref<const Array<T, 2, 1> >& r1,
                           const Ref<const Array<T, 2, 1> >& n1, const T half_l1,
                           const Ref<const Array<T, 2, 1> >& r2, 
                           const Ref<const Array<T, 2, 1> >& n2, const T half_l2,
                           const T R, const T dmin, const T delta, 
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

    // Compute the forces via forcesJKRLagrange() *without* enforcing the
    // orientation vector norm constraint 
    Array<T, 2, 4> forces1_unconstrained = -forcesJKRLagrange<T, 2>(
        n1.matrix(), n2.matrix(), d12, R, s, t, dmin, false
    );

    // Do the same *while* enforcing the orientation vector norm constraint 
    Array<T, 2, 4> forces1_constrained = -forcesJKRLagrange<T, 2>(
        n1.matrix(), n2.matrix(), d12, R, s, t, dmin, true
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

    // Check that the two torque vectors are zero, if either is expected 
    //
    // The torque on either cell should be zero if:
    // (1) the force is acting on the center of the cell and the force vector
    //     is orthogonal to the cell orientation, or 
    // (2) the force is acting on either end of the cell and the force vector 
    //     is parallel to the cell orientation
    Matrix<T, 2, 1> dnorm = d12 / dist;
    bool cond1 = (abs(s) < delta && abs(dnorm.dot(n1.matrix())) < delta); 
    bool cond2 = (half_l1 - abs(s) < delta && 1.0 - abs(dnorm.dot(n1.matrix())) < delta);
    bool cond3 = (abs(t) < delta && abs(dnorm.dot(n2.matrix())) < delta); 
    bool cond4 = (half_l2 - abs(t) < delta && 1.0 - abs(dnorm.dot(n2.matrix())) < delta);
    if (cond1 || cond2) 
    {
        REQUIRE_THAT(
            forces1_constrained(0, Eigen::seq(2, 3)).matrix().norm(),
            Catch::Matchers::WithinAbs(0.0, delta)
        );
    }
    if (cond3 || cond4)
    {
        REQUIRE_THAT(
            forces1_constrained(1, Eigen::seq(2, 3)).matrix().norm(),
            Catch::Matchers::WithinAbs(0.0, delta)
        );         
    }

    // Compute the forces via finite differences and check against the 
    // unconstrained forces  
    Array<T, 2, 4> dEdq2 = forcesJKRFiniteDiff(
        r1, n1, half_l1, r2, n2, half_l2, R, dmin, delta
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
 * A generic test function for forceJKRNewton().
 *
 * This function compares the calculated forces against those obtained from 
 * forcesJKRLagrange().  
 *
 * @param r1 Cell 1 center. 
 * @param n1 Cell 1 orientation. 
 * @param half_l1 Cell 1 half-length. 
 * @param r2 Cell 2 center. 
 * @param n2 Cell 2 orientation. 
 * @param half_l2 Cell 2 half-length. 
 * @param R Cell radius, including the EPS. 
 * @param dmin Minimum distance at which the potential is nonzero. 
 */
void testForceJKRNewton(const Ref<const Array<T, 2, 1> >& r1,
                        const Ref<const Array<T, 2, 1> >& n1, const T half_l1,
                        const Ref<const Array<T, 2, 1> >& r2, 
                        const Ref<const Array<T, 2, 1> >& n2, const T half_l2,
                        const T R, const T dmin) 
{
    // Compute the distance vector from cell 1 to cell 2 
    K kernel; 
    Segment_3 seg1 = generateSegment<T>(r1, n1, half_l1); 
    Segment_3 seg2 = generateSegment<T>(r2, n2, half_l2);
    auto result = distBetweenCells<T>(seg1, seg2, 0, r1, n1, half_l1, 1, r2, n2, half_l2, kernel);
    Matrix<T, 2, 1> d12 = std::get<0>(result); 
    T s = std::get<1>(result); 
    T t = std::get<2>(result);

    // Compute the forces via forcesJKRLagrange()
    Array<T, 2, 4> forces1 = -forcesJKRLagrange<T, 2>(
        n1.matrix(), n2.matrix(), d12, R, s, t, dmin, true
    );

    // Compute the force vector on cell 1 due to cell 2 via forceJKRNewton() 
    Array<T, 2, 1> force_21 = forceJKRNewton<T, 2>(d12, R, dmin);

    // Compute the force vector on cell 2 due to cell 1
    Array<T, 2, 1> force_12 = -force_21; 

    // Compute the torque on cell 1 due to cell 2 
    Matrix<T, 3, 1> u1, u2, v1, v2, cross1, cross2; 
    u1 << n1(0), n1(1), 0; 
    u2 << n2(0), n2(1), 0; 
    v1 << force_21(0), force_21(1), 0; 
    v2 << force_12(0), force_12(1), 0; 
    cross1 = (s * u1.cross(v1)).cross(u1);
    cross2 = (t * u2.cross(v2)).cross(u2); 

    // Check that the force vectors match 
    REQUIRE_THAT(
        (forces1(0, Eigen::seq(0, 1)) - force_21.transpose()).matrix().norm(),
        Catch::Matchers::WithinAbs(0.0, 1e-8)
    );
    REQUIRE_THAT(
        (forces1(1, Eigen::seq(0, 1)) - force_12.transpose()).matrix().norm(), 
        Catch::Matchers::WithinAbs(0.0, 1e-8)
    );

    // Check that the torque vectors match 
    REQUIRE_THAT(
        (cross1.head(2) - forces1(0, Eigen::seq(2, 3)).matrix().transpose()).norm(),
        Catch::Matchers::WithinAbs(0.0, 1e-8)
    ); 
    REQUIRE_THAT(
        (cross2.head(2) - forces1(1, Eigen::seq(2, 3)).matrix().transpose()).norm(),
        Catch::Matchers::WithinAbs(0.0, 1e-8)
    ); 
}

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
 * @param dmin Minimum distance at which the potential is nonzero. 
 * @param delta Increment for finite difference approximation. 
 * @param target_force_21 Pre-computed force vector on cell 1 due to cell 2. 
 */
void testForcesKiharaLagrange(const Ref<const Array<T, 2, 1> >& r1,
                              const Ref<const Array<T, 2, 1> >& n1, const T half_l1,
                              const Ref<const Array<T, 2, 1> >& r2, 
                              const Ref<const Array<T, 2, 1> >& n2, const T half_l2,
                              const T R, const T exp, const T dmin, const T delta, 
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

    // Check that the two torque vectors are zero, if either is expected 
    //
    // The torque on either cell should be zero if:
    // (1) the force is acting on the center of the cell and the force vector
    //     is orthogonal to the cell orientation, or 
    // (2) the force is acting on either end of the cell and the force vector 
    //     is parallel to the cell orientation
    Matrix<T, 2, 1> dnorm = d12 / dist;
    bool cond1 = (abs(s) < delta && abs(dnorm.dot(n1.matrix())) < delta); 
    bool cond2 = (half_l1 - abs(s) < delta && 1.0 - abs(dnorm.dot(n1.matrix())) < delta);
    bool cond3 = (abs(t) < delta && abs(dnorm.dot(n2.matrix())) < delta); 
    bool cond4 = (half_l2 - abs(t) < delta && 1.0 - abs(dnorm.dot(n2.matrix())) < delta);
    if (cond1 || cond2) 
    {
        REQUIRE_THAT(
            forces1_constrained(0, Eigen::seq(2, 3)).matrix().norm(),
            Catch::Matchers::WithinAbs(0.0, delta)
        );
    }
    if (cond3 || cond4)
    {
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
 * A generic test function for forceKiharaNewton().
 *
 * This function compares the calculated forces against those obtained from 
 * forcesKiharaLagrange().  
 *
 * @param r1 Cell 1 center. 
 * @param n1 Cell 1 orientation. 
 * @param half_l1 Cell 1 half-length. 
 * @param r2 Cell 2 center. 
 * @param n2 Cell 2 orientation. 
 * @param half_l2 Cell 2 half-length. 
 * @param R Cell radius, including the EPS. 
 * @param exp Exponent in Kihara potential. 
 * @param dmin Minimum distance at which the potential is nonzero. 
 */
void testForceKiharaNewton(const Ref<const Array<T, 2, 1> >& r1,
                           const Ref<const Array<T, 2, 1> >& n1, const T half_l1,
                           const Ref<const Array<T, 2, 1> >& r2, 
                           const Ref<const Array<T, 2, 1> >& n2, const T half_l2,
                           const T R, const T exp, const T dmin) 
{
    // Compute the distance vector from cell 1 to cell 2 
    K kernel; 
    Segment_3 seg1 = generateSegment<T>(r1, n1, half_l1); 
    Segment_3 seg2 = generateSegment<T>(r2, n2, half_l2);
    auto result = distBetweenCells<T>(seg1, seg2, 0, r1, n1, half_l1, 1, r2, n2, half_l2, kernel);
    Matrix<T, 2, 1> d12 = std::get<0>(result); 
    T s = std::get<1>(result); 
    T t = std::get<2>(result);

    // Compute the forces via forcesKiharaLagrange()
    Array<T, 2, 4> forces1 = -forcesKiharaLagrange<T, 2>(
        n1.matrix(), n2.matrix(), d12, R, s, t, exp, dmin, true
    );

    // Compute the force vector on cell 1 due to cell 2 via forceKiharaNewton() 
    Array<T, 2, 1> force_21 = forceKiharaNewton<T, 2>(d12, R, exp, dmin);

    // Compute the force vector on cell 2 due to cell 1
    Array<T, 2, 1> force_12 = -force_21; 

    // Compute the torque on cell 1 due to cell 2 
    Matrix<T, 3, 1> u1, u2, v1, v2, cross1, cross2; 
    u1 << n1(0), n1(1), 0; 
    u2 << n2(0), n2(1), 0; 
    v1 << force_21(0), force_21(1), 0; 
    v2 << force_12(0), force_12(1), 0; 
    cross1 = (s * u1.cross(v1)).cross(u1);
    cross2 = (t * u2.cross(v2)).cross(u2); 

    // Check that the force vectors match 
    REQUIRE_THAT(
        (forces1(0, Eigen::seq(0, 1)) - force_21.transpose()).matrix().norm(),
        Catch::Matchers::WithinAbs(0.0, 1e-8)
    );
    REQUIRE_THAT(
        (forces1(1, Eigen::seq(0, 1)) - force_12.transpose()).matrix().norm(), 
        Catch::Matchers::WithinAbs(0.0, 1e-8)
    );

    // Check that the torque vectors match 
    REQUIRE_THAT(
        (cross1.head(2) - forces1(0, Eigen::seq(2, 3)).matrix().transpose()).norm(),
        Catch::Matchers::WithinAbs(0.0, 1e-8)
    ); 
    REQUIRE_THAT(
        (cross2.head(2) - forces1(1, Eigen::seq(2, 3)).matrix().transpose()).norm(),
        Catch::Matchers::WithinAbs(0.0, 1e-8)
    ); 
}

/**
 * A generic test function for forcesGBKLagrange(). 
 *
 * @param r1 Cell 1 center. 
 * @param n1 Cell 1 orientation. 
 * @param half_l1 Cell 1 half-length. 
 * @param r2 Cell 2 center. 
 * @param n2 Cell 2 orientation. 
 * @param half_l2 Cell 2 half-length. 
 * @param R Cell radius, including the EPS.
 * @param Rcell Cell radius, excluding the EPS.   
 * @param expd Exponent determining the distance dependence in the potential.
 * @param dmin Minimum distance at which the potential is nonzero. 
 * @param delta Increment for finite difference approximation. 
 * @param target_force_21 Pre-computed force vector on cell 1 due to cell 2. 
 */
void testForcesGBKLagrange(const Ref<const Array<T, 2, 1> >& r1,
                           const Ref<const Array<T, 2, 1> >& n1, const T half_l1,
                           const Ref<const Array<T, 2, 1> >& r2, 
                           const Ref<const Array<T, 2, 1> >& n2, const T half_l2,
                           const T R, const T Rcell, const T expd, const T dmin,
                           const T delta, 
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

    // Compute the forces via forcesGBKLagrange() *without* enforcing 
    // the orientation vector norm constraint 
    Array<T, 2, 4> forces1_unconstrained = -forcesGBKLagrange<T, 2>(
        r1.matrix(), n1.matrix(), half_l1, r2.matrix(), n2.matrix(), half_l2, 
        R, Rcell, d12, s, t, expd, 1.0, dmin, false
    );

    // Do the same *while* enforcing the orientation vector norm constraint 
    Array<T, 2, 4> forces1_constrained = -forcesGBKLagrange<T, 2>(
        r1.matrix(), n1.matrix(), half_l1, r2.matrix(), n2.matrix(), half_l2,
        R, Rcell, d12, s, t, expd, 1.0, dmin, true
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
    // To compute this, we need to follow three steps: 
    // 1) Calculate the derivative of the anisotropy parameter w.r.t n1 
    // 2) Use the product rule, deps * U + eps * dU, where eps is the
    //    anisotropy parameter and U is the *Kihara* potential 
    // 3) Apply the Lagrangian constraint on the orientation vector norm 
    //    constraint
    T eps = anisotropyParamGBK1<T, 2>(n1, half_l1, n2, half_l2, Rcell, 1.0); 
    Array<T, 2, 4> deps = anisotropyParamGBK1FiniteDiff(
        n1, half_l1, n2, half_l2, Rcell, 1.0, delta
    );
    T Ek = potentialKihara<T>(dist, R, expd, dmin);
    Array<T, 2, 4> dEkdq = forcesKiharaLagrange<T, 2>(
        n1.matrix(), n2.matrix(), d12, R, s, t, expd, dmin, false
    ); 
    Array<T, 2, 1> dEdn1 = deps(0, Eigen::seq(2, 3)) * Ek + eps * dEkdq(0, Eigen::seq(2, 3));
    REQUIRE_THAT(
        (forces1_unconstrained(0, Eigen::seq(2, 3)) + dEdn1.transpose()).abs().maxCoeff(), 
        Catch::Matchers::WithinAbs(0.0, delta)
    );
    Array<T, 2, 1> dEdn1_constrained = dEdn1 - 2 * lambda1 * n1; 
    REQUIRE_THAT(
        (forces1_constrained(0, Eigen::seq(2, 3)) + dEdn1_constrained.transpose()).abs().maxCoeff(), 
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

    // Check that the two torque vectors are equal if the two cells are parallel
    bool parallel = (1.0 - abs(n1.matrix().dot(n2.matrix())) < delta); 
    if (parallel)
    {
        REQUIRE_THAT(
            (forces1_constrained(0, Eigen::seq(2, 3)) - forces1_constrained(1, Eigen::seq(2, 3))).abs().maxCoeff(),
            Catch::Matchers::WithinAbs(0.0, delta)
        );
    }

    // Compute the forces via finite differences and check against the 
    // unconstrained forces  
    Array<T, 2, 4> dEdq2 = forcesGBKFiniteDiff(
        r1, n1, half_l1, r2, n2, half_l2, R, Rcell, expd, dmin, delta
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
 * A generic test function for forceGBKNewton() and torqueGBKNewton().
 *
 * This function compares the calculated forces and torques against those
 * obtained from forcesGBKLagrange().  
 *
 * @param r1 Cell 1 center. 
 * @param n1 Cell 1 orientation. 
 * @param half_l1 Cell 1 half-length. 
 * @param r2 Cell 2 center. 
 * @param n2 Cell 2 orientation. 
 * @param half_l2 Cell 2 half-length.
 * @param R Cell radius, including the EPS.
 * @param Rcell Cell radius, excluding the EPS. 
 * @param expd Exponent determining the distance dependence in the potential.
 * @param dmin Minimum distance at which the potential is nonzero.
 */
void testForceGBKNewton(const Ref<const Array<T, 2, 1> >& r1,
                        const Ref<const Array<T, 2, 1> >& n1, const T half_l1,
                        const Ref<const Array<T, 2, 1> >& r2, 
                        const Ref<const Array<T, 2, 1> >& n2, const T half_l2,
                        const T R, const T Rcell, const T expd, const T dmin) 
{
    // Compute the distance vector from cell 1 to cell 2 
    K kernel; 
    Segment_3 seg1 = generateSegment<T>(r1, n1, half_l1); 
    Segment_3 seg2 = generateSegment<T>(r2, n2, half_l2);
    auto result = distBetweenCells<T>(seg1, seg2, 0, r1, n1, half_l1, 1, r2, n2, half_l2, kernel);
    Matrix<T, 2, 1> d12 = std::get<0>(result); 
    T s = std::get<1>(result); 
    T t = std::get<2>(result);

    // Compute the forces via forcesGBKLagrange()
    Array<T, 2, 4> forces1 = -forcesGBKLagrange<T, 2>(
        r1.matrix(), n1.matrix(), half_l1, r2.matrix(), n2.matrix(), half_l2, 
        R, Rcell, d12, s, t, expd, 1.0, dmin, true
    );

    // Compute the force vector on cell 1 due to cell 2 via forceGBKNewton() 
    Array<T, 2, 1> force_21 = forceGBKNewton<T, 2>(
        n1.matrix(), half_l1, n2.matrix(), half_l2, R, Rcell, d12, expd, 1.0,
        dmin
    );

    // Compute the torque vector on cell 1 due to cell 2 and vice versa via
    // torqueGBKNewton()
    Array<T, 3, 1> torque_21 = torqueGBKNewton<T, 2>(
        n1.matrix(), half_l1, n2.matrix(), half_l2, R, Rcell, d12, s, t, expd,
        1.0, dmin
    ); 
    Array<T, 3, 1> torque_12 = torqueGBKNewton<T, 2>(
        n2.matrix(), half_l2, n1.matrix(), half_l1, R, Rcell, -d12, t, s, expd,
        1.0, dmin
    ); 

    // Compute the force vector on cell 2 due to cell 1
    Array<T, 2, 1> force_12 = -force_21; 

    // Check that the force vectors match
    REQUIRE_THAT(
        (forces1(0, Eigen::seq(0, 1)) - force_21.transpose()).matrix().norm(),
        Catch::Matchers::WithinAbs(0.0, 1e-8)
    );
    REQUIRE_THAT(
        (forces1(1, Eigen::seq(0, 1)) - force_12.transpose()).matrix().norm(), 
        Catch::Matchers::WithinAbs(0.0, 1e-8)
    );

    // Check that the torque vectors match
    Matrix<T, 3, 1> u1, u2; 
    u1 << n1(0), n1(1), 0; 
    u2 << n2(0), n2(1), 0; 
    REQUIRE_THAT(
        (torque_21.matrix().cross(u1).head(2) - forces1(0, Eigen::seq(2, 3)).matrix().transpose()).norm(),
        Catch::Matchers::WithinAbs(0.0, 1e-8)
    ); 
    REQUIRE_THAT(
        (torque_12.matrix().cross(u2).head(2) - forces1(1, Eigen::seq(2, 3)).matrix().transpose()).norm(),
        Catch::Matchers::WithinAbs(0.0, 1e-8)
    );
}

/* ------------------------------------------------------------------ //
 *           TEST MODULES FOR FORCE PARAMETER CALIBRATION             //
 * ------------------------------------------------------------------ */
/**
 * A series of tests for JKR potential/force parameter calibration. 
 */
TEST_CASE(
    "Tests for JKR potential/force parameter calibration",
    "[potentialJKR(), forceJKRNewton()]"
)
{
    const T R = 0.8;
    const T Rcell = 0.5; 
    const T E0 = 3900.0;
    const T Ecell = 1000.0 * E0; 
    Array<T, 2, 1> dnorm; 
    dnorm << 1, 0;

    // Define cell 1: r1 = (0, 0), n1 = (1, 0), l1 = 1
    Array<T, 2, 1> r1, n1, r2, n2; 
    r1 << 0, 0; 
    n1 << 1, 0; 

    // Define an array of distances 
    const int n = 1000; 
    Array<T, Dynamic, 1> dists = Array<T, Dynamic, 1>::LinSpaced(n, 2 * Rcell - 0.05, 2 * R + 0.05);
    std::vector<T> dmins { 1.13, 1.3, 1.52 };
    std::vector<T> coefs { 2600.0, 2900.0, 7000.0 };

    // Define output arrays for potentials and forces 
    Array<T, Dynamic, Dynamic> potentials(n, dmins.size()), forces(n, dmins.size());
    
    std::cout << "Parameter calibration for JKR potentials" << std::endl; 
    std::cout << "----------------------------------------" << std::endl; 

    // For each distance and exponent ...
    for (int j = 0; j < dmins.size(); ++j)
    {
        for (int i = 0; i < n; ++i)
        {
            // Calculate the corresponding hybrid potential and force for 
            // a second cell the given distance vector away  
            Matrix<T, 2, 1> d12; 
            d12 << dists(i), 0; 
            potentials(i, j) = (
                potentialHertz<T>(dists(i), R, Rcell, E0, Ecell) +
                coefs[j] * potentialJKR<T>(dists(i), R, dmins[j])
            );
            Array<T, 2, 1> force = coefs[j] * forceJKRNewton<T, 2>(d12, R, dmins[j]);
            if (dists(i) < 2 * Rcell)
            {
                force -= 2.5 * (
                    E0 * sqrt(R) * pow(2 * (R - Rcell), 1.5) +
                    Ecell * sqrt(Rcell) * pow(2 * Rcell - dists(i), 1.5)
                ) * dnorm; 
            }
            else if (dists(i) < 2 * R)
            {
                force -= 2.5 * E0 * sqrt(R) * pow(2 * R - dists(i), 1.5) * dnorm; 
            }
            forces(i, j) = force(0);    // Get only the x-coordinate  
        }

        // Find the distance at which the potential is minimized 
        Eigen::Index minpi;
        potentials.col(j).minCoeff(&minpi);

	// Check that the minimum distance is close to 1.2
        std::cout << "Potential for dmin = " << dmins[j] << ", A = " << coefs[j]
                  << ": equilibrium distance = " << dists(minpi)
                  << "; minimum potential value = " << potentials(minpi, j)
                  << "; force at equilibrium distance = " << forces(minpi, j) << std::endl; 

        // Check that the force at this distance is near zero 
        REQUIRE(forces(minpi - 1, j) < 0);
	REQUIRE(forces(minpi + 1, j) > 0);
    }
}

/**
 * A series of tests for Kihara potential/force parameter calibration. 
 */
/*
TEST_CASE(
    "Tests for Kihara potential/force parameter calibration",
    "[potentialKihara(), forceKiharaNewton()]"
)
{
    const T R = 0.8;
    const T Rcell = 0.5; 
    const T E0 = 3900.0;
    const T Ecell = 100.0 * E0; 
    const T dmin = 1.05;
    Array<T, 2, 1> dnorm; 
    dnorm << 1, 0; 

    // Define an array of distances 
    const int n = 1000; 
    Array<T, Dynamic, 1> dists = Array<T, Dynamic, 1>::LinSpaced(n, dmin - 0.05, 2 * R + 0.05);
    std::vector<T> exps { 2.0, 3.0, 4.0, 5.0, 6.0 };
    std::vector<T> coefs { 43000.0, 29000.0, 24000.0, 21000.0, 20000.0 }; 

    // Define output arrays for potentials and forces 
    Array<T, Dynamic, 5> potentials(n, 5), forces(n, 5);

    // For each distance and exponent ...
    for (int j = 0; j < exps.size(); ++j)
    {
        for (int i = 0; i < n; ++i)
        {        
            // Calculate the corresponding hybrid potential and force
            Matrix<T, 2, 1> d12; 
            d12 << dists(i), 0; 
            potentials(i, j) = (
                potentialHertz<T>(dists(i), R, Rcell, E0, Ecell) +
                coefs[j] * potentialKihara<T>(dists(i), R, exps[j], dmin)
            );
            Array<T, 2, 1> force = coefs[j] * forceKiharaNewton<T, 2>(d12, R, exps[j], dmin);
            if (dists(i) < R + Rcell)
            {
                force -= 2.5 * (
                    E0 * sqrt(R) * pow(R - Rcell, 1.5) +
                    Ecell * sqrt(R) * pow(R + Rcell - dists(i), 1.5)
                ) * dnorm; 
            }
            else if (dists(i) < 2 * R)
            {
                force -= 2.5 * E0 * sqrt(R) * pow(2 * R - dists(i), 1.5) * dnorm; 
            }
            forces(i, j) = force(0);    // Get only the x-coordinate  
        }

        // Find the distance at which the potential is minimized 
        Eigen::Index minpi;
        potentials.col(j).minCoeff(&minpi);

	// Check that the minimum distance is close to 1.2 
	REQUIRE_THAT(dists(minpi), Catch::Matchers::WithinAbs(1.2, 0.01)); 

        // Check that the force at this distance is near zero 
        REQUIRE(forces(minpi - 1, j) < 0);
	REQUIRE(forces(minpi + 1, j) > 0);
    }
}
*/

/**
 * A series of tests for Gay-Berne-Kihara potential/force parameter calibration. 
 */
/*
TEST_CASE(
    "Tests for GBK potential/force parameter calibration",
    "[potentialGBK(), forceGBKNewton()]"
)
{
    const T R = 0.8;
    const T Rcell = 0.5; 
    const T E0 = 3900.0;
    const T Ecell = 100.0 * E0; 
    const T dmin = 1.05;
    Array<T, 2, 1> dnorm; 
    dnorm << 1, 0;

    // Define cell 1: r1 = (0, 0), n1 = (1, 0), l1 = 1
    Array<T, 2, 1> r1, n1, r2, n2; 
    r1 << 0, 0; 
    n1 << 1, 0; 

    // Define an array of distances 
    const int n = 1000; 
    Array<T, Dynamic, 1> dists = Array<T, Dynamic, 1>::LinSpaced(n, dmin - 0.05, 2 * R + 0.05);
    std::vector<T> exps { 2.0, 3.0, 4.0, 5.0, 6.0 };
    std::vector<T> coefs { 35000.0, 23000.0, 19000.0, 17000.0, 16000.0 }; 

    // Define output arrays for potentials and forces 
    Array<T, Dynamic, 5> potentials(n, 5), forces(n, 5);

    // For each distance and exponent ...
    for (int j = 0; j < exps.size(); ++j)
    {
        for (int i = 0; i < n; ++i)
        {
	    // Define cell 2: r2 = (1 + dist, 0), n2 = (1, 0), l2 = 1
	    r2 << 1 + dists(i), 0; 
	    n2 << 1, 0; 

            // Calculate the corresponding hybrid potential and force
            Matrix<T, 2, 1> d12; 
            d12 << dists(i), 0; 
            potentials(i, j) = (
                potentialHertz<T>(dists(i), R, Rcell, E0, Ecell) +
                coefs[j] * potentialGBK<T, 2>(
		    r1, n1, 0.5, r2, n2, 0.5, R, Rcell, dists(i), exps[j], 1.0, dmin
		)
            );
            Array<T, 2, 1> force = coefs[j] * forceGBKNewton<T, 2>(
		n1, 0.5, n2, 0.5, R, Rcell, d12, exps[j], 1.0, dmin
	    );
            if (dists(i) < R + Rcell)
            {
                force -= 2.5 * (
                    E0 * sqrt(R) * pow(R - Rcell, 1.5) +
                    Ecell * sqrt(R) * pow(R + Rcell - dists(i), 1.5)
                ) * dnorm; 
            }
            else if (dists(i) < 2 * R)
            {
                force -= 2.5 * E0 * sqrt(R) * pow(2 * R - dists(i), 1.5) * dnorm; 
            }
            forces(i, j) = force(0);    // Get only the x-coordinate  
        }

        // Find the distance at which the potential is minimized 
        Eigen::Index minpi;
        potentials.col(j).minCoeff(&minpi);

	// Check that the minimum distance is close to 1.2 
	REQUIRE_THAT(dists(minpi), Catch::Matchers::WithinAbs(1.2, 0.01)); 

        // Check that the force at this distance is near zero 
        REQUIRE(forces(minpi - 1, j) < 0);
	REQUIRE(forces(minpi + 1, j) > 0);
    }
}
*/

/* ------------------------------------------------------------------ //
 *           TEST MODULES FOR SKEW CELL-CELL CONFIGURATIONS           //
 * ------------------------------------------------------------------ */
/**
 * A series of tests for squaredAspectRatioParam(), anisotropyParamGBK1(), 
 * and anisotropyParamWithDerivsGBK1(). 
 */
TEST_CASE(
    "Tests for GBK helper functions",
    "[squaredAspectRatioParam(), anisotropyParamGBK1(), anisotropyParamWithDerivsGBK1()]"
)
{
    // First try the computation with two skew cells 
    Matrix<T, 2, 1> n1, n2; 
    n1 << sin(boost::math::constants::three_quarters_pi<T>()), cos(boost::math::constants::three_quarters_pi<T>()); 
    n2 << cos(boost::math::constants::sixth_pi<T>()), sin(boost::math::constants::sixth_pi<T>()); 
    const T Rcell = 0.5; 
    const T half_l1 = 0.7; 
    const T half_l2 = 0.9;
    const T chi2 = squaredAspectRatioParam<T>(half_l1, half_l2, Rcell);
    const T chi2_target = (
        ((pow(0.7 + 0.5, 2) - 0.25) * (pow(0.9 + 0.5, 2) - 0.25)) / 
        ((pow(0.9 + 0.5, 2) + 0.25) * (pow(0.7 + 0.5, 2) + 0.25))
    );
    REQUIRE_THAT(chi2, Catch::Matchers::WithinAbs(chi2_target, 1e-8));

    const T exp = 1; 
    const T eps = anisotropyParamGBK1<T, 2>(n1, half_l1, n2, half_l2, Rcell, exp);
    const T eps_target = pow(1.0 - chi2 * n1.dot(n2) * n1.dot(n2), -0.5 * exp);
    REQUIRE_THAT(eps, Catch::Matchers::WithinAbs(eps_target, 1e-8));

    auto result = anisotropyParamWithDerivsGBK1<T, 2>(n1, half_l1, n2, half_l2, Rcell, exp); 
    const T eps2 = result.first; 
    Matrix<T, 2, 4> deps = result.second;
    Array<T, 2, 4> deps_target = anisotropyParamGBK1FiniteDiff(n1, half_l1, n2, half_l2, Rcell, exp, 1e-8);
    REQUIRE_THAT(eps2, Catch::Matchers::WithinAbs(eps_target, 1e-8));
    REQUIRE_THAT(
        (deps.array() - deps_target).abs().maxCoeff(),
        Catch::Matchers::WithinAbs(0.0, 1e-8)
    );

    // TODO More tests with parallel and perpendicular cells  
}

/**
 * A series of tests for forcesJKRLagrange() for skew cell-cell configurations. 
 */
TEST_CASE("Tests for forcesJKRLagrange(), skew cells", "[forcesJKRLagrange()]") 
{
    const T R = 0.8;
    const T dmin = 1.2;
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
    testForcesJKRLagrange(
        r1, n1, 1, r2, n2, 0.5, R, dmin, delta, target_force_21
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
    target_force_21 = d12 * boost::math::constants::pi<T>() * R * (2 * R - dmin); 
    testForcesJKRLagrange(
        r1, n1, 0.5, r2, n2, 0.5, R, dmin, delta, target_force_21
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
    testForcesJKRLagrange(
        r1, n1, 0.5, r2, n2, 0.5, R, dmin, delta, target_force_21
    );

    // r1 = (0, 0), n1 = (1, 0), l1 = 1
    // r2 = (0.5 + 1.4 / sqrt(2), 1.4 / sqrt(2)), n2 = (1 / sqrt(2), -1 / sqrt(2)), l2 = 1
    //
    // The shortest distance between the two cells is (1.4 / sqrt(2), 1.4 / sqrt(2))
    r2(0) = 0.5 + 1.4 / sqrt(2.0); 
    r2(1) = 1.4 / sqrt(2.0); 
    n2(0) = 1.0 / sqrt(2.0); 
    n2(1) = -1.0 / sqrt(2.0);
    d12(0) = 1.4 / sqrt(2.0); 
    d12(1) = 1.4 / sqrt(2.0);
    T dist = d12.matrix().norm(); 
    Array<T, 2, 1> dnorm = d12 / dist;
    target_force_21 = dnorm * boost::math::constants::pi<T>() * R * (2 * R - dist); 
    testForcesJKRLagrange(
        r1, n1, 0.5, r2, n2, 0.5, R, dmin, delta, target_force_21
    ); 
}

/**
 * A series of tests for forceJKRNewton() for skew cell-cell configurations. 
 */
TEST_CASE("Tests for forceJKRNewton(), skew cells", "[forceJKRNewton()]") 
{
    const T R = 0.8;
    const T dmin = 1.2;

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
    testForceJKRNewton(r1, n1, 1, r2, n2, 0.5, R, dmin);

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
    testForceJKRNewton(r1, n1, 0.5, r2, n2, 0.5, R, dmin);

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
    testForceJKRNewton(r1, n1, 0.5, r2, n2, 0.5, R, dmin);

    // r1 = (0, 0), n1 = (1, 0), l1 = 1
    // r2 = (0.5 + 1.4 / sqrt(2), 1.4 / sqrt(2)), n2 = (1 / sqrt(2), -1 / sqrt(2)), l2 = 1
    //
    // The shortest distance between the two cells is (1.4 / sqrt(2), 1.4 / sqrt(2))
    r2(0) = 0.5 + 1.4 / sqrt(2.0); 
    r2(1) = 1.4 / sqrt(2.0); 
    n2(0) = 1.0 / sqrt(2.0); 
    n2(1) = -1.0 / sqrt(2.0);
    testForceJKRNewton(r1, n1, 0.5, r2, n2, 0.5, R, dmin);
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
        r1, n1, 1, r2, n2, 0.5, R, exp, dmin, delta, target_force_21
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
        r1, n1, 0.5, r2, n2, 0.5, R, exp, dmin, delta, target_force_21
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
        r1, n1, 0.5, r2, n2, 0.5, R, exp, dmin, delta, target_force_21
    );

    // r1 = (0, 0), n1 = (1, 0), l1 = 1
    // r2 = (0.5 + 1.4 / sqrt(2), 1.4 / sqrt(2)), n2 = (1 / sqrt(2), -1 / sqrt(2)), l2 = 1
    //
    // The shortest distance between the two cells is (1.4 / sqrt(2), 1.4 / sqrt(2))
    r2(0) = 0.5 + 1.4 / sqrt(2.0); 
    r2(1) = 1.4 / sqrt(2.0); 
    n2(0) = 1.0 / sqrt(2.0); 
    n2(1) = -1.0 / sqrt(2.0);
    d12(0) = 1.4 / sqrt(2.0); 
    d12(1) = 1.4 / sqrt(2.0);
    T dist = d12.matrix().norm(); 
    Array<T, 2, 1> dnorm = d12 / dist;
    target_force_21 = dnorm * exp * (pow(dist, -exp - 1) - pow(2 * R, -exp - 1));
    testForcesKiharaLagrange(
        r1, n1, 0.5, r2, n2, 0.5, R, exp, dmin, delta, target_force_21
    ); 
}

/**
 * A series of tests for forceKiharaNewton() for skew cell-cell configurations. 
 */
TEST_CASE("Tests for forceKiharaNewton(), skew cells", "[forceKiharaNewton()]") 
{
    const T R = 0.8;
    const T exp = 4; 
    const T dmin = 1.05;

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
    testForceKiharaNewton(r1, n1, 1, r2, n2, 0.5, R, exp, dmin);

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
    testForceKiharaNewton(r1, n1, 0.5, r2, n2, 0.5, R, exp, dmin);

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
    testForceKiharaNewton(r1, n1, 0.5, r2, n2, 0.5, R, exp, dmin);

    // r1 = (0, 0), n1 = (1, 0), l1 = 1
    // r2 = (0.5 + 1.4 / sqrt(2), 1.4 / sqrt(2)), n2 = (1 / sqrt(2), -1 / sqrt(2)), l2 = 1
    //
    // The shortest distance between the two cells is (1.4 / sqrt(2), 1.4 / sqrt(2))
    r2(0) = 0.5 + 1.4 / sqrt(2.0); 
    r2(1) = 1.4 / sqrt(2.0); 
    n2(0) = 1.0 / sqrt(2.0); 
    n2(1) = -1.0 / sqrt(2.0);
    testForceKiharaNewton(r1, n1, 0.5, r2, n2, 0.5, R, exp, dmin);
}

/**
 * A series of tests for forcesGBKLagrange() for skew cell-cell configurations. 
 */
TEST_CASE("Tests for forcesGBKLagrange(), skew cells", "[forcesGBKLagrange()]") 
{
    const T R = 0.8;
    const T Rcell = 0.5; 
    const T expd = 4; 
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
    testForcesGBKLagrange(
        r1, n1, 1, r2, n2, 0.5, R, Rcell, expd, dmin, delta, target_force_21
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
    T eps = anisotropyParamGBK1<T, 2>(n1.matrix(), 0.5, n2.matrix(), 0.5, Rcell, 1.0);
    target_force_21 = d12 * eps * expd * (pow(dmin, -expd - 1) - pow(2 * R, -expd - 1));
    testForcesGBKLagrange(
        r1, n1, 0.5, r2, n2, 0.5, R, Rcell, expd, dmin, delta, target_force_21
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
    eps = anisotropyParamGBK1<T, 2>(n1.matrix(), 0.5, n2.matrix(), 0.5, Rcell, 1.0);
    target_force_21 = d12 * eps * expd * (pow(dmin, -expd - 1) - pow(2 * R, -expd - 1));  
    testForcesGBKLagrange(
        r1, n1, 0.5, r2, n2, 0.5, R, Rcell, expd, dmin, delta, target_force_21
    );

    // r1 = (0, 0), n1 = (1, 0), l1 = 1
    // r2 = (0.5 + 1.4 / sqrt(2), 1.4 / sqrt(2)), n2 = (1 / sqrt(2), -1 / sqrt(2)), l2 = 1
    //
    // The shortest distance between the two cells is (1.4 / sqrt(2), 1.4 / sqrt(2))
    r2(0) = 0.5 + 1.4 / sqrt(2.0); 
    r2(1) = 1.4 / sqrt(2.0); 
    n2(0) = 1.0 / sqrt(2.0); 
    n2(1) = -1.0 / sqrt(2.0);
    d12(0) = 1.4 / sqrt(2.0); 
    d12(1) = 1.4 / sqrt(2.0);
    T dist = d12.matrix().norm(); 
    Array<T, 2, 1> dnorm = d12 / dist;
    eps = anisotropyParamGBK1<T, 2>(n1.matrix(), 0.5, n2.matrix(), 0.5, Rcell, 1.0);
    target_force_21 = dnorm * eps * expd * (pow(dist, -expd - 1) - pow(2 * R, -expd - 1));
    testForcesGBKLagrange(
        r1, n1, 0.5, r2, n2, 0.5, R, Rcell, expd, dmin, delta, target_force_21
    );
}

/**
 * A series of tests for forceGBKNewton() for skew cell-cell configurations. 
 */
TEST_CASE("Tests for forceGBKNewton(), skew cells", "[forceGBKNewton()]") 
{
    const T R = 0.8;
    const T Rcell = 0.5;
    const T expd = 4; 
    const T dmin = 1.05;

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
    testForceGBKNewton(r1, n1, 1, r2, n2, 0.5, R, Rcell, expd, dmin);

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
    testForceGBKNewton(r1, n1, 0.5, r2, n2, 0.5, R, Rcell, expd, dmin);

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
    testForceGBKNewton(r1, n1, 0.5, r2, n2, 0.5, R, Rcell, expd, dmin);

    // r1 = (0, 0), n1 = (1, 0), l1 = 1
    // r2 = (0.5 + 1.4 / sqrt(2), 1.4 / sqrt(2)), n2 = (1 / sqrt(2), -1 / sqrt(2)), l2 = 1
    //
    // The shortest distance between the two cells is (1.4 / sqrt(2), 1.4 / sqrt(2))
    r2(0) = 0.5 + 1.4 / sqrt(2.0); 
    r2(1) = 1.4 / sqrt(2.0); 
    n2(0) = 1.0 / sqrt(2.0); 
    n2(1) = -1.0 / sqrt(2.0);
    testForceGBKNewton(r1, n1, 0.5, r2, n2, 0.5, R, Rcell, expd, dmin);
}

/* ------------------------------------------------------------------ //
 *         TEST MODULES FOR PARALLEL CELL-CELL CONFIGURATIONS         //
 * ------------------------------------------------------------------ */
/**
 * A series of tests for forcesKiharaLagrange() for parallel cell-cell
 * configurations.
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
        r1, n1, 0.5, r2, n2, 0.5, R, exp, dmin, delta, target_force_21
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
        r1, n1, 0.5, r2, n2, 0.5, R, exp, dmin, delta, target_force_21
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
        r1, n1, 0.5, r2, n2, 0.5, R, exp, dmin, delta, target_force_21
    );

    // r1 = (0, 0), n1 = (1, 0), l1 = 1
    // r2 = (0.2, 1.5), n2 = (1, 0), l2 = 1
    //
    // The shortest distance between the two cells is, again, the vector
    // (0, 1.5), but the cells are not aligned and there should be nonzero
    // torque
    r2(0) = 0.2;
    testForcesKiharaLagrange(
        r1, n1, 0.5, r2, n2, 0.5, R, exp, dmin, delta, target_force_21
    );

    // r1 = (0, 0), n1 = (1, 0), l1 = 2
    // r2 = (-0.3, 1.5), n2 = (1, 0), l2 = 1
    //
    // The shortest distance between the two cells is, again, the vector 
    // (0, 1.5), but cell 1 should experience a nonzero torque while cell 2
    // experiences zero torque
    r2(0) = -0.3;
    testForcesKiharaLagrange(
        r1, n1, 0.5, r2, n2, 0.5, R, exp, dmin, delta, target_force_21
    );
}

/**
 * A series of tests for forceKiharaNewton() for parallel cell-cell configurations.
 */
TEST_CASE("Tests for forceKiharaNewton(), parallel cells", "[forceKiharaNewton()]")
{
    const T R = 0.8;
    const T exp = 4; 
    const T dmin = 1.05;

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
    testForceKiharaNewton(r1, n1, 0.5, r2, n2, 0.5, R, exp, dmin);

    // r1 = (0, 0), n1 = (1, 0), l1 = 1
    // r2 = (1 + 1.4 * cos(pi/6), 1.4 * sin(pi/6)), n2 = (1, 0), l2 = 1
    //
    // The shortest distance between the two cells is the vector
    // (1.4 * cos(pi/6), 1.4 * sin(pi/6)), which has norm 1.4
    r2(0) = 1.0 + 1.4 * cos(boost::math::constants::sixth_pi<T>());
    r2(1) = 1.4 * sin(boost::math::constants::sixth_pi<T>());  
    testForceKiharaNewton(r1, n1, 0.5, r2, n2, 0.5, R, exp, dmin);

    // r1 = (0, 0), n1 = (1, 0), l1 = 1
    // r2 = (0, 1.5), n2 = (1, 0), l2 = 1
    //
    // The shortest distance between the two cells is the vector (0, 1.5)
    // 
    // Since the cells are aligned, there should be zero torque
    r2(0) = 0; 
    r2(1) = 1.5; 
    testForceKiharaNewton(r1, n1, 0.5, r2, n2, 0.5, R, exp, dmin);

    // r1 = (0, 0), n1 = (1, 0), l1 = 1
    // r2 = (0.2, 1.5), n2 = (1, 0), l2 = 1
    //
    // The shortest distance between the two cells is, again, the vector
    // (0, 1.5), but the cells are not aligned and there should be nonzero
    // torque
    r2(0) = 0.2; 
    testForceKiharaNewton(r1, n1, 0.5, r2, n2, 0.5, R, exp, dmin);

    // r1 = (0, 0), n1 = (1, 0), l1 = 2
    // r2 = (-0.3, 1.5), n2 = (1, 0), l2 = 1
    //
    // The shortest distance between the two cells is, again, the vector 
    // (0, 1.5), but cell 1 should experience a nonzero torque while cell 2
    // experiences zero torque
    r2(0) = -0.3;
    testForceKiharaNewton(r1, n1, 1, r2, n2, 0.5, R, exp, dmin);
}

/**
 * A series of tests for forcesGBKLagrange() for parallel cell-cell configurations.
 */
TEST_CASE("Tests for forcesGBKLagrange(), parallel cells", "[forcesGBKLagrange()]")
{
    const T R = 0.8;
    const T Rcell = 0.5; 
    const T expd = 4; 
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
    T eps = anisotropyParamGBK1<T, 2>(n1.matrix(), 0.5, n2.matrix(), 0.5, Rcell, 1.0);
    Array<T, 2, 1> target_force_21 = dnorm * eps * expd * (pow(dist, -expd - 1) - pow(2 * R, -expd - 1));
    testForcesGBKLagrange(
        r1, n1, 0.5, r2, n2, 0.5, R, Rcell, expd, dmin, delta, target_force_21
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
    target_force_21 = dnorm * eps * expd * (pow(dist, -expd - 1) - pow(2 * R, -expd - 1));
    testForcesGBKLagrange(
        r1, n1, 0.5, r2, n2, 0.5, R, Rcell, expd, dmin, delta, target_force_21
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
    target_force_21 = dnorm * eps * expd * (pow(dist, -expd - 1) - pow(2 * R, -expd - 1));
    testForcesGBKLagrange(
        r1, n1, 0.5, r2, n2, 0.5, R, Rcell, expd, dmin, delta, target_force_21
    );

    // r1 = (0, 0), n1 = (1, 0), l1 = 1
    // r2 = (0.2, 1.5), n2 = (1, 0), l2 = 1
    //
    // The shortest distance between the two cells is, again, the vector
    // (0, 1.5), but the cells are not aligned and there should be nonzero
    // torque
    r2(0) = 0.2;
    testForcesGBKLagrange(
        r1, n1, 0.5, r2, n2, 0.5, R, Rcell, expd, dmin, delta, target_force_21
    );

    // r1 = (0, 0), n1 = (1, 0), l1 = 2
    // r2 = (-0.3, 1.5), n2 = (1, 0), l2 = 1
    //
    // The shortest distance between the two cells is, again, the vector 
    // (0, 1.5), but cell 1 should experience a nonzero torque while cell 2
    // experiences zero torque
    r2(0) = -0.3;
    testForcesGBKLagrange(
        r1, n1, 0.5, r2, n2, 0.5, R, Rcell, expd, dmin, delta, target_force_21
    );
}

/**
 * A series of tests for forceGBKNewton() for parallel cell-cell configurations.
 */
TEST_CASE("Tests for forceGBKNewton(), parallel cells", "[forceGBKNewton()]")
{
    const T R = 0.8;
    const T Rcell = 0.5;
    const T expd = 4; 
    const T dmin = 1.05;

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
    testForceGBKNewton(r1, n1, 0.5, r2, n2, 0.5, R, Rcell, expd, dmin);

    // r1 = (0, 0), n1 = (1, 0), l1 = 1
    // r2 = (1 + 1.4 * cos(pi/6), 1.4 * sin(pi/6)), n2 = (1, 0), l2 = 1
    //
    // The shortest distance between the two cells is the vector
    // (1.4 * cos(pi/6), 1.4 * sin(pi/6)), which has norm 1.4
    r2(0) = 1.0 + 1.4 * cos(boost::math::constants::sixth_pi<T>());
    r2(1) = 1.4 * sin(boost::math::constants::sixth_pi<T>());  
    testForceGBKNewton(r1, n1, 0.5, r2, n2, 0.5, R, Rcell, expd, dmin);

    // r1 = (0, 0), n1 = (1, 0), l1 = 1
    // r2 = (0, 1.5), n2 = (1, 0), l2 = 1
    //
    // The shortest distance between the two cells is the vector (0, 1.5)
    // 
    // Since the cells are aligned, there should be zero torque
    r2(0) = 0; 
    r2(1) = 1.5; 
    testForceGBKNewton(r1, n1, 0.5, r2, n2, 0.5, R, Rcell, expd, dmin);

    // r1 = (0, 0), n1 = (1, 0), l1 = 1
    // r2 = (0.2, 1.5), n2 = (1, 0), l2 = 1
    //
    // The shortest distance between the two cells is, again, the vector
    // (0, 1.5), but the cells are not aligned and there should be nonzero
    // torque
    r2(0) = 0.2; 
    testForceGBKNewton(r1, n1, 0.5, r2, n2, 0.5, R, Rcell, expd, dmin);

    // r1 = (0, 0), n1 = (1, 0), l1 = 2
    // r2 = (-0.3, 1.5), n2 = (1, 0), l2 = 1
    //
    // The shortest distance between the two cells is, again, the vector 
    // (0, 1.5), but cell 1 should experience a nonzero torque while cell 2
    // experiences zero torque
    r2(0) = -0.3;
    testForceGBKNewton(r1, n1, 1, r2, n2, 0.5, R, Rcell, expd, dmin);
}

/* ------------------------------------------------------------------ //
 *      TEST MODULES FOR PERPENDICULAR CELL-CELL CONFIGURATIONS       //
 * ------------------------------------------------------------------ */
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
        r1, n1, 0.5, r2, n2, 0.5, R, exp, dmin, delta, target_force_21
    );

    // r1 = (0, 0), n1 = (1, 0), l1 = 1
    // r2 = (1.5, 0.2), n2 = (0, 1), l2 = 1
    //
    // The shortest distance between the two cells is, again, the vector
    // (1, 0), but cell 1 should experience zero torque while cell 2 experiences
    // a nonzero torque 
    r2(1) = 0.2;  
    testForcesKiharaLagrange(
        r1, n1, 0.5, r2, n2, 0.5, R, exp, dmin, delta, target_force_21
    );
}

/**
 * A series of tests for forceKiharaNewton() for perpendicular cell-cell
 * configurations. 
 */
TEST_CASE("Tests for forceKiharaNewton(), perpendicular cells", "[forceKiharaNewton()]") 
{
    const T R = 0.8;
    const T exp = 4; 
    const T dmin = 1.05;

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
    testForceKiharaNewton(r1, n1, 0.5, r2, n2, 0.5, R, exp, dmin);

    // r1 = (0, 0), n1 = (1, 0), l1 = 1
    // r2 = (1.5, 0.2), n2 = (0, 1), l2 = 1
    //
    // The shortest distance between the two cells is, again, the vector
    // (1, 0), but cell 1 should experience zero torque while cell 2 experiences
    // a nonzero torque 
    r2(1) = 0.2;  
    testForceKiharaNewton(r1, n1, 0.5, r2, n2, 0.5, R, exp, dmin);
}

/**
 * A series of tests for forcesGBKLagrange() for perpendicular cell-cell
 * configurations.
 */
TEST_CASE("Tests for forcesGBKLagrange(), perpendicular cells", "[forcesGBKLagrange()]")
{
    const T R = 0.8;
    const T Rcell = 0.5; 
    const T expd = 4; 
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
    T eps = anisotropyParamGBK1<T, 2>(n1.matrix(), 0.5, n2.matrix(), 0.5, Rcell, 1.0);
    Array<T, 2, 1> target_force_21 = d12 * eps * expd * (pow(dmin, -expd - 1) - pow(2 * R, -expd - 1));
    testForcesGBKLagrange(
        r1, n1, 0.5, r2, n2, 0.5, R, Rcell, expd, dmin, delta, target_force_21
    );

    // r1 = (0, 0), n1 = (1, 0), l1 = 1
    // r2 = (1.5, 0.2), n2 = (0, 1), l2 = 1
    //
    // The shortest distance between the two cells is, again, the vector
    // (1, 0), but cell 1 should experience zero torque while cell 2 experiences
    // a nonzero torque 
    r2(1) = 0.2;  
    testForcesGBKLagrange(
        r1, n1, 0.5, r2, n2, 0.5, R, Rcell, expd, dmin, delta, target_force_21
    );
}

/**
 * A series of tests for forceGBKNewton() for perpendicular cell-cell
 * configurations. 
 */
TEST_CASE("Tests for forceGBKNewton(), perpendicular cells", "[forceGBKNewton()]") 
{
    const T R = 0.8;
    const T Rcell = 0.5; 
    const T expd = 4; 
    const T dmin = 1.05;

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
    testForceGBKNewton(r1, n1, 0.5, r2, n2, 0.5, R, Rcell, expd, dmin);

    // r1 = (0, 0), n1 = (1, 0), l1 = 1
    // r2 = (1.5, 0.2), n2 = (0, 1), l2 = 1
    //
    // The shortest distance between the two cells is, again, the vector
    // (1, 0), but cell 1 should experience zero torque while cell 2 experiences
    // a nonzero torque 
    r2(1) = 0.2;  
    testForceGBKNewton(r1, n1, 0.5, r2, n2, 0.5, R, Rcell, expd, dmin);
}


