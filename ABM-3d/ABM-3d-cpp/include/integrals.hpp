/**
 * Functions for computing a series of auxiliary integrals involved in 
 * the 3-D biofilm simulations. 
 *
 * Authors:
 *     Kee-Myoung Nam
 *
 * Last updated:
 *     4/16/2025
 */

#ifndef BIOFILM_3D_AUXILIARY_INTEGRALS_HPP
#define BIOFILM_3D_AUXILIARY_INTEGRALS_HPP

#include <cmath>
#include <boost/math/constants/constants.hpp>
#include <boost/multiprecision/mpfr.hpp>

// Expose math functions for both standard and boost MPFR types
using std::sqrt; 
using boost::multiprecision::sqrt; 
using std::pow;
using boost::multiprecision::pow;

/**
 * Return the cell-body coordinate at which the cell-surface overlap is zero.
 *
 * The z-coordinate of the cell's orientation vector is assumed to be nonzero.
 *
 * @param rz z-coordinate of cell center.
 * @param nz z-coordinate of cell orientation.
 * @param R Cell radius.
 * @returns Cell-body coordinate at which the cell-surface overlap is zero.
 */
template <typename T>
T sstar(const T rz, const T nz, const T R)
{
    #ifdef CHECK_CELL_ORIENTATION_ZCOORD_NONZERO
        if (nz == 0)
            throw std::invalid_argument(
                "Cell z-orientation cannot be nonzero when calculating `sstar()`"
            ); 
    #endif
    return (R - rz) / nz;
}

/**
 * Return the cell-surface overlap, \phi_i(s).
 *
 * @param rz z-coordinate of cell center.
 * @param nz z-coordinate of cell orientation.
 * @param R Cell radius.
 * @param s Cell-body coordinate. 
 * @returns Cell-surface overlap. 
 */
template <typename T>
T phi(const T rz, const T nz, const T R, const T s)
{
    return R - rz - s * nz;
}

/**
 * Return the cell-surface overlap, \delta_i(s).
 *
 * @param rz z-coordinate of cell center.
 * @param nz z-coordinate of cell orientation.
 * @param R Cell radius.
 * @param s Cell-body coordinate. 
 * @returns Cell-surface overlap. 
 */
template <typename T>
T overlap(const T rz, const T nz, const T R, const T s)
{
    T p = phi<T>(rz, nz, R, s);
    if (p > 0)
        return p;
    else
        return 0;
}

/**
 * Return the exponentiated cell-surface overlap, \delta_i^\gamma(s), where 
 * \gamma is a real exponent.
 *
 * @param rz z-coordinate of cell center.
 * @param nz z-coordinate of cell orientation.
 * @param R Cell radius.
 * @param s Cell-body coordinate. 
 * @param gamma Exponent.
 * @returns Exponentiated cell-surface overlap. 
 */
template <typename T>
T overlapGamma(const T rz, const T nz, const T R, const T s, const T gamma)
{
    T p = phi<T>(rz, nz, R, s); 
    if (p > 0)
        return pow(p, gamma);
    else
        return 0;
}

/**
 * Compute the integral of \delta_i^\gamma(s), where \gamma is a real exponent,
 * from s = -l_i/2 to s = +l_i/2, where l_i is the length of the cell.
 *
 * The z-coordinate of the cell's orientation vector is assumed to be positive.
 *
 * @param rz z-coordinate of cell center.
 * @param nz z-coordinate of cell orientation.
 * @param R Cell radius.
 * @param half_l Half of cell length.
 * @param gamma Exponent; should not be -1.
 * @param ss Pre-computed value of `sstar(rz, nz, R)`.
 * @returns Desired integral.  
 */
template <typename T>
T integral1(const T rz, const T nz, const T R, const T half_l, const T gamma,
            const T ss)
{
    #ifdef CHECK_CELL_ORIENTATION_ZCOORD_POSITIVE
        if (nz <= 0)
            throw std::invalid_argument(
                "Cell z-orientation must be positive when calculating auxiliary integral"
            ); 
    #endif

    if (ss > half_l)
    {
        T overlap1 = pow(phi<T>(rz, nz, R, -half_l), gamma + 1);
        T overlap2 = pow(phi<T>(rz, nz, R, half_l), gamma + 1);
        return (overlap1 - overlap2) / (nz * (gamma + 1));
    }
    else if (ss > -half_l)   // -half_l < ss <= half_l
    {
        // overlap2 = 0
        T overlap1 = pow(phi<T>(rz, nz, R, -half_l), gamma + 1);
        return overlap1 / (nz * (gamma + 1));
    }
    else                     // ss <= -half_l
    {
        return 0;
    }
}

/**
 * Compute the integral of s * \delta_i^\gamma(s), where \gamma is a real
 * exponent, from s = -l_i/2 to s = +l_i/2, where l_i is the length of the
 * cell.
 *
 * The z-coordinate of the cell's orientation vector is assumed to be positive.
 *
 * @param rz z-coordinate of cell center.
 * @param nz z-coordinate of cell orientation.
 * @param R Cell radius. 
 * @param half_l Half of cell length.
 * @param gamma Exponent; should not be -1 or -2.
 * @param ss Pre-computed value of `sstar(rz, nz, R)`.
 * @returns Desired integral.  
 */
template <typename T>
T integral2(const T rz, const T nz, const T R, const T half_l, const T gamma,
            const T ss)
{
    #ifdef CHECK_CELL_ORIENTATION_ZCOORD_POSITIVE
        if (nz <= 0)
            throw std::invalid_argument(
                "Cell z-orientation must be positive when calculating auxiliary integral"
            ); 
    #endif

    T term1 = integral1<T>(rz, nz, R, half_l, gamma + 1, ss);
    T term2 = 0; 
    if (ss > half_l)
    {
        T overlap1 = pow(phi<T>(rz, nz, R, -half_l), gamma + 1);
        T overlap2 = pow(phi<T>(rz, nz, R, half_l), gamma + 1);
        term2 = -half_l * (overlap1 + overlap2); 
    }
    else if (ss > -half_l)    // -half_l < ss <= half_l
    {
        // Overlap at s = ss is zero 
        T overlap1 = pow(phi<T>(rz, nz, R, -half_l), gamma + 1);
        term2 = -half_l * overlap1;
    }
    return (term1 + term2) / (nz * (gamma + 1)); 
}

/**
 * Compute the integral of s^2 * \delta_i^\gamma(s), where \gamma is a real
 * exponent, from s = -l_i/2 to s = +l_i/2, where l_i is the length of the
 * cell.
 *
 * The z-coordinate of the cell's orientation vector is assumed to be positive.
 *
 * @param rz z-coordinate of cell center.
 * @param nz z-coordinate of cell orientation.
 * @param R Cell radius. 
 * @param half_l Half of cell length.
 * @param gamma Exponent; should not be -1 or -2.
 * @param ss Pre-computed value of `sstar(rz, nz, R)`.
 * @returns Desired integral.  
 */
template <typename T>
T integral3(const T rz, const T nz, const T R, const T half_l, const T gamma,
            const T ss)
{
    #ifdef CHECK_CELL_ORIENTATION_ZCOORD_POSITIVE
        if (nz <= 0)
            throw std::invalid_argument(
                "Cell z-orientation must be positive when calculating auxiliary integral"
            ); 
    #endif

    T term1 = 2 * integral2<T>(rz, nz, R, half_l, gamma + 1, ss);
    T term2 = 0;
    if (ss > half_l)
    {
        T overlap1 = pow(phi<T>(rz, nz, R, -half_l), gamma + 1);
        T overlap2 = pow(phi<T>(rz, nz, R, half_l), gamma + 1);
        term2 = half_l * half_l * (overlap1 - overlap2);
    }
    else if (ss > -half_l)    // -half_l < ss <= half_l
    {
        // Overlap at s = ss is zero 
        T overlap1 = pow(phi<T>(rz, nz, R, -half_l), gamma + 1);
        term2 = half_l * half_l * overlap1;
    }
    return (term1 + term2) / (nz * (gamma + 1));
}

/**
 * Compute the integral of \Theta(\delta_i(s)), where \Theta is the Heaviside
 * step function, from s = -l_i/2 to s = +l_i/2, where l_i is the length of
 * the cell.
 *
 * The z-coordinate of the cell's orientation vector is assumed to be positive.
 *
 * @param nz z-coordinate of cell orientation.
 * @param half_l Half of cell length.
 * @param ss Pre-computed value of `sstar(rz, nz, R)`, whose input values are
 *           defined elsewhere and not passed into this function.
 * @returns Desired integral.  
 */
template <typename T>
T integral4(const T nz, const T half_l, const T ss)
{
    #ifdef CHECK_CELL_ORIENTATION_ZCOORD_POSITIVE
        if (nz <= 0)
            throw std::invalid_argument(
                "Cell z-orientation must be positive when calculating auxiliary integral"
            ); 
    #endif

    if (ss > half_l)
        return 2 * half_l;    // = half_l - (-half_l)
    else if (ss > -half_l)    // -half_l < ss <= half_l
        return ss + half_l;   // = ss - (-half_l)
    else
        return 0;
}

/**
 * Compute the integral of s * \Theta(\delta_i(s)), where \Theta is the
 * Heaviside step function, from s = -l_i/2 to s = +l_i/2, where l_i is the
 * length of the cell.
 *
 * The z-coordinate of the cell's orientation vector is assumed to be positive.
 *
 * @param nz z-coordinate of cell orientation.
 * @param half_l Half of cell length.
 * @param ss Pre-computed value of `sstar(rz, nz, R)`.
 * @returns Desired integral.  
 */
template <typename T>
T integral5(const T nz, const T half_l, const T ss)
{
    #ifdef CHECK_CELL_ORIENTATION_ZCOORD_POSITIVE
        if (nz <= 0)
            throw std::invalid_argument(
                "Cell z-orientation must be positive when calculating auxiliary integral"
            ); 
    #endif

    if (-half_l < ss && ss <= half_l)
        return (ss * ss - half_l * half_l) / 2.0;
    else 
        return 0.0; 
}

/**
 * Compute the integral of s^2 * \Theta(\delta_i(s)), where \Theta is the
 * Heaviside step function, from s = -l_i/2 to s = +l_i/2, where l_i is the
 * length of the cell.
 *
 * The z-coordinate of the cell's orientation vector is assumed to be positive.
 *
 * @param nz z-coordinate of cell orientation.
 * @param half_l Half of cell length.
 * @param ss Pre-computed value of `sstar(rz, nz, R)`.
 * @returns Desired integral.  
 */
template <typename T>
T integral6(const T nz, const T half_l, const T ss)
{
    #ifdef CHECK_CELL_ORIENTATION_ZCOORD_POSITIVE
        if (nz <= 0)
            throw std::invalid_argument(
                "Cell z-orientation must be positive when calculating auxiliary integral"
            ); 
    #endif

    if (ss > half_l)
        return 2 * half_l * half_l * half_l / 3.0;
    else if (ss > -half_l)    // -half_l < ss <= half_l
        return (ss * ss * ss + half_l * half_l * half_l) / 3.0; 
    else
        return 0.0;
}

/**
 * Compute the integral of the cell-surface contact area density a_i(s) from
 * s = -l_i/2 to s = +l_i/2, where l_i is the length of the cell.
 *
 * The z-coordinate of the cell's orientation vector is assumed to be positive.
 *
 * @param rz z-coordinate of cell center.
 * @param nz z-coordinate of cell orientation.
 * @param R Cell radius.
 * @param half_l Half of cell length.
 * @param ss Pre-computed value of `sstar(rz, nz, R)`.
 * @returns Desired integral.
 */
template <typename T>
T areaIntegral1(const T rz, const T nz, const T R, const T half_l, const T ss)
{
    T term1 = sqrt(R) * (1 - nz * nz) * integral1<T>(rz, nz, R, half_l, 0.5, ss);
    T term2 = boost::math::constants::pi<T>() * R * nz * nz * integral4<T>(nz, half_l, ss);
    return term1 + term2;
}

/**
 * Compute the integral of s * a_i(s), where a_i(s) is the cell-surface
 * contact area density, from s = -l_i/2 to s = +l_i/2, where l_i is the 
 * length of the cell.
 *
 * The z-coordinate of the cell's orientation vector is assumed to be positive.
 *
 * @param rz z-coordinate of cell center.
 * @param nz z-coordinate of cell orientation.
 * @param R Cell radius.
 * @param half_l Half of cell length.
 * @param ss Pre-computed value of `sstar(rz, nz, R)`.
 * @returns Desired integral.
 */
template <typename T>
T areaIntegral2(const T rz, const T nz, const T R, const T half_l, const T ss)
{
    T term1 = sqrt(R) * (1 - nz * nz) * integral2<T>(rz, nz, R, half_l, 0.5, ss);
    T term2 = boost::math::constants::pi<T>() * R * nz * nz * integral5<T>(nz, half_l, ss);
    return term1 + term2;
}

/**
 * Compute the integral of s^2 * a_i(s), where a_i(s) is the cell-surface
 * contact area density, from s = -l_i/2 to s = +l_i/2, where l_i is the 
 * length of the cell.
 *
 * The z-coordinate of the cell's orientation vector is assumed to be positive.
 *
 * @param rz z-coordinate of cell center.
 * @param nz z-coordinate of cell orientation.
 * @param R Cell radius.
 * @param half_l Half of cell length.
 * @param ss Pre-computed value of `sstar(rz, nz, R)`.
 * @returns Desired integral.
 */
template <typename T>
T areaIntegral3(const T rz, const T nz, const T R, const T half_l, const T ss)
{
    T term1 = sqrt(R) * (1 - nz * nz) * integral3<T>(rz, nz, R, half_l, 0.5, ss);
    T term2 = boost::math::constants::pi<T>() * R * nz * nz * integral6<T>(nz, half_l, ss);
    return term1 + term2;
}

/**
 * Compute all three area integrals in one fell swoop. 
 *
 * This function minimizes separate function calls for calculating the six
 * auxiliary integrals. 
 *
 * The z-coordinate of the cell's orientation vector is assumed to be positive.
 *
 * @param rz z-coordinate of cell center.
 * @param nz z-coordinate of cell orientation.
 * @param R Cell radius.
 * @param half_l Half of cell length.
 * @param ss Pre-computed value of `sstar(rz, nz, R)`.
 * @returns Desired integral.
 */
template <typename T>
std::tuple<T, T, T> areaIntegrals(const T rz, const T nz, const T R,
                                  const T half_l, const T ss)
{
    // Calculate integrals 4, 5, and 6
    T int4 = integral4<T>(nz, half_l, ss); 
    T int5 = integral5<T>(nz, half_l, ss); 
    T int6 = integral6<T>(nz, half_l, ss); 

    // Calculate integral 1 for exponents 0.5, 1.5, and 2.5
    T int1a = integral1<T>(rz, nz, R, half_l, 0.5, ss); 
    T int1b = integral1<T>(rz, nz, R, half_l, 1.5, ss); 
    T int1c = integral1<T>(rz, nz, R, half_l, 2.5, ss);

    // Then area integral 1 is obtained from integrals 1a and 4
    T term1 = sqrt(R) * (1 - nz * nz) * int1a; 
    T term2 = boost::math::constants::pi<T>() * R * nz * nz * int4;
    T area1 = term1 + term2;

    // Get overlaps at the two endpoints, accounting for partial contacts
    T overlap1 = 0;
    T overlap2 = 0;
    if (ss > half_l)
    {
        overlap1 = phi<T>(rz, nz, R, -half_l);
        overlap2 = phi<T>(rz, nz, R, half_l); 
    }
    else if (ss > -half_l)   // In which case -half_l < ss <= half_l
    {
        overlap1 = phi<T>(rz, nz, R, -half_l); 
    }

    // To get area integral 2, we need integral 2 with exponent 0.5, which
    // in turn requires integral 1b
    T overlap1_pow1 = pow(overlap1, 1.5); 
    T overlap2_pow1 = pow(overlap2, 1.5); 
    term2 = -half_l * (overlap1_pow1 + overlap2_pow1); 
    T int2a = (int1b + term2) / (1.5 * nz);
    term1 = sqrt(R) * (1 - nz * nz) * int2a; 
    term2 = boost::math::constants::pi<T>() * R * nz * nz * int5;
    T area2 = term1 + term2;

    // To get area integral 3, we need integral 3 with exponent 0.5, which
    // in turn requires integral 2 with exponent 1.5, which in turn requires
    // integral 1c
    //
    // First calculate integral 2 with exponent 1.5 ...
    T overlap1_pow2 = overlap1_pow1 * overlap1; 
    T overlap2_pow2 = overlap2_pow1 * overlap2;  
    term2 = -half_l * (overlap1_pow2 + overlap2_pow2); 
    T int2b = (int1c + term2) / (2.5 * nz);

    // ... then calculate integral 3 with exponent 0.5 ... 
    term1 = 2 * int2b;
    term2 = half_l * half_l * (overlap1_pow1 - overlap2_pow1);  
    T int3 = (term1 + term2) / (1.5 * nz); 

    // ... then finally calculate area integral 3 
    term1 = sqrt(R) * (1 - nz * nz) * int3; 
    term2 = boost::math::constants::pi<T>() * R * nz * nz * int6;
    T area3 = term1 + term2;

    return std::make_tuple(area1, area2, area3); 
}

#endif
