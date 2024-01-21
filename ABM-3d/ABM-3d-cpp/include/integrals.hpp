/**
 * Functions for computing a series of auxiliary integrals involved in 
 * the 3-D biofilm simulations. 
 *
 * Authors:
 *     Kee-Myoung Nam
 *
 * Last updated:
 *     1/20/2023
 */

#ifndef BIOFILM_3D_AUXILIARY_INTEGRALS_HPP
#define BIOFILM_3D_AUXILIARY_INTEGRALS_HPP

#include <cmath>
#include <Eigen/Dense>
#include <boost/math/constants/constants.hpp>

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
    T p = phi(rz, nz, R, s);
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
    T p = phi(rz, nz, R, s); 
    if (p > 0)
        return std::pow(p, gamma);
    else
        return 0;
}

/**
 * Compute the integral of \delta_i^\gamma(s), where \gamma is a real exponent,
 * from s = -l_i/2 to s = +l_i/2, where l_i is the length of the cell.
 *
 * The z-coordinate of the cell's orientation vector is assumed to be negative.
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
    if (ss < -half_l)
    {
        T overlap1 = std::pow(phi(rz, nz, R, -half_l), gamma + 1);
        T overlap2 = std::pow(phi(rz, nz, R, half_l), gamma + 1);
        return -(overlap2 - overlap1) / (nz * (gamma + 1));
    }
    else if (ss < half_l)
    {
        // overlap1 = 0
        T overlap2 = std::pow(phi(rz, nz, R, half_l), gamma + 1);
        return -overlap2 / (nz * (gamma + 1));
    }
    else 
    {
        return 0;
    }
}

/**
 * Compute the integral of s * \delta_i^\gamma(s), where \gamma is a real
 * exponent, from s = -l_i/2 to s = +l_i/2, where l_i is the length of the
 * cell.
 *
 * The z-coordinate of the cell's orientation vector is assumed to be negative.
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
    T term1 = integral1(rz, nz, R, half_l, gamma + 1, ss);
    T term2 = 0; 
    if (ss < -half_l)
    {
        T overlap1 = std::pow(phi(rz, nz, R, -half_l), gamma + 1);
        T overlap2 = std::pow(phi(rz, nz, R, half_l), gamma + 1);
        term2 = half_l * overlap2 + half_l * overlap1;    // = ... - (-half_l) * overlap1
    }
    else if (ss < half_l)
    {
        // Overlap at s = ss is zero 
        T overlap2 = std::pow(phi(rz, nz, R, half_l), gamma + 1);
        term2 = half_l * overlap2;
    }
    return (term1 - term2) / (nz * (gamma + 1)); 
}

/**
 * Compute the integral of s^2 * \delta_i^\gamma(s), where \gamma is a real
 * exponent, from s = -l_i/2 to s = +l_i/2, where l_i is the length of the
 * cell.
 *
 * The z-coordinate of the cell's orientation vector is assumed to be negative.
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
    T term1 = 2 * integral2(rz, nz, R, half_l, gamma + 1, ss);
    T term2 = 0;
    if (ss < -half_l)
    {
        T overlap1 = std::pow(phi(rz, nz, R, -half_l), gamma + 1);
        T overlap2 = std::pow(phi(rz, nz, R, half_l), gamma + 1);
        term2 = half_l * half_l * (overlap2 - overlap1);
    }
    else if (ss < half_l)
    {
        // Overlap at s = ss is zero 
        T overlap2 = std::pow(phi(rz, nz, R, half_l), gamma + 1);
        term2 = half_l * half_l * overlap2;
    }
    return (term1 - term2) / (nz * (gamma + 1));
}

/**
 * Compute the integral of \Theta(\delta_i(s)), where \Theta is the Heaviside
 * step function, from s = -l_i/2 to s = +l_i/2, where l_i is the length of
 * the cell.
 *
 * The z-coordinate of the cell's orientation vector is assumed to be negative.
 *
 * TODO Fix arguments here
 *
 * @param rz z-coordinate of cell center.
 * @param nz z-coordinate of cell orientation.
 * @param R Cell radius.
 * @param half_l Half of cell length.
 * @param ss Pre-computed value of `sstar(rz, nz, R)`.
 * @returns Desired integral.  
 */
template <typename T>
T integral4(const T rz, const T nz, const T R, const T half_l, const T ss)
{
    if (ss < -half_l)
        return 2 * half_l;
    else if (ss < half_l)
        return half_l - ss;
    else
        return 0;
}

/**
 * Compute the integral of s * \Theta(\delta_i(s)), where \Theta is the
 * Heaviside step function, from s = -l_i/2 to s = +l_i/2, where l_i is the
 * length of the cell.
 *
 * The z-coordinate of the cell's orientation vector is assumed to be negative.
 *
 * @param rz z-coordinate of cell center.
 * @param nz z-coordinate of cell orientation.
 * @param R Cell radius.
 * @param half_l Half of cell length.
 * @param ss Pre-computed value of `sstar(rz, nz, R)`.
 * @returns Desired integral.  
 */
template <typename T>
T integral5(const T rz, const T nz, const T R, const T half_l, const T ss)
{
    T term1 = integral1(rz, nz, R, half_l, 1.0, ss);
    T term2 = 0;
    if (ss < -half_l)
    {
        T overlap1 = phi(rz, nz, R, -half_l);
        T overlap2 = phi(rz, nz, R, half_l);
        term2 = half_l * overlap2 + half_l * overlap1;   // = ... - (-half_l) * overlap1
    }
    else if (ss < half_l)
    {
        // Overlap at s = ss is zero 
        T overlap2 = phi(rz, nz, R, half_l);
        term2 = half_l * overlap2;
    }
    return (term1 - term2) / nz;
}

/**
 * Compute the integral of s^2 * \Theta(\delta_i(s)), where \Theta is the
 * Heaviside step function, from s = -l_i/2 to s = +l_i/2, where l_i is the
 * length of the cell.
 *
 * The z-coordinate of the cell's orientation vector is assumed to be negative.
 *
 * @param rz z-coordinate of cell center.
 * @param nz z-coordinate of cell orientation.
 * @param R Cell radius.
 * @param half_l Half of cell length.
 * @param ss Pre-computed value of `sstar(rz, nz, R)`.
 * @returns Desired integral.  
 */
template <typename T>
T integral6(const T rz, const T nz, const T R, const T half_l, const T ss)
{
    T term1 = 2 * integral2(rz, nz, R, half_l, 1.0, ss);
    T term2 = 0;
    if (ss < -half_l)
    {
        T overlap1 = phi(rz, nz, R, -half_l);
        T overlap2 = phi(rz, nz, R, half_l);
        term2 = half_l * half_l * (overlap2 - overlap1);
    }
    else if (ss < half_l)
    {
        // Overlap at s = ss is zero 
        T overlap2 = phi(rz, nz, R, half_l);
        term2 = half_l * half_l * overlap2;
    }
    return (term1 - term2) / nz;
}

/**
 * Compute the integral of the cell-surface contact area density a_i(s) from
 * s = -l_i/2 to s = +l_i/2, where l_i is the length of the cell.
 *
 * The z-coordinate of the cell's orientation vector is assumed to be negative.
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
    T term1 = std::pow(R, 0.5) * (1 - nz * nz) * integral1(rz, nz, R, half_l, 0.5, ss);
    T term2 = boost::math::constants::pi<T>() * R * nz * nz * integral4(rz, nz, R, half_l, ss);
    return term1 + term2;
}

/**
 * Compute the integral of s * a_i(s), where a_i(s) is the cell-surface
 * contact area density, from s = -l_i/2 to s = +l_i/2, where l_i is the 
 * length of the cell.
 *
 * The z-coordinate of the cell's orientation vector is assumed to be negative.
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
    T term1 = std::pow(R, 0.5) * (1 - nz * nz) * integral2(rz, nz, R, half_l, 0.5, ss);
    T term2 = boost::math::constants::pi<T>() * R * nz * nz * integral5(rz, nz, R, half_l, ss);
    return term1 + term2;
}

/**
 * Compute the integral of s^2 * a_i(s), where a_i(s) is the cell-surface
 * contact area density, from s = -l_i/2 to s = +l_i/2, where l_i is the 
 * length of the cell.
 *
 * The z-coordinate of the cell's orientation vector is assumed to be negative.
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
    T term1 = std::pow(R, 0.5) * (1 - nz * nz) * integral3(rz, nz, R, half_l, 0.5, ss);
    T term2 = boost::math::constants::pi<T>() * R * nz * nz * integral6(rz, nz, R, half_l, ss);
    return term1 + term2;
}

#endif
