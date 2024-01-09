/**
 * Functions for computing a series of auxiliary integrals involved in 
 * the 3-D biofilm simulations. 
 *
 * Authors:
 *     Kee-Myoung Nam
 *
 * Last updated:
 *     1/8/2023
 */

#ifndef BIOFILM_3D_AUXILIARY_INTEGRALS_HPP
#define BIOFILM_3D_AUXILIARY_INTEGRALS_HPP

#include <cmath>
#include <Eigen/Dense>
#include <boost/math/constants/constants.hpp>

/**
 * Compute the integral of a cell-surface overlap, \delta_i(s), raised by
 * an exponent, \gamma, from s = -l_i/2 to s = +l_i/2, where l_i is the 
 * length of the cell.
 *
 * The z-coordinate of the cell's orientation vector is assumed to be nonzero.
 *
 * @param rz z-coordinate of cell center.
 * @param nz z-coordinate of cell orientation.
 * @param half_l Half of cell length.
 * @param R Cell radius. 
 * @param gamma Exponent; should not be -1.
 * @returns Desired integral.  
 */
template <typename T>
T integral1(T rz, T nz, T half_l, T R, T gamma)
{
    T overlap1 = std::pow(R - rz + half_l * nz, gamma + 1); 
    T overlap2 = std::pow(R - rz - half_l * nz, gamma + 1);
    return -(overlap2 - overlap1) / ((gamma + 1) * nz); 
}

/**
 * Compute the integral of s * \delta_i^\gamma(s), where \delta_i(s) is 
 * the cell-surface overlap and \gamma is a real exponent, from s = -l_i/2
 * to s = +l_i/2, where l_i is the length of the cell.
 *
 * The z-coordinate of the cell's orientation vector is assumed to be nonzero.
 *
 * @param rz z-coordinate of cell center.
 * @param nz z-coordinate of cell orientation.
 * @param half_l Half of cell length.
 * @param R Cell radius. 
 * @param gamma Exponent; should not be -1 or -2.
 * @returns Desired integral.  
 */
template <typename T>
T integral2(T rz, T nz, T half_l, T R, T gamma)
{
    T overlap1 = std::pow(R - rz + half_l * nz, gamma + 1); 
    T overlap2 = std::pow(R - rz - half_l * nz, gamma + 1);
    T term1 = -half_l * (overlap2 + overlap1);
    T term2 = integral1(rz, nz, half_l, R, gamma + 1);
    return (term1 + term2) / ((gamma + 1) * nz); 
}

/**
 * Compute the integral of s^2 * \delta_i^\gamma(s), where \delta_i(s) is 
 * the cell-surface overlap and \gamma is a real exponent, from s = -l_i/2
 * to s = +l_i/2, where l_i is the length of the cell.
 *
 * The z-coordinate of the cell's orientation vector is assumed to be nonzero.
 *
 * @param rz z-coordinate of cell center.
 * @param nz z-coordinate of cell orientation.
 * @param half_l Half of cell length.
 * @param R Cell radius. 
 * @param gamma Exponent; should not be -1 or -2 or -3.
 * @returns Desired integral.  
 */
template <typename T>
T integral3(T rz, T nz, T half_l, T R, T gamma)
{
    T overlap1 = std::pow(R - rz + half_l * nz, gamma + 1); 
    T overlap2 = std::pow(R - rz - half_l * nz, gamma + 1);
    T term1 = -half_l * half_l * (overlap2 - overlap1);
    T term2 = 2 * integral2(rz, nz, half_l, R, gamma + 1);
    return (term1 + term2) / ((gamma + 1) * nz); 
}

/**
 * The Heaviside step function.
 *
 * @param s Input value.
 * @returns The Heaviside step function evaluated at s, i.e., +1 if s >= 0,
 *          0 otherwise. 
 */
template <typename T>
T step(T s)
{
    if (s >= 0)
        return 1;
    else
        return 0;
}

/**
 * Compute the integral of \Theta(\delta_i(s)), where \Theta is the Heaviside
 * step function and \delta_i(s) is the cell-surface overlap, from s = -l_i/2
 * to s = +l_i/2, where l_i is the length of the cell.
 *
 * The z-coordinate of the cell's orientation vector is assumed to be nonzero.
 *
 * @param rz z-coordinate of cell center.
 * @param nz z-coordinate of cell orientation.
 * @param half_l Half of cell length.
 * @param R Cell radius.
 * @returns Desired integral.  
 */
template <typename T>
T integral4(T rz, T nz, T half_l, T R)
{
    T overlap1 = R - rz + half_l * nz; 
    T overlap2 = R - rz - half_l * nz;
    T step1 = step(overlap1);
    T step2 = step(overlap2);
    return -(step2 * overlap2 - step1 * overlap1) / nz;
}

/**
 * Compute the integral of s * \Theta(\delta_i(s)), where \Theta is the
 * Heaviside step function and \delta_i(s) is the cell-surface overlap, 
 * from s = -l_i/2 to s = +l_i/2, where l_i is the length of the cell.
 *
 * The z-coordinate of the cell's orientation vector is assumed to be nonzero.
 *
 * @param rz z-coordinate of cell center.
 * @param nz z-coordinate of cell orientation.
 * @param half_l Half of cell length.
 * @param R Cell radius.
 * @returns Desired integral.  
 */
template <typename T>
T integral5(T rz, T nz, T half_l, T R)
{
    T overlap1 = R - rz + half_l * nz;
    T overlap2 = R - rz - half_l * nz;
    T step1 = step(overlap1);
    T step2 = step(overlap2);
    T term1 = -half_l * (step2 * overlap2 + step1 * overlap1) / nz;
    T term2 = (-step2 * overlap2 * overlap2 + step1 * overlap1 * overlap1) / (2 * nz * nz);
    return term1 + term2; 
}

/**
 * Compute the integral of s * \delta_i(s) * \Theta(\delta_i(s)), where
 * \Theta is the Heaviside step function and \delta_i(s) is the cell-surface
 * overlap, from s = -l_i/2 to s = +l_i/2, where l_i is the length of the cell.
 *
 * The z-coordinate of the cell's orientation vector is assumed to be nonzero.
 *
 * @param rz z-coordinate of cell center.
 * @param nz z-coordinate of cell orientation.
 * @param half_l Half of cell length.
 * @param R Cell radius.
 * @returns Desired integral.  
 */
template <typename T>
T integral6(T rz, T nz, T half_l, T R)
{
    T overlap1 = R - rz + half_l * nz;
    T overlap2 = R - rz - half_l * nz;
    T step1 = step(overlap1);
    T step2 = step(overlap2);
    T term1 = -half_l * (step2 * overlap2 * overlap2 + step1 * overlap1 * overlap1) / (2 * nz);
    T term2 = (-step2 * overlap2 * overlap2 * overlap2 + step1 * overlap1 * overlap1 * overlap1) / (6 * nz * nz);
    return term1 + term2; 
}

/**
 * Compute the integral of s^2 * \Theta(\delta_i(s)), where \Theta is the
 * Heaviside step function and \delta_i(s) is the cell-surface overlap, from
 * s = -l_i/2 to s = +l_i/2, where l_i is the length of the cell.
 *
 * The z-coordinate of the cell's orientation vector is assumed to be nonzero.
 *
 * @param rz z-coordinate of cell center.
 * @param nz z-coordinate of cell orientation.
 * @param half_l Half of cell length.
 * @param R Cell radius.
 * @returns Desired integral.  
 */
template <typename T>
T integral7(T rz, T nz, T half_l, T R)
{
    T overlap1 = R - rz + half_l * nz;
    T overlap2 = R - rz - half_l * nz;
    T step1 = step(overlap1);
    T step2 = step(overlap2);
    T term1 = half_l * half_l * (-step2 * overlap2 + step1 * overlap1) / nz;
    T term2 = 2 * integral6(rz, nz, half_l, R) / nz;
    return term1 + term2; 
}

/**
 * Compute the integral of the cell-surface contact area density a_i(s) from
 * s = -l_i/2 to s = +l_i/2, where l_i is the length of the cell.
 *
 * The z-coordinate of the cell's orientation vector is assumed to be nonzero.
 *
 * @param rz z-coordinate of cell center.
 * @param nz z-coordinate of cell orientation.
 * @param half_l Half of cell length.
 * @param R Cell radius.
 * @returns Desired integral.
 */
template <typename T>
T areaIntegral1(T rz, T nz, T half_l, T R)
{
    T overlap1 = R - rz + half_l * nz;
    T overlap2 = R - rz - half_l * nz;
    T step1 = step(overlap1);
    T step2 = step(overlap2);
    T b = step2 * overlap2 - step1 * overlap1;
    T term1 = std::pow(R, 0.5) * (1 - nz * nz) * integral1(rz, nz, half_l, R, 0.5);
    T term2 = boost::math::constants::pi<T>() * R * nz * b;
    return term1 - term2;
}

/**
 * Compute the integral of s * a_i(s), where a_i(s) is the cell-surface
 * contact area density, from s = -l_i/2 to s = +l_i/2, where l_i is the 
 * length of the cell.
 *
 * The z-coordinate of the cell's orientation vector is assumed to be nonzero.
 *
 * @param rz z-coordinate of cell center.
 * @param nz z-coordinate of cell orientation.
 * @param half_l Half of cell length.
 * @param R Cell radius.
 * @returns Desired integral.
 */
template <typename T>
T areaIntegral2(T rz, T nz, T half_l, T R)
{
    T int1 = integral2(rz, nz, half_l, R, 0.5);
    T int2 = integral5(rz, nz, half_l, R);
    T term1 = std::pow(R, 0.5) * (1 - nz * nz) * int1;
    T term2 = boost::math::constants::pi<T>() * R * nz * nz * int2;
    return term1 + term2;
}

/**
 * Compute the integral of s^2 * a_i(s), where a_i(s) is the cell-surface
 * contact area density, from s = -l_i/2 to s = +l_i/2, where l_i is the 
 * length of the cell.
 *
 * The z-coordinate of the cell's orientation vector is assumed to be nonzero.
 *
 * @param rz z-coordinate of cell center.
 * @param nz z-coordinate of cell orientation.
 * @param half_l Half of cell length.
 * @param R Cell radius.
 * @returns Desired integral.
 */
template <typename T>
T areaIntegral3(T rz, T nz, T half_l, T R)
{
    T int1 = integral3(rz, nz, half_l, R, 0.5);
    T int2 = integral7(rz, nz, half_l, R);
    T term1 = std::pow(R, 0.5) * (1 - nz * nz) * int1;
    T term2 = boost::math::constants::pi<T>() * R * nz * nz * int2;
    return term1 + term2;
}

#endif
