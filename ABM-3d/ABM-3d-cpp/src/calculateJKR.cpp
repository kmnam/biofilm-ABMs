/**
 * Calculate JKR contact forces over a collection of cell-cell configurations. 
 *
 * Authors:
 *     Kee-Myoung Nam
 *
 * Last updated:
 *     6/8/2025
 */
#include <iostream>
#include <fstream>
#include <string>
#include <iomanip>
#include <utility>
#include <Eigen/Dense>
#include <boost/multiprecision/mpfr.hpp>
#include <boost/random.hpp>
#include "../include/adhesion.hpp"
#include "../include/jkr.hpp"

using boost::multiprecision::abs;
using boost::multiprecision::sqrt;
using boost::multiprecision::log10; 
using boost::multiprecision::pow; 

typedef boost::multiprecision::number<boost::multiprecision::mpfr_float_backend<100> > PreciseType; 

/**
 * Calculate the Hertz-JKR equilibrium cell-cell distance for the given
 * surface energy density.
 *
 * @param R Cell radius (including the EPS). 
 * @param E Elastic modulus.
 * @param gamma Surface energy density. 
 * @param delta_min Minimum overlap distance. 
 * @param delta_max Maximum overlap distance.
 * @param rng Random number generator.  
 * @param d_delta Increment for finite differences approximation.
 * @param tol Tolerance for Newton's method.  
 * @param imag_tol Tolerance for determining whether a root for the the JKR
 *                 contact radius polynomial is real.
 * @param aberth_tol Tolerance for Aberth-Ehrlich method. 
 */
template <typename T>
T jkrEquilibriumDistance(const T R, const T E, const T gamma, const T delta_min, 
                         const T delta_max, boost::random::mt19937& rng,
                         const T d_delta = 1e-8, const T tol = 1e-8, 
                         const T imag_tol = 1e-8, const T aberth_tol = 1e-20)
{
    // Get an initial cell-cell distance
    T dmin = 2 * R - delta_max;
    T dmax = 2 * R - delta_min;
    boost::random::uniform_01<> uniform_dist;  
    T dist = dmin + (dmax - dmin) * uniform_dist(rng);
    T delta = 2 * R - dist;  
    T update = std::numeric_limits<T>::infinity();
    int iter = 0; 

    // While the update between consecutive overlaps is larger than the 
    // given tolerance ... 
    while (abs(update) > tol)
    {
        // Compute the contact radius for each overlap in the mesh
        T radius = jkrContactRadius<T>(delta, R, E, gamma, imag_tol, aberth_tol).second;

        // Compute the corresponding force 
        T f_hertz = (4. / 3.) * E * radius * radius * radius / R; 
        T f_jkr = 4 * sqrt(boost::math::constants::pi<T>() * gamma * E) * pow(radius, 1.5);
        T force = f_hertz - f_jkr;

        // Estimate the derivative of this force w.r.t the overlap
        T radius_plus = jkrContactRadius<T>(delta + d_delta, R, E, gamma, imag_tol, aberth_tol).second;
        T radius_minus = jkrContactRadius<T>(delta - d_delta, R, E, gamma, imag_tol, aberth_tol).second;
        T f_hertz_plus = (4. / 3.) * E * radius_plus * radius_plus * radius_plus / R; 
        T f_jkr_plus = 4 * sqrt(boost::math::constants::pi<T>() * gamma * E) * pow(radius_plus, 1.5); 
        T force_plus = f_hertz_plus - f_jkr_plus; 
        T f_hertz_minus = (4. / 3.) * E * radius_minus * radius_minus * radius_minus / R; 
        T f_jkr_minus = 4 * sqrt(boost::math::constants::pi<T>() * gamma * E) * pow(radius_minus, 1.5); 
        T force_minus = f_hertz_minus - f_jkr_minus;
        T deriv = (force_plus - force_minus) / (2 * d_delta);

        // Update the overlap according to Newton's method 
        update = -force / deriv;
        delta += update;
    }

    return 2 * R - delta;  
}

/**
 * Calculate the surface energy density for which the desired Hertz-JKR 
 * equilibrium cell-cell distance is achieved.
 *
 * @param R Cell radius (including the EPS). 
 * @param E Elastic modulus.
 * @param deq_target Target equilibrium cell-cell distance. 
 * @param gamma_min Minimum surface energy density.
 * @param gamma_max Maximum surface energy density.  
 * @param delta_min Minimum overlap distance. 
 * @param delta_max Maximum overlap distance.
 * @param rng Random number generator.
 * @param tol Tolerance for (steepest) gradient descent.
 * @param d_log_gamma Increment for finite differences approximation w.r.t.
 *                    log10(gamma) during gradient descent. 
 * @param max_learn_rate Maximum learning rate.
 * @param d_delta Increment for finite differences approximation w.r.t. 
 *                overlap in jkrEquilibriumDistance().
 * @param newton_tol Tolerance for Newton's method in jkrEquilibriumDistance().  
 * @param imag_tol Tolerance for determining whether a root for the the JKR
 *                 contact radius polynomial is real.
 * @param aberth_tol Tolerance for Aberth-Ehrlich method.
 * @param verbose If true, print iteration details to stdout.  
 */
template <typename T>
T jkrOptimalSurfaceEnergyDensity(const T R, const T E, const T deq_target, 
                                 const T gamma_min, const T gamma_max,
                                 const T delta_min, const T delta_max,
                                 boost::random::mt19937& rng, const T tol = 1e-8,
                                 const T d_log_gamma = 1e-6,
                                 const T max_learn_rate = 1.0,  
                                 const T d_delta = 1e-8, const T newton_tol = 1e-8,
                                 const T imag_tol = 1e-8, const T aberth_tol = 1e-20,
                                 const bool verbose = false)
{
    // Get an initial value for gamma
    boost::random::uniform_01<> dist;  
    T log_gamma = log10(gamma_min) + (log10(gamma_max) - log10(gamma_min)) * dist(rng);
    T gamma = pow(10.0, log_gamma); 
    T update = std::numeric_limits<T>::infinity();
    int iter = 0; 
    if (verbose)
        std::cout << std::setprecision(10);

    // Calculate the deviation of the current equilibrium distance from
    // the target equilibrium distance 
    T deq = jkrEquilibriumDistance<T>(
        R, E, gamma, delta_min, delta_max, rng, d_delta, newton_tol,
        imag_tol, aberth_tol
    );
    T error = abs(deq - deq_target);

    // While the update between consecutive values of gamma is larger than
    // the given tolerance ... 
    while (abs(update) > tol)
    {
        // Estimate the derivative of this deviation w.r.t. gamma
        T gamma_plus = pow(10.0, log_gamma + d_log_gamma);
        T gamma_minus = pow(10.0, log_gamma - d_log_gamma);  
        T deq_plus = jkrEquilibriumDistance<T>(
            R, E, gamma_plus, delta_min, delta_max, rng, d_delta, newton_tol,
            imag_tol, aberth_tol
        );
        T error_plus = abs(deq_plus - deq_target);  
        T deq_minus = jkrEquilibriumDistance<T>(
            R, E, gamma_minus, delta_min, delta_max, rng, d_delta, newton_tol,
            imag_tol, aberth_tol
        );
        T error_minus = abs(deq_minus - deq_target);  
        T deriv = (error_plus - error_minus) / (2 * d_log_gamma);

        // Determine a learning rate that satisfies the Armijo condition
        // via backtracking line search (Nocedal and Wright, Algorithm 3.1)
        //
        // Start with the maximum learning rate 
        T learn_rate = max_learn_rate;
        update = -learn_rate * deriv;

        // Check if the proposed update exceeds the given bounds  
        if (pow(10.0, log_gamma + update) < gamma_min)
        {
            update = log10(gamma_min) - log_gamma;
            learn_rate = -update / deriv; 
        }
        else if (pow(10.0, log_gamma + update) > gamma_max)
        {
            update = log10(gamma_max) - log_gamma;
            learn_rate = -update / deriv; 
        }

        // Compute the new cell-cell equilibrium distance 
        T deq_new = jkrEquilibriumDistance<T>(
            R, E, pow(10.0, log_gamma + update), delta_min, delta_max, rng,
            d_delta, newton_tol, imag_tol, aberth_tol
        );
        T error_new = abs(deq_new - deq_target);

        // If the Armijo condition is not satisfied (set c = 0.1) ...  
        while (error_new > error - 0.1 * learn_rate * deriv * deriv)
        {
            // Lower the learning rate and try again (set contraction factor
            // \rho = 0.5) 
            learn_rate *= 0.5;
            update = -learn_rate * deriv;

            // Check if the proposed update exceeds the given bounds  
            if (pow(10.0, log_gamma + update) < gamma_min)
            {
                update = log10(gamma_min) - log_gamma;
                learn_rate = -update / deriv; 
            }
            else if (pow(10.0, log_gamma + update) > gamma_max)
            {
                update = log10(gamma_max) - log_gamma;
                learn_rate = -update / deriv; 
            }
            
            // Compute the new cell-cell equilibrium distance 
            deq_new = jkrEquilibriumDistance<T>(
                R, E, pow(10.0, log_gamma + update), delta_min, delta_max, rng,
                d_delta, newton_tol, imag_tol, aberth_tol
            );
            error_new = abs(deq_new - deq_target); 
        }
        
        // Print iteration details if desired 
        if (verbose)
            std::cout << "Iteration " << iter << ": gamma = " << gamma << ", "
                      << "deq = " << deq << ", log update = " << update << std::endl; 

        // Update gamma according to the given learning rate 
        log_gamma += update;
        gamma = pow(10.0, log_gamma + update); 
        deq = deq_new; 
        error = error_new; 
        iter++; 
    } 

    return gamma; 
}

int main(int argc, char** argv)
{
    const PreciseType R = 0.8;
    const PreciseType Rcell = 0.5; 
    const PreciseType E = 3900.0;
    const PreciseType gamma_min = 100.0; 
    const PreciseType gamma_max = 1000.0;
    const PreciseType delta_min = 0.0;
    const PreciseType delta_max = 2 * R - 2 * Rcell;
    boost::random::mt19937 rng(1234567890);
    const PreciseType tol = 1e-8; 
    const PreciseType d_log_gamma = 1e-8;  
    const PreciseType max_learn_rate = 1.0;
    const PreciseType d_delta = 1e-8;
    const PreciseType newton_tol = 1e-8;
    const PreciseType imag_tol = 1e-8; 
    const PreciseType aberth_tol = 1e-20; 
    const bool verbose = true;
    
    PreciseType deq_target = 1.3;    // Surface separation of 300 nm 
    jkrOptimalSurfaceEnergyDensity<PreciseType>(
        R, E, deq_target, gamma_min, gamma_max, delta_min, delta_max, rng, tol,
        d_log_gamma, max_learn_rate, d_delta, newton_tol, imag_tol, aberth_tol,
        verbose
    );
    deq_target = 1.2;                // Surface separation of 200 nm 
    jkrOptimalSurfaceEnergyDensity<PreciseType>(
        R, E, deq_target, gamma_min, gamma_max, delta_min, delta_max, rng, tol,
        d_log_gamma, max_learn_rate, d_delta, newton_tol, imag_tol, aberth_tol,
        verbose
    ); 
    deq_target = 1.1;                // Surface separation of 100 nm 
    jkrOptimalSurfaceEnergyDensity<PreciseType>(
        R, E, deq_target, gamma_min, gamma_max, delta_min, delta_max, rng, tol,
        d_log_gamma, max_learn_rate, d_delta, newton_tol, imag_tol, aberth_tol,
        verbose
    ); 
}
