/**
 * A simple implementation of the Newton-Raphson method. 
 *
 * Authors:
 *     Kee-Myoung Nam
 *
 * Last updated:
 *     8/4/2025
 */

#ifndef NEWTON_RAPHSON_HPP
#define NEWTON_RAPHSON_HPP

#include <iostream>
#include <cmath>
#include <utility>
#include <limits>
#include <functional>

using std::abs;

template <typename T>
std::pair<T, T> newtonRaphson(const std::function<T(const T)>& func, const T x_init,
                              const T dx, const T xmin = 0.0,
                              const T xmax = std::numeric_limits<double>::infinity(),
                              const T tol = 1e-8, const int max_iter = 1000,
                              const T max_learn_rate = 1.0,
                              const T armijo_factor = 0.1,
                              const T contraction_factor = 0.5,
                              const bool verbose = false)
{
    T x_curr = x_init; 
    T f_curr = func(x_curr); 
    T update = std::numeric_limits<T>::infinity(); 
    int iter = 0;  
    while (abs(f_curr) > tol && iter < max_iter)
    {
        // Estimate the derivative at the current value of x 
        T deriv = (func(x_curr + dx) - func(x_curr - dx)) / (2 * dx); 

        // Determine a learning rate that satisfies the Armijo condition
        // via backtracking line search (Nocedal and Wright, Algorithm 3.1)
        //
        // Start with the maximum learning rate 
        T learn_rate = max_learn_rate;
        update = -learn_rate * deriv;
        T x_next = x_curr + update; 

        // Check if the proposed update exceeds the given bounds  
        if (x_next < xmin)
            update = xmin - x_curr; 
        else if (gamma_new > max_gamma)
            update = xmax - x_curr; 
        learn_rate = -update / deriv;
        x_next = x_curr + update; 

        // Compute the function at the new value of x
        T f_next = func(x_next);  

        // If the Armijo condition is not satisfied ... 
        while (abs(f_next) > abs(f_curr) - armijo_factor * learn_rate * deriv * deriv) 
        {
            // Lower the learning rate and try again
            learn_rate *= contraction_factor; 
            update = -learn_rate * deriv;
            x_next = x_curr + update; 

            // Check if the proposed update exceeds the given bounds  
            if (x_next < xmin)
                update = xmin - x_curr; 
            else if (gamma_new > max_gamma)
                update = xmax - x_curr; 
            learn_rate = -update / deriv;
            x_next = x_curr + update; 
            
            // Compute the function at the new value of x
            f_next = func(x_next);  
        }
        
        // Update x and f according to the given learning rate 
        x_curr = x_next; 
        f_curr = f_next; 
        iter++;

        // Print iteration details if desired 
        if (verbose)
            std::cout << "Iteration " << iter << ": x = " << x_curr << ", "
                      << "f(x) = " << f_curr << ", update = " << update
                      << std::endl; 
    }

    return std::make_pair(x_curr, f_curr);  
}

#endif
