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
                              const bool verbose = false)
{
    T x_curr = x_init; 
    T f_curr = func(x_curr); 
    T update = std::numeric_limits<T>::infinity();
    int iter = 0; 
    if (verbose)
        std::cout << "Iteration " << iter << ": x = " << x_curr << ", "
                  << "f(x) = " << f_curr << std::endl; 

    // While f(x) is far from zero ... 
    while (abs(f_curr) > tol && iter < max_iter)
    {
        // Estimate the derivative at the current value of x 
        T deriv = (func(x_curr + dx) - func(x_curr - dx)) / (2 * dx);

        // Perform the Newton-Raphson update 
        update = -f_curr / deriv;  
        x_curr += update; 
        f_curr = func(x_curr); 
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
