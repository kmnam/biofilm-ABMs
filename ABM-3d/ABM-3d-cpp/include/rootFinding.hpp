/**
 * A simple implementation of the Newton-Raphson and Brent methods for
 * root-finding.  
 *
 * Authors:
 *     Kee-Myoung Nam
 *
 * Last updated:
 *     9/30/2025
 */

#ifndef NEWTON_RAPHSON_BRENT_HPP
#define NEWTON_RAPHSON_BRENT_HPP

#include <iostream>
#include <cmath>
#include <utility>
#include <limits>
#include <functional>
#include <Eigen/Dense>

using std::abs;
using std::max;
using std::min;
using std::isnan;
using std::isinf; 

using namespace Eigen; 

/**
 * Use the Newton-Raphson method to identify a root of the given function. 
 *
 * @param func Input function. 
 * @param x0 Initial x-value. 
 * @param dx Increment for finite difference approximation. 
 * @param tol Terminate the algorithm if the function value is below the 
 *            given tolerance.
 * @param max_iter Maximum number of iterations. 
 * @param verbose If true, print intermittent output to stdout.
 * @returns The localized root and the value of the function at the root.   
 */
template <typename T>
std::pair<T, T> newtonRaphson(const std::function<T(const T)>& func, const T x0,
                              const T dx, const T tol = 1e-8, const int max_iter = 1000,
                              const bool verbose = false)
{
    // Initialize the algorithm 
    T x_curr = x0; 
    T f_curr = func(x_curr); 
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
        T update = -f_curr / deriv;  
        x_curr += update;
        f_curr = func(x_curr); 
        iter++;

        // Print iteration details if desired 
        if (verbose)
            std::cout << "Iteration " << iter << ": update = " << update << ", "
                      << "x = " << x_curr << ", "  << "f(x) = " << f_curr 
                      << std::endl; 
    }

    return std::make_pair(x_curr, f_curr);  
}

/**
 * Identify a bracket of x-values for the given function.
 *
 * The function should evaluate to different signs at the two ends of the 
 * bracket.  
 *
 * @param func Input function. 
 * @param xmin Minimum x-value. 
 * @param xmax Maximum x-value. 
 * @param dx Increment for mesh generation. 
 * @returns Bracket of x-values for the given function. 
 * @throws std::runtime_error If a valid bracket could not be identified.  
 */
template <typename T>
std::pair<T, T> findBracket(const std::function<T(const T)>& func, const T xmin,
                            const T xmax, const T dx)
{
    const int n = static_cast<int>((xmax - xmin) / dx); 
    Array<T, Dynamic, 1> mesh = Array<T, Dynamic, 1>::LinSpaced(n, xmin, xmax);

    // Calculate the function's value at each point along the mesh 
    Array<T, Dynamic, 1> fmesh = Array<T, Dynamic, 1>::Zero(n); 
    for (int i = 0; i < n; ++i)
        fmesh(i) = func(mesh(i));

    // Find indices at which the function crosses the zero line 
    std::vector<int> idx; 
    for (int i = 0; i < n - 1; ++i)
    {
        T f1 = fmesh(i);
        T f2 = fmesh(i + 1); 
        if (!isinf(f1) && !isnan(f1) && !isinf(f2) && !isnan(f2))
        {
            if ((f1 > 0 && f2 < 0) || (f1 < 0 && f2 > 0))
            {
                idx.push_back(i); 
            }
        }
    }

    // Return the bracket corresponding to the largest zero-crossing x-value 
    if (idx.size() > 0)
    {
        int i = idx[idx.size() - 1];
        return std::make_pair(mesh(i), mesh(i + 1));  
    }
    else 
    {
        throw std::runtime_error("Failed to find bracket within the given range");
    } 
}

/**
 * Use Brent's method to identify a root of the given function.
 *
 * @param func Input function. 
 * @param xmin Minimum value of initial bracket. 
 * @param xmax Maximum value of initial bracket. 
 * @param tol Terminate the algorithm if the interval width reaches below the
 *            given tolerance.
 * @param max_iter Maximum number of iterations. 
 * @param verbose If true, print intermittent output to stdout.
 * @returns The interval containing the root. 
 */
template <typename T>
std::pair<T, T> brent(const std::function<T(const T)>& func, const T xmin,
                      const T xmax, const T tol = 1e-8, const int max_iter = 1000,
                      const bool verbose = false)
{
    // Initialize bracket [a, b]
    T a = xmin; 
    T b = xmax;
    T fa = func(a); 
    T fb = func(b);

    // Set c = a (the bracket going forward is [b, c] or [c, b])
    T c = a;  
    T fc = fa;

    // Bracket widths from previous two iterations 
    T d = b - a;     // Last iteration
    T e = b - a;     // Second-to-last iteration

    T update = std::numeric_limits<double>::infinity(); 
    for (int i = 0; i < max_iter; ++i)
    {
        // Swap b and c if desired
        if (abs(fb) > abs(fc))
        {
            a = b; 
            b = c; 
            c = a;
            fa = fb; 
            fb = fc; 
            fc = fa; 
        }
        
        // Check whether b is a root or the interval width is below the desired
        // tolerance 
        T m = 0.5 * (c - b);
        if (abs(m) <= tol || fb == 0)
            return std::make_pair(b, b);

        // Identify the update, using either secant method or inverse quadratic
        // interpolation
        bool use_interpolation = false;  
        if (abs(e) >= tol && abs(fa) > abs(fb))
        {
            T p, q; 
            T s = fb / fa;

            if (a == c)
            {
                // Use secant method
                p = 2 * m * s; 
                q = 1 - s;
            } 
            else 
            {
                // Use inverse quadratic interpolation (Eqn. 9.3.2 in Numerical
                // Recipes in C)
                q = fa / fc; 
                T r = fb / fc; 
                p = s * (2 * m * q * (q - r) - (b - a) * (r - 1));
                q = (q - 1) * (r - 1) * (s - 1); 
            } 

            // Fix signs for p and q 
            if (p > 0) 
                q *= -1; 
            p = abs(p); 

            // Decide whether to accept the update 
            T min1 = 3 * m * q - abs(tol * q); 
            T min2 = abs(e * q);
            if (2 * p < min(min1, min2))     // Accept 
            {
                d = p / q;
                use_interpolation = true;  
            }
        }

        // If the interpolation is undesired, use bisection
        if (!use_interpolation)
        {
            if (m > 0)
                d = max(abs(m), tol); 
            else 
                d = -max(abs(m), tol);
            e = d;  
        }
        else    // Otherwise, use the interpolation  
        {
            e = d; 
        }

        // Define the updated estimate and bracket  
        T z = b + d;
        T fz = func(z);
        a = b; 
        fa = fb; 
        if ((fz < 0 && fc < 0) || (fz > 0 && fc > 0))
        {
            c = z;  
            fc = fz; 
        }
        else 
        {
            b = z; 
            fb = fz; 
        }

        if (verbose)
        {
            if (b < c)
                std::cout << "Iteration " << i << ": [" << b << ", " << c << "]" << std::endl; 
            else 
                std::cout << "Iteration " << i << ": [" << c << ", " << b << "]" << std::endl; 
        }
    }

    if (b < c)
        return std::make_pair(b, c);
    else 
        return std::make_pair(c, b);  
}

#endif
