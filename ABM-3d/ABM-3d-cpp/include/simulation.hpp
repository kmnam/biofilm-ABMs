/**
 * Functions for running simulations with flexible initial conditions. 
 *
 * In what follows, a population of N cells is represented as a 2-D array 
 * with N rows, whose columns are as specified in `indices.hpp`.
 * 
 * Authors:
 *     Kee-Myoung Nam
 *
 * Last updated:
 *     10/3/2025
 */

#ifndef BIOFILM_SIMULATIONS_3D_HPP
#define BIOFILM_SIMULATIONS_3D_HPP

#include <iostream>
#include <fstream>
#include <cmath>
#include <chrono>
#include <vector>
#include <map>
#include <Eigen/Dense>
#include <boost/container_hash/hash.hpp>
#include <boost/random.hpp>
#include <boost/multiprecision/mpfr.hpp>
#include "indices.hpp"
#include "ellipsoid.hpp"
#include "jkr.hpp"
#include "growth.hpp"
#include "mechanics.hpp"
#include "utils.hpp"
#include "switch.hpp"

using namespace Eigen;

// Expose math functions for both standard and boost MPFR types
using std::pow;
using boost::multiprecision::pow;
using std::sqrt;
using boost::multiprecision::sqrt;
using std::max;
using boost::multiprecision::max;

/**
 * An enum that enumerates the different numerical integration methods. 
 */
enum class IntegrationMode
{
    VELOCITY_VERLET = 0, 
    HEUN_EULER = 1,
    BOGACKI_SHAMPINE = 2,
    RUNGE_KUTTA_FEHLBERG = 3,
    DORMAND_PRINCE = 4
}; 

/**
 * Return a string containing a floating-point number, specified to the 
 * given precision. 
 *
 * @param x Input value.
 * @param precision Precision.
 * @returns Output string.
 */
template <typename T>
std::string floatToString(T x, const int precision = 10)
{
    std::stringstream ss;
    ss << std::setprecision(precision);
    ss << x;
    return ss.str();
}

/**
 * Tabulate the JKR contact radius at a given collection of overlap distances.
 *
 * @param delta Input mesh of overlap distances.
 * @param R Cell radius (including the EPS). 
 * @param E0 Elastic modulus. 
 * @param imag_tol Tolerance for determining whether a root for the the JKR
 *                 contact radius polynomial is real.
 * @param aberth_tol Tolerance for Aberth-Ehrlich method. 
 * @returns Table of calculated JKR contact radii. 
 */
template <typename T, int N = 100>
std::unordered_map<int, T> calculateJKRContactRadii(const Ref<const Matrix<T, Dynamic, 1> >& delta,
                                                    const T gamma, const T R,
                                                    const T E0,
                                                    const T imag_tol = 1e-20, 
                                                    const T aberth_tol = 1e-20)
{
    std::unordered_map<int, T> radii; 

    // For each overlap distance ... 
    for (int i = 0; i < delta.size(); ++i)
    {
        // Calculate the JKR contact radius
        auto result = jkrContactRadius<T, N>(delta(i), R / 2, E0, gamma, imag_tol, aberth_tol); 
        radii[i] = result.second; 
    }

    return radii;  
} 

/**
 * Tabulate the JKR contact radius at a given collection of overlap distances
 * and surface adhesion energy densities. 
 *
 * @param delta Input mesh of overlap distances.
 * @param gamma Input mesh of surface adhesion energy densities. 
 * @param R Cell radius (including the EPS). 
 * @param E0 Elastic modulus. 
 * @param imag_tol Tolerance for determining whether a root for the the JKR
 *                 contact radius polynomial is real.
 * @param aberth_tol Tolerance for Aberth-Ehrlich method. 
 * @returns Table of calculated JKR contact radii. 
 */
template <typename T, int N = 100>
R2ToR1Table<T> calculateJKRContactRadii(const Ref<const Matrix<T, Dynamic, 1> >& delta,
                                        const Ref<const Matrix<T, Dynamic, 1> >& gamma, 
                                        const T R, const T E0,
                                        const T imag_tol = 1e-20, 
                                        const T aberth_tol = 1e-20)
{
    R2ToR1Table<T> radii; 

    // For each overlap distance ... 
    for (int i = 0; i < delta.size(); ++i)
    {
        for (int j = 0; j < gamma.size(); ++j)
        {
            // Calculate the JKR contact radius
            auto result = jkrContactRadius<T, N>(
                delta(i), R / 2, E0, gamma(j), imag_tol, aberth_tol
            ); 
            radii[std::make_pair(i, j)] = result.second; 
        }
    }

    return radii;  
} 

/**
 * Tabulate the principal radii of curvature at a given collection of contact
 * points on a cell. 
 *
 * Each contact point is parametrized by:
 *
 * (1) the angle between the overlap vector and the cell's orientation vector,
 * (2) the cell's half-length, and 
 * (3) the centerline coordinate at which the overlap vector is positioned.
 *
 * @param theta Input mesh of overlap-orientation angles. 
 * @param half_l Input mesh of cell half-lengths. 
 * @param coords Input mesh of cell centerline coordinates.
 * @param R Cell radius (including the EPS).  
 * @param calibrate_endpoint_radii If true, calibrate the principal radii of 
 *                                 curvature so that the minimum radius is R
 *                                 and the maximum radius is always greater 
 *                                 than R.
 * @param project_tol Tolerance for the ellipsoid projection method. 
 * @param project_max_iter Maximum number of iterations for the ellipsoid
 *                         projection method. 
 * @returns Table of calculated principal radii of curvature. 
 */
template <typename T>
R3ToR2Table<T> calculateCurvatureRadiiTable(const Ref<const Matrix<T, Dynamic, 1> >& theta,
                                            const Ref<const Matrix<T, Dynamic, 1> >& half_l,
                                            const Ref<const Matrix<T, Dynamic, 1> >& coords,
                                            const T R, 
                                            const bool calibrate_endpoint_radii = true, 
                                            const T project_tol = 1e-6,
                                            const int project_max_iter = 100)
{
    R3ToR2Table<T> radii; 

    // For each cell half-length ... 
    for (int j = 0; j < half_l.size(); ++j)
    {
        T rmax_factor = 1.0; 
        if (calibrate_endpoint_radii)
        {
            // Get the principal radii of curvature at the endpoint of the 
            // inscribed ellipsoid
            rmax_factor = (half_l(j) + R) / R;   // = R / (R * R / (half_l(j) + R)) 
        }

        // For each overlap-orientation angle ... 
        for (int i = 0; i < theta.size(); ++i)
        {
            // For each centerline coordinate ... 
            for (int k = 0; k < coords.size(); ++k)
            {
                // Calculate the principal radii of curvature 
                std::pair<T, T> radii_ = projectAndGetPrincipalRadiiOfCurvature<T>(
                    half_l(j), R, theta(i), coords(k) * half_l(j), project_tol,
                    project_max_iter
                );
                T rmax = radii_.first; 
                T rmin = radii_.second; 

                // Calibrate if desired
                std::tuple<int, int, int> tuple = std::make_tuple(i, j, k); 
                if (!calibrate_endpoint_radii) 
                    radii[tuple] = std::make_pair(rmax, rmin);
                else
                    radii[tuple] = std::make_pair(rmax_factor * rmax, R); 
            }
        }
    }

    return radii; 
} 

/**
 * Tabulate the JKR force magnitudes and contact radii at a given collection 
 * of cell-cell contact configurations, which are parametrized by the equivalent
 * principal radii of curvature and overlap distance. 
 *
 * Each contact point is parametrized by:
 *
 * (1, 2) the equivalent principal radii of curvature at the contact point, and
 * (3) the overlap distance. 
 *
 * The surface adhesion energy density is fixed.  
 *
 * @param Rx Input mesh of values for the larger equivalent principal radius
 *           of curvature at the contact point.  
 * @param Ry Input mesh of values for the smaller equivalent principal radius
 *           of curvature at the contact point. 
 * @param delta Input mesh of overlap distances.
 * @param gamma Surface adhesion energy density. 
 * @param E0 Elastic modulus. 
 * @param max_overlap If non-negative, cap the overlap distance at this 
 *                    maximum value.
 * @param min_aspect_ratio Minimum aspect ratio of the JKR contact area.
 * @param max_aspect_ratio Maximum aspect ratio for anisotropic JKR contacts. 
 * @param brent_tol Tolerance for Brent's method. 
 * @param brent_max_iter Maximum number of iterations for Brent's method.
 * @param init_bracket_dx Increment for bracket initialization. 
 * @param n_tries_bracket Number of attempts for bracket initialization. 
 * @param imag_tol Tolerance for determining whether a root for the the JKR
 *                 contact radius polynomial is real.
 * @param aberth_tol Tolerance for Aberth-Ehrlich method. 
 * @returns Table of calculated JKR force magnitudes and contact radii. 
 */
template <typename T, int N = 100>
R3ToR2Table<T> calculateJKRForceTable(const Ref<const Matrix<T, Dynamic, 1> >& Rx,
                                      const Ref<const Matrix<T, Dynamic, 1> >& Ry,
                                      const Ref<const Matrix<T, Dynamic, 1> >& delta,
                                      const T gamma, const T E0,
                                      const T max_overlap = -1, 
                                      const T min_aspect_ratio = 0.01,
                                      const T max_aspect_ratio = 0.99,  
                                      const T brent_tol = 1e-8, 
                                      const int brent_max_iter = 1000, 
                                      const T init_bracket_dx = 1e-3, 
                                      const int n_tries_bracket = 5,
                                      const T imag_tol = 1e-20, 
                                      const T aberth_tol = 1e-20)
{
    R3ToR2Table<T> forces; 

    // For each cell-cell configuration ...
    for (int i = 0; i < Rx.size(); ++i)
    {
        for (int j = 0; j < Ry.size(); ++j)
        {
            if (Rx(i) >= Ry(j))
            {
                std::cout << "... Calculating anisotropic JKR forces for Rx = "
                          << Rx(i) << ", Ry = " << Ry(j) << std::endl; 
                for (int k = 0; k < delta.size(); ++k)
                {
                    // Store the JKR force magnitude and contact radius 
                    auto tuple = std::make_tuple(i, j, k); 
                    auto result = jkrContactAreaAndForceEllipsoid<T, N>(
                        Rx(i), Ry(j), delta(k), E0, gamma, max_overlap, 
                        min_aspect_ratio, max_aspect_ratio, brent_tol, 
                        brent_max_iter, init_bracket_dx, n_tries_bracket, 
                        imag_tol, aberth_tol, false
                    );
                    forces[tuple] = std::make_pair(std::get<0>(result), std::get<1>(result));
                }
            }
        }
    } 

    return forces; 
}

/**
 * Tabulate the JKR force magnitudes and contact radii at a given collection 
 * of cell-cell contact configurations, which are parametrized by the equivalent
 * principal radii of curvature and overlap distance. 
 *
 * Each contact point is parametrized by:
 *
 * (1, 2) the equivalent principal radii of curvature at the contact point, 
 * (3) the overlap distance, and 
 * (4) the surface adhesion energy density.  
 *
 * @param Rx Input mesh of values for the larger equivalent principal radius
 *           of curvature at the contact point.  
 * @param Ry Input mesh of values for the smaller equivalent principal radius
 *           of curvature at the contact point. 
 * @param delta Input mesh of overlap distances.
 * @param gamma Input mesh of surface adhesion energy densities. 
 * @param E0 Elastic modulus. 
 * @param max_overlap If non-negative, cap the overlap distance at this 
 *                    maximum value.
 * @param min_aspect_ratio Minimum aspect ratio of the contact area. 
 * @param max_aspect_ratio Maximum aspect ratio for anisotropic JKR contacts. 
 * @param brent_tol Tolerance for Brent's method. 
 * @param brent_max_iter Maximum number of iterations for Brent's method.
 * @param init_bracket_dx Increment for bracket initialization. 
 * @param n_tries_bracket Number of attempts for bracket initialization. 
 * @param imag_tol Tolerance for determining whether a root for the the JKR
 *                 contact radius polynomial is real.
 * @param aberth_tol Tolerance for Aberth-Ehrlich method. 
 * @returns Table of calculated JKR force magnitudes and contact radii. 
 */
template <typename T, int N = 100>
R4ToR2Table<T> calculateJKRForceTable(const Ref<const Matrix<T, Dynamic, 1> >& Rx,
                                      const Ref<const Matrix<T, Dynamic, 1> >& Ry,
                                      const Ref<const Matrix<T, Dynamic, 1> >& delta,
                                      const Ref<const Matrix<T, Dynamic, 1> >& gamma, 
                                      const T E0, const T max_overlap = -1, 
                                      const T min_aspect_ratio = 0.01, 
                                      const T max_aspect_ratio = 0.99,  
                                      const T brent_tol = 1e-8, 
                                      const int brent_max_iter = 1000, 
                                      const T init_bracket_dx = 1e-3, 
                                      const int n_tries_bracket = 5,
                                      const T imag_tol = 1e-20, 
                                      const T aberth_tol = 1e-20)
{
    R4ToR2Table<T> forces; 

    // For each cell-cell configuration ...
    for (int i = 0; i < Rx.size(); ++i)
    {
        for (int j = 0; j < Ry.size(); ++j)
        {
            std::cout << "... Calculating anisotropic JKR forces for Rx = "
                      << Rx(i) << ", Ry = " << Ry(j) << std::endl; 
            for (int k = 0; k < delta.size(); ++k)
            {
                for (int m = 0; m < gamma.size(); ++m)
                {
                    // Store the JKR force magnitude and contact radius 
                    auto tuple = std::make_tuple(i, j, k, m); 
                    auto result = jkrContactAreaAndForceEllipsoid<T, N>(
                        Rx(i), Ry(j), delta(k), E0, gamma(m), max_overlap, 
                        min_aspect_ratio, max_aspect_ratio, brent_tol, 
                        brent_max_iter, init_bracket_dx, n_tries_bracket, 
                        imag_tol, aberth_tol, false
                    );
                    forces[tuple] = std::make_pair(std::get<0>(result), std::get<1>(result));
                } 
            }
        }
    } 

    return forces; 
}

/**
 * Parse the given file of pre-computed values for the principal radii of 
 * curvature at a collection of contact points. 
 *
 * Each contact point is parametrized by:
 *
 * (1) the angle between the overlap vector and the cell's orientation vector,
 * (2) the cell's half-length, and 
 * (3) the centerline coordinate at which the overlap vector is positioned.
 *
 * @param filename Input filename. 
 * @returns 
 */
template <typename T>
std::tuple<Matrix<T, Dynamic, 1>,
           Matrix<T, Dynamic, 1>, 
           Matrix<T, Dynamic, 1>, 
           R3ToR2Table<T> > parseCurvatureRadiiTable(const std::string& filename)
{
    std::vector<T> theta, half_l, coords; 
    R3ToR2Table<T> radii; 

    // Open the file 
    std::ifstream infile(filename);

    // Parse the first line in the file 
    std::string line, token; 
    T theta_curr, half_l_curr, coord_curr, Rx_curr, Ry_curr, 
      theta_next, half_l_next, coord_next, Rx_next, Ry_next;
    int theta_i = 0; 
    int half_l_i = 0; 
    int coord_i = 0;  
    std::stringstream ss; 
    std::getline(infile, line);
    ss << line;
    std::getline(ss, token, '\t'); 
    theta_curr = static_cast<T>(std::stod(token)); 
    std::getline(ss, token, '\t'); 
    half_l_curr = static_cast<T>(std::stod(token)); 
    std::getline(ss, token, '\t'); 
    coord_curr = static_cast<T>(std::stod(token)); 
    std::getline(ss, token, '\t'); 
    Rx_curr = static_cast<T>(std::stod(token)); 
    std::getline(ss, token, '\t'); 
    Ry_curr = static_cast<T>(std::stod(token)); 
    theta.push_back(theta_curr); 
    half_l.push_back(half_l_curr); 
    coords.push_back(coord_curr); 
    radii[std::make_tuple(0, 0, 0)] = std::make_pair(Rx_curr, Ry_curr); 

    // For each subsequent line in the file ...
    while (std::getline(infile, line))
    {
        // Get each token in the line
        ss.str(std::string()); 
        ss.clear(); 
        ss << line; 
        std::getline(ss, token, '\t'); 
        theta_next = static_cast<T>(std::stod(token)); 
        std::getline(ss, token, '\t'); 
        half_l_next = static_cast<T>(std::stod(token)); 
        std::getline(ss, token, '\t'); 
        coord_next = static_cast<T>(std::stod(token)); 
        std::getline(ss, token, '\t');
        Rx_next = static_cast<T>(std::stod(token)); 
        std::getline(ss, token, '\t'); 
        Ry_next = static_cast<T>(std::stod(token)); 

        // Check which of the three input values is new 
        if (theta_next != theta_curr)          // Encountered a new theta value 
        {
            theta.push_back(theta_next); 
            theta_i++;
            half_l_i = 0; 
            coord_i = 0; 
        }
        else if (half_l_next != half_l_curr)   // Encountered a new half_l value 
        {
            // If still processing the zeroth theta value, add to list 
            if (theta_i == 0)
                half_l.push_back(half_l_next); 
            half_l_i++; 
            coord_i = 0; 
        } 
        else     // Encountered a new coord value 
        {
            // If still processing the zeroth theta and half_l values, add to list
            if (theta_i == 0 && half_l_i == 0)
                coords.push_back(coord_next); 
            coord_i++; 
        }
        auto tuple = std::make_tuple(theta_i, half_l_i, coord_i); 
        radii[tuple] = std::make_pair(Rx_next, Ry_next); 

        theta_curr = theta_next; 
        half_l_curr = half_l_next; 
        coord_curr = coord_next; 
        Rx_curr = Rx_next; 
        Ry_curr = Ry_next; 
    }

    Matrix<T, Dynamic, 1> theta_(theta.size()); 
    for (int i = 0; i < theta.size(); ++i)
        theta_(i) = theta[i]; 
    Matrix<T, Dynamic, 1> half_l_(half_l.size()); 
    for (int i = 0; i < half_l.size(); ++i)
        half_l_(i) = half_l[i]; 
    Matrix<T, Dynamic, 1> coords_(coords.size()); 
    for (int i = 0; i < coords.size(); ++i)
        coords_(i) = coords[i];  
    return std::make_tuple(theta_, half_l_, coords_, radii); 
}

/**
 * Parse the given file of pre-computed values for the JKR force and contact
 * radius at a collection of cell-cell configurations. 
 *
 * Each contact point is parametrized by:
 *
 * (1, 2) the equivalent principal radii of curvature at the contact point, and 
 * (3) the overlap distance.
 *
 * The surface adhesion energy density is fixed.  
 *
 * @param filename Input filename. 
 * @returns 
 */
template <typename T>
std::tuple<Matrix<T, Dynamic, 1>,
           Matrix<T, Dynamic, 1>, 
           Matrix<T, Dynamic, 1>, 
           T,
           R3ToR2Table<T> > parseReducedJKRForceTable(const std::string& filename)
{
    std::vector<T> Rx, Ry, delta;
    T gamma; 
    R3ToR2Table<T> forces;

    // Open the file 
    std::ifstream infile(filename);

    // Parse the first line in the file, which specifies the surface adhesion
    // energy density 
    std::string line, token; 
    std::getline(infile, line); 
    gamma = static_cast<T>(std::stod(line)); 

    // Parse the first line in the file 
    T Rx_curr, Ry_curr, delta_curr, force_curr, radius_curr,
      Rx_next, Ry_next, delta_next, force_next, radius_next;
    int Rx_i = 0; 
    int Ry_i = 0; 
    int delta_i = 0; 
    std::getline(infile, line);
    std::stringstream ss; 
    ss << line; 
    std::getline(ss, token, '\t');
    Rx_curr = static_cast<T>(std::stod(token)); 
    std::getline(ss, token, '\t'); 
    Ry_curr = static_cast<T>(std::stod(token)); 
    std::getline(ss, token, '\t'); 
    delta_curr = static_cast<T>(std::stod(token)); 
    std::getline(ss, token, '\t'); 
    force_curr = static_cast<T>(std::stod(token)); 
    std::getline(ss, token, '\t'); 
    radius_curr = static_cast<T>(std::stod(token));
    Rx.push_back(Rx_curr); 
    Ry.push_back(Ry_curr); 
    delta.push_back(delta_curr); 
    forces[std::make_tuple(0, 0, 0)] = std::make_pair(force_curr, radius_curr); 

    // For each subsequent line in the file ...
    while (std::getline(infile, line))
    {
        // Get each token in the line
        ss.str(std::string()); 
        ss.clear(); 
        ss << line; 
        std::getline(ss, token, '\t');
        Rx_next = static_cast<T>(std::stod(token)); 
        std::getline(ss, token, '\t'); 
        Ry_next = static_cast<T>(std::stod(token)); 
        std::getline(ss, token, '\t'); 
        delta_next = static_cast<T>(std::stod(token)); 
        std::getline(ss, token, '\t'); 
        force_next = static_cast<T>(std::stod(token)); 
        std::getline(ss, token, '\t'); 
        radius_next = static_cast<T>(std::stod(token));

        // Check which of the four input values is new
        if (Rx_next != Rx_curr)         // Encountered a new Rx value 
        {
            Rx.push_back(Rx_next); 
            Rx_i++;
            Ry_i = 0; 
            delta_i = 0; 
        }
        else if (Ry_next != Ry_curr)    // Encountered a new Ry value 
        {
            // If still processing the zeroth Rx value, add to list 
            if (Rx_i == 0)
                Ry.push_back(Ry_next); 
            Ry_i++; 
            delta_i = 0; 
        } 
        else     // Encountered a new delta value 
        {
            // If still processing the zeroth Rx and Ry values, add to list
            if (Rx_i == 0 && Ry_i == 0)
                delta.push_back(delta_next); 
            delta_i++; 
        }
        auto tuple = std::make_tuple(Rx_i, Ry_i, delta_i); 
        forces[tuple] = std::make_pair(force_next, radius_next); 

        Rx_curr = Rx_next; 
        Ry_curr = Ry_next; 
        delta_curr = delta_next; 
        force_curr = force_next; 
        radius_curr = radius_next; 
    }

    Matrix<T, Dynamic, 1> Rx_(Rx.size());
    for (int i = 0; i < Rx.size(); ++i)
        Rx_(i) = Rx[i];  
    Matrix<T, Dynamic, 1> Ry_(Ry.size());
    for (int i = 0; i < Ry.size(); ++i)
        Ry_(i) = Ry[i];  
    Matrix<T, Dynamic, 1> delta_(delta.size());
    for (int i = 0; i < delta.size(); ++i)
        delta_(i) = delta[i]; 
    return std::make_tuple(Rx_, Ry_, delta_, gamma, forces);  
}

/**
 * Parse the given file of pre-computed values for the JKR force and contact
 * radius at a collection of cell-cell configurations. 
 *
 * Each contact point is parametrized by:
 *
 * (1, 2) the equivalent principal radii of curvature at the contact point, 
 * (3) the overlap distance, and 
 * (4) the surface adhesion energy density.  
 *
 * @param filename Input filename. 
 * @returns 
 */
template <typename T>
std::tuple<Matrix<T, Dynamic, 1>,
           Matrix<T, Dynamic, 1>, 
           Matrix<T, Dynamic, 1>, 
           Matrix<T, Dynamic, 1>,
           R4ToR2Table<T> > parseJKRForceTable(const std::string& filename)
{
    std::vector<T> Rx, Ry, delta, gamma; 
    R4ToR2Table<T> forces;

    // Open the file 
    std::ifstream infile(filename);

    // Parse the first line in the file 
    std::string line, token; 
    T Rx_curr, Ry_curr, delta_curr, gamma_curr, force_curr, radius_curr,
      Rx_next, Ry_next, delta_next, gamma_next, force_next, radius_next;
    int Rx_i = 0; 
    int Ry_i = 0; 
    int delta_i = 0; 
    int gamma_i = 0; 
    std::stringstream ss; 
    std::getline(infile, line);
    ss << line; 
    std::getline(ss, token, '\t');
    Rx_curr = static_cast<T>(std::stod(token)); 
    std::getline(ss, token, '\t'); 
    Ry_curr = static_cast<T>(std::stod(token)); 
    std::getline(ss, token, '\t'); 
    delta_curr = static_cast<T>(std::stod(token)); 
    std::getline(ss, token, '\t'); 
    gamma_curr = static_cast<T>(std::stod(token)); 
    std::getline(ss, token, '\t'); 
    force_curr = static_cast<T>(std::stod(token)); 
    std::getline(ss, token, '\t'); 
    radius_curr = static_cast<T>(std::stod(token));
    Rx.push_back(Rx_curr); 
    Ry.push_back(Ry_curr); 
    delta.push_back(delta_curr); 
    gamma.push_back(gamma_curr);
    forces[std::make_tuple(0, 0, 0, 0)] = std::make_pair(force_curr, radius_curr); 

    // For each subsequent line in the file ...
    while (std::getline(infile, line))
    {
        // Get each token in the line
        ss.str(std::string()); 
        ss.clear(); 
        ss << line; 
        std::getline(ss, token, '\t');
        Rx_next = static_cast<T>(std::stod(token)); 
        std::getline(ss, token, '\t'); 
        Ry_next = static_cast<T>(std::stod(token)); 
        std::getline(ss, token, '\t'); 
        delta_next = static_cast<T>(std::stod(token)); 
        std::getline(ss, token, '\t'); 
        gamma_next = static_cast<T>(std::stod(token)); 
        std::getline(ss, token, '\t'); 
        force_next = static_cast<T>(std::stod(token)); 
        std::getline(ss, token, '\t'); 
        radius_next = static_cast<T>(std::stod(token));

        // Check which of the four input values is new
        if (Rx_next != Rx_curr)         // Encountered a new Rx value 
        {
            Rx.push_back(Rx_next); 
            Rx_i++;
            Ry_i = 0; 
            delta_i = 0; 
            gamma_i = 0;  
        }
        else if (Ry_next != Ry_curr)    // Encountered a new Ry value 
        {
            // If still processing the zeroth Rx value, add to list 
            if (Rx_i == 0)
                Ry.push_back(Ry_next); 
            Ry_i++; 
            delta_i = 0; 
            gamma_i = 0; 
        } 
        else if (delta_next != delta_curr)    // Encountered a new delta value
        {
            // If still processing the zeroth Rx and Ry values, add to list
            if (Rx_i == 0 && Ry_i == 0)
                delta.push_back(delta_next); 
            delta_i++; 
            gamma_i = 0; 
        }
        else     // Encountered a new gamma value 
        {
            // If still processing the zeroth Rx, Ry, and delta values, add to list
            if (Rx_i == 0 && Ry_i == 0 && delta_i == 0)
                gamma.push_back(gamma_next); 
            gamma_i++; 
        }
        auto tuple = std::make_tuple(Rx_i, Ry_i, delta_i, gamma_i); 
        forces[tuple] = std::make_pair(force_next, radius_next); 

        Rx_curr = Rx_next; 
        Ry_curr = Ry_next; 
        delta_curr = delta_next; 
        gamma_curr = gamma_next; 
        force_curr = force_next; 
        radius_curr = radius_next; 
    }

    Matrix<T, Dynamic, 1> Rx_(Rx.size());
    for (int i = 0; i < Rx.size(); ++i)
        Rx_(i) = Rx[i];  
    Matrix<T, Dynamic, 1> Ry_(Ry.size());
    for (int i = 0; i < Ry.size(); ++i)
        Ry_(i) = Ry[i];  
    Matrix<T, Dynamic, 1> delta_(delta.size());
    for (int i = 0; i < delta.size(); ++i)
        delta_(i) = delta[i]; 
    Matrix<T, Dynamic, 1> gamma_(gamma.size());
    for (int i = 0; i < gamma.size(); ++i)
        gamma_(i) = gamma[i]; 
    return std::make_tuple(Rx_, Ry_, delta_, gamma_, forces);  
}

/**
 * Run a simulation with the given initial population of cells.
 *
 * This function runs simulations in which the cells switch between multiple 
 * groups that differ by growth rate and an additional physical attribute.
 * The growth rate and chosen physical attribute are taken to be normally
 * distributed variables that exhibit a specified mean and standard deviation.
 *
 * @param cells_init Initial population of cells.
 * @param parents_init Initial vector of parent cell IDs for each cell generated
 *                     throughout the simulation. 
 * @param max_iter Maximum number of iterations. 
 * @param n_cells Maximum number of cells.
 * @param max_time Maximum simulation time.  
 * @param R Cell radius (including the EPS). 
 * @param Rcell Cell radius (excluding the EPS).
 * @param L0 Initial cell length.
 * @param Ldiv Cell division length.
 * @param E0 Elastic modulus of EPS.
 * @param Ecell Elastic modulus of cell.
 * @param max_stepsize Maximum stepsize per iteration.
 * @param min_stepsize Minimum stepsize per iteration. 
 * @param write If true, write simulation output to file(s). 
 * @param outprefix Output filename prefix. 
 * @param dt_write Write cells to file during each iteration in which the time
 *                 has passed a multiple of this value. 
 * @param iter_update_neighbors Update neighboring cells every this many 
 *                              iterations. 
 * @param iter_update_stepsize Update stepsize every this many iterations. 
 * @param max_error_allowed Upper bound on maximum Runge-Kutta error allowed
 *                          per iteration.
 * @param min_error Minimum Runge-Kutta error. 
 * @param max_tries_update_stepsize Maximum number of tries to update stepsize
 *                                  due to Runge-Kutta error. 
 * @param neighbor_threshold Threshold for distinguishing between neighboring
 *                           and non-neighboring cells.
 * @param nz_threshold Threshold for determining whether the z-orientation of 
 *                     each cell is zero.  
 * @param rng_seed Random number generator seed. 
 * @param n_groups Number of groups.
 * @param group_attributes Indices of attributes that differ between groups.
 * @param growth_means Mean growth rate for cells in each group.
 * @param growth_stds Standard deviation of growth rate for cells in each
 *                    group.
 * @param switch_mode Switching mode. Can by NONE (0), MARKOV (1), or INHERIT
 *                    (2).
 * @param switch_rates Array of between-group switching rates. In the Markovian
 *                     mode (`switch_mode` is MARKOV), this is the matrix of
 *                     transition rates; in the inheritance mode (`switch_mode`
 *                     is INHERIT), this is the matrix of transition probabilities
 *                     at each division event. 
 * @param daughter_length_std Standard deviation of daughter length ratio 
 *                            distribution. 
 * @param daughter_angle_xy_bound Bound on daughter cell re-orientation angle
 *                                in xy-plane.
 * @param daughter_angle_z_bound Bound on daughter cell re-orientation angle 
 *                               out of xy-plane.
 * @param max_rxy_noise Maximum noise to be added to each generalized force in
 *                      the x- and y-directions.
 * @param max_rz_noise Maximum noise to be added to each generalized force in
 *                     the z-direction.
 * @param max_nxy_noise Maximum noise to be added to each generalized torque in
 *                      the x- and y-directions.
 * @param max_nz_noise Maximum noise to be added to each generalized torque in
 *                     the z-direction.
 * @param basal_only If true, keep track of only the basal layer of cells
 *                   throughout the simulation. 
 * @param basal_min_overlap A cell is in the basal layer if its cell-surface 
 *                          overlap is greater than this value. Can be negative.
 * @param adhesion_mode Choice of potential used to model cell-cell adhesion.
 *                      Can be NONE (0), KIHARA (1), or GBK (2).
 * @param adhesion_params Parameters required to compute cell-cell adhesion
 *                        forces.
 * @param adhesion_curvature_filename File containing pre-computed values for
 *                                    principal radii of curvature. 
 * @param adhesion_jkr_forces_filename File containing pre-computed values for
 *                                     JKR forces. 
 * @param no_surface If true, omit the surface from the simulations. 
 * @param n_cells_start_switch Number of cells at which to begin switching.
 *                             All switching is suppressed until this number
 *                             of cells is reached. 
 * @param track_poles If true, keep track of pole birth times.
 * @returns Final population of cells.  
 */
template <typename T>
std::pair<Array<T, Dynamic, Dynamic>, std::vector<int> >
    runSimulation(const Ref<const Array<T, Dynamic, Dynamic> >& cells_init,
                  std::vector<int>& parents_init,
                  const IntegrationMode integration_mode, 
                  const int max_iter,
                  const int n_cells,
                  const T max_time, 
                  const T R,
                  const T Rcell,
                  const T L0,
                  const T Ldiv,
                  const T E0,
                  const T Ecell,
                  const T M0, 
                  const T max_stepsize,
                  const T min_stepsize,
                  const bool write,
                  const std::string outprefix,
                  const T dt_write,
                  const int iter_update_neighbors,
                  const int iter_update_stepsize,
                  const T max_error_allowed,
                  const T min_error,
                  const int max_tries_update_stepsize,
                  const T neighbor_threshold,
                  const T nz_threshold,
                  const int rng_seed,
                  const int n_groups,
                  std::vector<int>& group_attributes,
                  const Ref<const Array<T, Dynamic, 1> >& growth_means,
                  const Ref<const Array<T, Dynamic, 1> >& growth_stds,
                  const Ref<const Array<T, Dynamic, Dynamic> >& attribute_values,
                  const SwitchMode switch_mode,
                  const Ref<const Array<T, Dynamic, Dynamic> >& switch_rates,
                  const T switch_timescale, 
                  const T daughter_length_std,
                  const T daughter_angle_xy_bound,
                  const T daughter_angle_z_bound,
                  const T max_rxy_noise,
                  const T max_rz_noise,
                  const T max_nxy_noise,
                  const T max_nz_noise, 
                  const bool basal_only,
                  const T basal_min_overlap, 
                  const AdhesionMode adhesion_mode, 
                  std::unordered_map<std::string, T>& adhesion_params,
                  const std::string adhesion_curvature_filename, 
                  const std::string adhesion_jkr_forces_filename, 
                  const FrictionMode friction_mode,
                  const bool no_surface = false,
                  const int n_cells_start_switch = 0,
                  const bool track_poles = false,
                  const T cell_cell_coulomb_coeff = 1.0,
                  const T cell_surface_coulomb_coeff = 1.0,
                  const int n_start_multithread = 50)
{
    Array<T, Dynamic, Dynamic> cells(cells_init);
    T t = 0;
    T dt = max_stepsize; 
    int iter = 0;
    int n = cells.rows();
    auto t_real = std::chrono::system_clock::now();
    bool started_multithread = false;  
    boost::random::mt19937 rng(rng_seed);

    // Define additional column indices, if desired
    int __colidx_gamma = -1;
    int __colidx_eta_cell_cell = -1;  
    int __colidx_negpole_t0 = -1; 
    int __colidx_pospole_t0 = -1;
    if (adhesion_mode != AdhesionMode::NONE)
    {
        __colidx_gamma = 21;  
    }
    if (friction_mode != FrictionMode::NONE)
    {
        __colidx_eta_cell_cell = (adhesion_mode != AdhesionMode::NONE ? 22 : 21);
    }
    if (track_poles)
    {
        if (adhesion_mode != AdhesionMode::NONE && friction_mode != FrictionMode::NONE)
        {
            __colidx_negpole_t0 = 23; 
            __colidx_pospole_t0 = 24;
        }
        else if (adhesion_mode != AdhesionMode::NONE || friction_mode != FrictionMode::NONE)
        {
            __colidx_negpole_t0 = 22; 
            __colidx_pospole_t0 = 23; 
        }
        else 
        {
            __colidx_negpole_t0 = 21;  
            __colidx_pospole_t0 = 22; 
        }
    }

    // Define Butcher tableau for the desired Runge-Kutta method
    Array<T, Dynamic, Dynamic> A;
    Array<T, Dynamic, 1> b, bs;
    T error_order = 0; 
    if (integration_mode == IntegrationMode::HEUN_EULER)
    {
        // Heun-Euler, order 2(1)
        A.resize(2, 2); 
        A << 0, 0, 1, 0; 
        b.resize(2); 
        b << 0.5, 0.5; 
        bs.resize(2); 
        bs << 1, 0;
        error_order = 1;  
    } 
    else if (integration_mode == IntegrationMode::BOGACKI_SHAMPINE)
    { 
        // Bogacki-Shampine, order 3(2)
        A.resize(4, 4); 
        A << 0,     0,     0,     0,
             1./2., 0,     0,     0,
             0,     3./4., 0,     0,
             2./9., 1./3., 4./9., 0;
        b.resize(4); 
        b << 2./9., 1./3., 4./9., 0;
        bs.resize(4); 
        bs << 7./24., 1./4., 1./3., 1./8.;
        error_order = 2;
    }
    else if (integration_mode == IntegrationMode::RUNGE_KUTTA_FEHLBERG)
    {
        // Runge-Kutta-Fehlberg, order 5(4)
        A.resize(6, 6);
        A << 0,           0,            0,            0,           0,        0,
             1./4.,       0,            0,            0,           0,        0,
             3./32.,      9./32.,       0,            0,           0,        0,
             1932./2197., -7200./2197., 7296./2197.,  0,           0,        0,
             439./216.,   -8.0,         3680./513.,   -845./4104., 0,        0,
             -8./27.,     2.0,          -3544./2565., 1859./4104., -11./40., 0;
        b.resize(6); 
        b << 16./135., 0, 6656./12825., 28561./56430., -9./50., 2./55.;
        bs.resize(6); 
        bs << 25./216., 0, 1408./2565., 2197./4104., -1./5., 0; 
        error_order = 4;
    }
    else if (integration_mode == IntegrationMode::DORMAND_PRINCE)
    {
        // Dormand-Prince, order 5(4)
        A.resize(7, 7); 
        A << 0,            0,             0,            0,         0,             0,       0,
             1./5.,        0,             0,            0,         0,             0,       0,
             3./40.,       9./40.,        0,            0,         0,             0,       0,
             44./45.,      -56./15.,      32./9.,       0,         0,             0,       0,
             19372./6561., -25360./2187., 64448./6561., -212./729, 0,             0,       0,
             9017./3168.,  -355./33.,     46732./5247., 49./176.,  -5103./18656., 0,       0,
             35./384.,     0,             500./1113.,   125./192., -2187./6784.,  11./84., 0;
        b.resize(7); 
        b << 35./384., 0, 500./1113., 125./192., -2187./6784., 11./84., 0;
        bs.resize(7); 
        bs << 5179./57600., 0, 7571./16695., 393./640., -92097./339200., 187./2100., 1./40.;
        error_order = 4;
    }

    // Prefactors for cell-cell repulsion forces 
    Array<T, 3, 1> repulsion_prefactors;
    repulsion_prefactors << (4. / 3.) * E0 * sqrt(R / 2),
                            (4. / 3.) * Ecell * sqrt(Rcell / 2), 
                            (4. / 3.) * E0 * sqrt(R / 2) * pow(2 * (R - Rcell), 1.5);

    // Compute initial array of neighboring cells
    Array<T, Dynamic, 7> neighbors = getCellNeighbors<T>(cells, neighbor_threshold, R, Ldiv);

    // Define cell density, if velocity Verlet is to be used
    T rho = 0; 
    if (integration_mode == IntegrationMode::VELOCITY_VERLET)
    {
        if (M0 <= 0)
            throw std::runtime_error(
                "Initial cell mass must be defined when using velocity Verlet"
            );

        T volume = boost::math::constants::pi<T>() * ((4. / 3.) * pow(R, 3) + R * R * L0);
        rho = M0 / volume; 
    }

    // Initialize parent IDs
    std::vector<int> parents(parents_init); 

    // Growth rate distribution functions: normal distributions with given means
    // and standard deviations
    const int n_attributes = group_attributes.size(); 
    boost::random::uniform_01<> uniform_dist; 
    std::vector<std::function<T(boost::random::mt19937&)> > growth_dists; 
    for (int i = 0; i < n_groups; ++i)
    { 
        T growth_mean = growth_means(i);
        T growth_std = growth_stds(i);
        std::function<T(boost::random::mt19937&)> growth_dist =
            [growth_mean, growth_std, &uniform_dist](boost::random::mt19937& rng)
            {
                return growth_mean + growth_std * standardNormal<T>(rng, uniform_dist);
            };
        growth_dists.push_back(growth_dist);
    }

    // Daughter cell length ratio distribution function: normal distribution
    // with mean 0.5 and given standard deviation
    std::function<T(boost::random::mt19937&)> daughter_length_dist =
        [daughter_length_std, &uniform_dist](boost::random::mt19937& rng)
        {
            T r = 0.5 + daughter_length_std * standardNormal<T>(rng, uniform_dist);
            if (r < 0)
                return 0.0;
            else if (r > 1)
                return 1.0;
            else
                return r; 
        };

    // Daughter angle distribution functions: two uniform distributions that 
    // are bounded by the given values 
    std::function<T(boost::random::mt19937&)> daughter_angle_xy_dist = 
        [daughter_angle_xy_bound, &uniform_dist](boost::random::mt19937& rng)
        {
            T r = static_cast<T>(uniform_dist(rng));
            return -daughter_angle_xy_bound + 2 * daughter_angle_xy_bound * r;
        };
    std::function<T(boost::random::mt19937&)> daughter_angle_z_dist = 
        [daughter_angle_z_bound, &uniform_dist](boost::random::mt19937& rng)
        {
            T r = static_cast<T>(uniform_dist(rng));
            return -daughter_angle_z_bound + 2 * daughter_angle_z_bound * r;
        };

    JKRData<T> jkr_data;
    jkr_data.max_gamma = 0;
    if (adhesion_mode != AdhesionMode::NONE)
    {
        // First parse polynomial-solving parameters, if given 
        T imag_tol = 1e-20; 
        T aberth_tol = 1e-20;
        if (adhesion_params.find("jkr_imag_tol") != adhesion_params.end())
            imag_tol = adhesion_params["jkr_imag_tol"]; 
        if (adhesion_params.find("jkr_aberth_tol") != adhesion_params.end())
            aberth_tol = adhesion_params["jkr_aberth_tol"];

        // Parse the desired equilibrium cell-cell distance and solve for 
        // the corresponding surface energy density  
        const T eqdist = adhesion_params["eqdist"];
        jkr_data.max_gamma = jkrOptimalSurfaceEnergyDensity<T, 100>(
            R, Rcell, E0, eqdist, 100.0, 1e-6, 1e-8, 1e-8, 1e-8, 1000, 1000,
            imag_tol, aberth_tol, true
        );
        jkr_data.gamma_fixed = (switch_mode == SwitchMode::NONE);
        if (switch_mode == SwitchMode::NONE || switch_timescale == 0)
            jkr_data.gamma_switch_rate = std::numeric_limits<T>::infinity(); 
        else
            jkr_data.gamma_switch_rate = jkr_data.max_gamma / switch_timescale;  

        // Initialize the surface energy density for each cell to be 
        // the maximum value for group 1 cells and zero for group 2 cells
        for (int i = 0; i < n; ++i)
        {
            if (cells(i, __colidx_group) == 1)
                cells(i, __colidx_gamma) = jkr_data.max_gamma; 
            else 
                cells(i, __colidx_gamma) = 0.0; 
        }

        // If isotropic JKR adhesion is desired, calculate JKR contact radii
        // for a range of overlap distances and surface energy densities
        if (adhesion_mode == AdhesionMode::JKR_ISOTROPIC)
        {
            int n_overlap = static_cast<int>(adhesion_params["n_mesh_overlap"]);
            int n_gamma = static_cast<int>(adhesion_params["n_mesh_gamma"]);
            jkr_data.overlaps = Matrix<T, Dynamic, 1>::LinSpaced(
                n_overlap, 0, 2 * (R - Rcell)
            );
            if (!jkr_data.gamma_fixed)
            {
                jkr_data.gamma = Matrix<T, Dynamic, 1>::LinSpaced(
                    n_gamma, 0, jkr_data.max_gamma
                );
                jkr_data.contact_radii = calculateJKRContactRadii<T, 100>(
                    jkr_data.overlaps, jkr_data.gamma, R, E0, imag_tol, aberth_tol
                );
            }
            else 
            {
                jkr_data.contact_radii_reduced = calculateJKRContactRadii<T, 100>(
                    jkr_data.overlaps, jkr_data.max_gamma, R, E0, imag_tol,
                    aberth_tol
                );
            } 
        }
        else     // Otherwise, if anisotropic JKR adhesion is desired ... 
        {
            // Determine whether the principal radii of curvature should be
            // computed 
            const bool precompute_jkr_forces = static_cast<bool>(
                adhesion_params["precompute_jkr_forces"]
            );

            // Calculate the principal radii of curvature and JKR contact
            // forces, if desired 
            if (precompute_jkr_forces)
            {
                bool calibrate_endpoint_radii = static_cast<bool>(
                    adhesion_params["calibrate_endpoint_radii"]
                ); 
                int n_theta = static_cast<int>(adhesion_params["n_mesh_theta"]);
                int n_half_l = static_cast<int>(adhesion_params["n_mesh_half_l"]);
                int n_coords = static_cast<int>(
                    adhesion_params["n_mesh_centerline_coords"]
                );
                int n_Rx = static_cast<int>(adhesion_params["n_mesh_curvature_radii"]); 
                int n_Ry = static_cast<int>(adhesion_params["n_mesh_curvature_radii"]); 
                T max_overlap = 2 * (R - Rcell);
                T min_aspect_ratio = adhesion_params["min_aspect_ratio"];
                T max_aspect_ratio = adhesion_params["max_aspect_ratio"]; 
                T project_tol = adhesion_params["ellipsoid_project_tol"]; 
                int project_max_iter = static_cast<int>(
                    adhesion_params["ellipsoid_project_max_iter"]
                ); 
                T brent_tol = adhesion_params["brent_tol"];
                int brent_max_iter = static_cast<int>(
                    adhesion_params["brent_max_iter"]
                );
                T init_bracket_dx = adhesion_params["init_bracket_dx"]; 
                int n_tries_bracket = static_cast<int>(
                    adhesion_params["n_tries_bracket"]
                ); 
                jkr_data.theta = Matrix<T, Dynamic, 1>::LinSpaced(
                    n_theta, 0.0, boost::math::constants::half_pi<T>()
                ); 
                jkr_data.half_l = Matrix<T, Dynamic, 1>::LinSpaced(
                    n_half_l, 0.5 * L0, 0.5 * Ldiv
                );  
                jkr_data.centerline_coords = Matrix<T, Dynamic, 1>::LinSpaced(
                    n_coords, 0.0, 1.0
                );
                jkr_data.curvature_radii = calculateCurvatureRadiiTable<T>(
                    jkr_data.theta, jkr_data.half_l, jkr_data.centerline_coords, 
                    R, calibrate_endpoint_radii, project_tol, project_max_iter
                );
                jkr_data.Rx = Matrix<T, Dynamic, 1>::LinSpaced(
                    n_Rx, 0.5 * R, R
                ); 
                jkr_data.Ry = Matrix<T, Dynamic, 1>::LinSpaced(
                    n_Ry, 0.5 * R, R
                );
                if (!jkr_data.gamma_fixed)
                {
                    jkr_data.forces = calculateJKRForceTable<T>(
                        jkr_data.Rx, jkr_data.Ry, jkr_data.overlaps,
                        jkr_data.gamma, E0, max_overlap, min_aspect_ratio,
                        max_aspect_ratio, brent_tol, brent_max_iter, 
                        init_bracket_dx, n_tries_bracket, imag_tol, aberth_tol
                    );
                }
                else 
                {
                    jkr_data.forces_reduced = calculateJKRForceTable<T>(
                        jkr_data.Rx, jkr_data.Ry, jkr_data.overlaps,
                        jkr_data.max_gamma, E0, max_overlap, min_aspect_ratio,
                        max_aspect_ratio, brent_tol, brent_max_iter, 
                        init_bracket_dx, n_tries_bracket, imag_tol, aberth_tol
                    );
                } 
            }
            else    // Otherwise, parse pre-computed values 
            {
                auto result1 = parseCurvatureRadiiTable<T>(adhesion_curvature_filename); 
                jkr_data.theta = std::get<0>(result1); 
                jkr_data.half_l = std::get<1>(result1); 
                jkr_data.centerline_coords = std::get<2>(result1); 
                jkr_data.curvature_radii = std::get<3>(result1);
                if (!jkr_data.gamma_fixed)
                { 
                    auto result2 = parseJKRForceTable<T>(adhesion_jkr_forces_filename); 
                    jkr_data.Rx = std::get<0>(result2); 
                    jkr_data.Ry = std::get<1>(result2); 
                    jkr_data.overlaps = std::get<2>(result2); 
                    jkr_data.gamma = std::get<3>(result2); 
                    jkr_data.forces = std::get<4>(result2);
                }
                else
                { 
                    auto result2 = parseReducedJKRForceTable<T>(adhesion_jkr_forces_filename); 
                    jkr_data.Rx = std::get<0>(result2); 
                    jkr_data.Ry = std::get<1>(result2); 
                    jkr_data.overlaps = std::get<2>(result2); 
                    jkr_data.max_gamma = std::get<3>(result2); 
                    jkr_data.forces_reduced = std::get<4>(result2);
                }
            }
        }
    }
    
    // Write simulation parameters to a dictionary
    std::map<std::string, std::string> params;
    const int precision = 10;
    params["n_cells"] = std::to_string(n_cells);
    params["R"] = floatToString<T>(R, precision);
    params["Rcell"] = floatToString<T>(Rcell, precision);
    params["L0"] = floatToString<T>(L0, precision);
    params["Ldiv"] = floatToString<T>(Ldiv, precision);
    params["E0"] = floatToString<T>(E0, precision);
    params["Ecell"] = floatToString<T>(Ecell, precision);
    params["integration_mode"] = std::to_string(static_cast<int>(integration_mode)); 
    params["max_stepsize"] = floatToString<T>(max_stepsize, precision);
    params["min_stepsize"] = floatToString<T>(min_stepsize, precision); 
    params["dt_write"] = floatToString<T>(dt_write, precision); 
    params["iter_update_neighbors"] = std::to_string(iter_update_neighbors);
    params["iter_update_stepsize"] = std::to_string(iter_update_stepsize);
    params["max_error_allowed"] = floatToString<T>(max_error_allowed, precision);
    params["max_tries_update_stepsize"] = std::to_string(max_tries_update_stepsize);
    params["neighbor_threshold"] = floatToString<T>(neighbor_threshold, precision);
    params["nz_threshold"] = floatToString<T>(nz_threshold, precision);
    params["random_seed"] = std::to_string(rng_seed);
    params["n_groups"] = std::to_string(n_groups);
    for (int i = 0; i < n_attributes; ++i)
    {
        std::stringstream ss; 
        ss << "group_attribute" << i + 1;
        params[ss.str()] = std::to_string(group_attributes[i]);
    }
    for (int i = 0; i < n_groups; ++i)
    {
        std::stringstream ss; 
        ss << "growth_mean" << i + 1;
        params[ss.str()] = floatToString<T>(growth_means(i), precision);
        ss.str(std::string());
        ss << "growth_std" << i + 1; 
        params[ss.str()] = floatToString<T>(growth_stds(i), precision);
        ss.str(std::string());
        for (int j = 0; j < n_attributes; ++j)
        {
            ss << "attribute_values_" << i + 1 << "_" << j + 1;
            params[ss.str()] = floatToString<T>(attribute_values(i, j), precision);
            ss.str(std::string());
        }
    }
    params["switch_mode"] = std::to_string(static_cast<int>(switch_mode));
    if (switch_mode != SwitchMode::NONE)
    {
        std::stringstream ss;
        for (int i = 0; i < n_groups; ++i)
        {
            for (int j = i + 1; j < n_groups; ++j)
            {
                ss << "switch_rate_" << i + 1 << "_" << j + 1;
                params[ss.str()] = floatToString<T>(switch_rates(i, j), precision);
                ss.str(std::string());
                ss << "switch_rate_" << j + 1 << "_" << i + 1;
                params[ss.str()] = floatToString<T>(switch_rates(j, i), precision);
                ss.str(std::string()); 
            }
        }
    }
    params["switch_timescale"] = floatToString<T>(switch_timescale, precision); 
    params["daughter_length_std"] = floatToString<T>(daughter_length_std, precision);
    params["daughter_angle_xy_bound"] = floatToString<T>(daughter_angle_xy_bound, precision);
    params["daughter_angle_z_bound"] = floatToString<T>(daughter_angle_z_bound, precision);
    params["max_rxy_noise"] = floatToString<T>(max_rxy_noise, precision);
    params["max_rz_noise"] = floatToString<T>(max_rz_noise, precision);
    params["max_nxy_noise"] = floatToString<T>(max_nxy_noise, precision);
    params["max_nz_noise"] = floatToString<T>(max_nz_noise, precision);
    params["basal_only"] = (basal_only ? "1" : "0"); 
    params["basal_min_overlap"] = floatToString<T>(basal_min_overlap, precision);  
    params["adhesion_mode"] = std::to_string(static_cast<int>(adhesion_mode)); 
    if (adhesion_mode != AdhesionMode::NONE)
    {
        for (auto&& item : adhesion_params)
        {
            std::stringstream ss; 
            std::string key = item.first; 
            T value = item.second;
            ss << "adhesion_" << key; 
            params[ss.str()] = floatToString<T>(value); 
        }
    }
    params["cell_cell_friction_mode"] = std::to_string(static_cast<int>(friction_mode));
    params["cell_cell_coulomb_coeff"] = floatToString<T>(cell_cell_coulomb_coeff, precision);
    params["cell_surface_coulomb_coeff"] = floatToString<T>(cell_surface_coulomb_coeff, precision);  
    params["track_poles"] = (track_poles ? "1" : "0");
    params["no_surface"] = (no_surface ? "1" : "0");
    params["n_cells_start_switch"] = std::to_string(n_cells_start_switch);

    // Write the initial population to file
    std::unordered_map<int, int> write_other_cols;
    if (adhesion_mode != AdhesionMode::NONE)
    {
        write_other_cols[__colidx_gamma] = 1;    // Write adhesion energy densities as floats
    }
    if (friction_mode != FrictionMode::NONE)
    {
        write_other_cols[__colidx_eta_cell_cell] = 1;   // Write friction coefficients as floats
    }
    if (track_poles)
    {
        write_other_cols[__colidx_negpole_t0] = 1;      // Write pole ages as floats
        write_other_cols[__colidx_pospole_t0] = 1;
    }
    if (write)
    {
        params["t_curr"] = floatToString<T>(t);
        std::stringstream ss_init; 
        ss_init << outprefix << "_init.txt";
        std::string filename_init = ss_init.str(); 
        writeCells<T>(cells, params, filename_init, write_other_cols);
    }
    
    // Define termination criterion, assuming that at least one of n_cells,
    // max_iter, or max_time is positive
    std::function<bool(int, int, T)> terminate = [&n_cells, &max_iter, &max_time](int n, int iter, T t) -> bool
    {
        // If none of the three criteria was given, then the termination
        // criterion is ill-defined, so always terminate
        if (n_cells <= 0 && max_iter <= 0 && max_time <= 0) 
            return true;

        // If all three criteria are given ...  
        if (n_cells > 0 && max_iter > 0 && max_time > 0)
        {
            return (n >= n_cells || iter >= max_iter || t >= max_time);
        } 
        // If n_cells was not given ... 
        else if (n_cells <= 0)
        {
            // Check if both max_iter and max_time were given
            if (max_iter > 0 && max_time > 0)
                return (iter >= max_iter || t >= max_time);
            else if (max_iter > 0)    // Otherwise, max_iter or max_time must have been given 
                return (iter >= max_iter); 
            else
                return (t >= max_time); 
        }
        // If max_iter was not given ...
        else if (max_iter <= 0)
        {
            // Here, n_cells must have been given, so we check if max_time 
            // was given  
            if (max_time > 0)
                return (n >= n_cells || t >= max_time); 
            else 
                return (n >= n_cells);  
        }
        else   // Otherwise, n_cells and max_iter must have been given 
        {
            return (n >= n_cells || iter >= max_iter); 
        } 
    };

    // Run the simulation ...
    while (!terminate(n, iter, t))
    {
        // Divide the cells that have reached division length
        Array<int, Dynamic, 1> to_divide = divideMaxLength<T>(cells, Ldiv);
        std::vector<std::pair<int, int> > daughter_pairs; 
        if (to_divide.sum() > 0)
            std::cout << "... Dividing " << to_divide.sum() << " cells "
                      << "(iteration " << iter << ")" << std::endl;
        if (track_poles)    // Track poles if desired 
        {
            auto div_result = divideCellsWithPoles<T>(
                cells, parents, t, R, Rcell, to_divide, growth_dists, rng,
                daughter_length_dist, daughter_angle_xy_dist, daughter_angle_z_dist,
                __colidx_negpole_t0, __colidx_pospole_t0
            );
            cells = div_result.first;
            daughter_pairs = div_result.second;
        }
        else                // Otherwise, simply divide 
        {
            auto div_result = divideCells<T>(
                cells, parents, t, R, Rcell, to_divide, growth_dists, rng,
                daughter_length_dist, daughter_angle_xy_dist, daughter_angle_z_dist
            );
            cells = div_result.first; 
            daughter_pairs = div_result.second; 
        }
        n = cells.rows();

        // If division has occurred ... 
        if (to_divide.sum() > 0)
        {
            // Switch cells between groups if desired 
            if (n >= n_cells_start_switch && switch_mode == SwitchMode::INHERIT)
            {
                switchGroupsInherit<T>(
                    cells, daughter_pairs, n_groups, switch_rates, growth_dists,
                    rng, uniform_dist
                );
            }
            
            // Update neighboring cells 
            neighbors = getCellNeighbors<T>(cells, neighbor_threshold, R, Ldiv);
        }

        // Update cell positions and orientations, using the desired integration
        // method 
        #ifdef _OPENMP
            if (n >= n_start_multithread && !started_multithread)
            {
                std::cout << "[NOTE] Started multithreading: detected "
                          << omp_get_max_threads() << " threads" << std::endl;
                started_multithread = true; 
            }
        #endif
        Array<T, Dynamic, Dynamic> cells_new;
        Array<T, Dynamic, 6> errors = Array<T, Dynamic, 6>::Zero(n, 6);  
        if (integration_mode != IntegrationMode::VELOCITY_VERLET)
        {
            // Update using adaptive Runge-Kutta
            auto result = stepRungeKuttaAdaptive<T>(
                A, b, bs, cells, neighbors, dt, iter, R, Rcell, E0, 
                repulsion_prefactors, nz_threshold, max_rxy_noise, max_rz_noise,
                max_nxy_noise, max_nz_noise, rng, uniform_dist, adhesion_mode,
                adhesion_params, jkr_data, __colidx_gamma, friction_mode,
                __colidx_eta_cell_cell, no_surface, cell_cell_coulomb_coeff,
                cell_surface_coulomb_coeff, n_start_multithread
            ); 
            cells_new = result.first;
            errors = result.second;

            // If the error is big, retry the step with a smaller stepsize (up to
            // a given maximum number of attempts)
            if (iter % iter_update_stepsize == 0)
            {
                // Enforce a composite error of the form tol * (1 + y), for the
                // maximum error
                //
                // Here, y (which determines the scale of the error) is taken to 
                // be the old cell positions and orientations 
                Array<T, Dynamic, 6> z = (
                    Array<T, Dynamic, 6>::Ones(n, 6) + cells(Eigen::all, __colseq_coords).abs()
                ); 
                Array<T, Dynamic, 6> max_scale = max_error_allowed * z;
                T max_error = max((errors / max_scale).maxCoeff(), min_error); 
                bool error_exceeded = (max_error > 1.0); 

                // Ensure that the updated stepsize is between 0.2 times and 10 times
                // the previous stepsize
                T factor = 0.9 * pow(1.0 / max_error, 1.0 / (error_order + 1)); 
                if (factor >= 10)
                    factor = 10;
                else if (factor < 0.2)
                    factor = 0.2;
                int j = 0;
                while (error_exceeded && j < max_tries_update_stepsize)
                {
                    // Try updating the stepsize by the given factor and re-run 
                    // the integration 
                    T dt_new = dt * factor; 
                    result = stepRungeKuttaAdaptive<T>(
                        A, b, bs, cells, neighbors, dt_new, iter, R, Rcell, E0,
                        repulsion_prefactors, nz_threshold, max_rxy_noise,
                        max_rz_noise, max_nxy_noise, max_nz_noise, rng, uniform_dist,
                        adhesion_mode, adhesion_params, jkr_data, __colidx_gamma,
                        friction_mode, __colidx_eta_cell_cell, no_surface,
                        cell_cell_coulomb_coeff, cell_surface_coulomb_coeff,
                        n_start_multithread
                    ); 
                    cells_new = result.first;
                    errors = result.second;

                    // Compute the new error
                    max_error = max((errors / max_scale).maxCoeff(), min_error); 
                    error_exceeded = (max_error > 1.0);  

                    // Multiply by the new factor (note that this factor is being 
                    // multiplied to the *original* dt to determine the new stepsize,
                    // so the factors across all loop iterations must be accumulated)
                    factor *= 0.9 * pow(1.0 / max_error, 1.0 / (error_order + 1)); 
                    if (factor >= 10)
                    {
                        factor = 10;
                        break;
                    }
                    else if (factor < 0.2)
                    {
                        factor = 0.2;
                        break;
                    }
                    j++;  
                }
                
                // Ensure that the proposed stepsize is between the minimum and 
                // maximum
                if (dt * factor < min_stepsize)
                    factor = min_stepsize / dt; 
                else if (dt * factor > max_stepsize)
                    factor = max_stepsize / dt;

                // Re-do the integration with the new stepsize
                dt *= factor;
                result = stepRungeKuttaAdaptive<T>(
                    A, b, bs, cells, neighbors, dt, iter, R, Rcell, E0, 
                    repulsion_prefactors, nz_threshold, max_rxy_noise, max_rz_noise,
                    max_nxy_noise, max_nz_noise, rng, uniform_dist, adhesion_mode,
                    adhesion_params, jkr_data, __colidx_gamma, friction_mode,
                    __colidx_eta_cell_cell, no_surface, cell_cell_coulomb_coeff,
                    cell_surface_coulomb_coeff, n_start_multithread
                ); 
                cells_new = result.first;
                errors = result.second;
            }
            // If desired, print a warning message if the error is big
            #ifdef DEBUG_WARN_LARGE_ERROR
                Array<T, Dynamic, 6> z = (
                    Array<T, Dynamic, 6>::Ones(n, 6) + cells(Eigen::all, __colseq_coords).abs()
                );
                Array<T, Dynamic, 6> max_scale = max_error_allowed * z;
                T max_error = max((errors / max_scale).maxCoeff(), min_error);
                if (max_error > 5)
                {
                    std::cout << "[WARN] Maximum error = " << max_error
                              << " is > 5 times the desired error "
                              << "(absolute tol = relative tol = " << max_error_allowed
                              << ", iteration " << iter << ", time = " << t
                              << ", dt = " << dt << ")" << std::endl;
                }
            #endif
        }
        else 
        {
            // Update using velocity Verlet 
            cells_new = stepVelocityVerlet<T>(
                cells, neighbors, dt, iter, R, Rcell, E0, rho, repulsion_prefactors,
                nz_threshold, max_rxy_noise, max_rz_noise, max_nxy_noise,
                max_nz_noise, rng, uniform_dist, adhesion_mode, adhesion_params,
                jkr_data, __colidx_gamma, friction_mode, __colidx_eta_cell_cell,
                no_surface, cell_cell_coulomb_coeff, cell_surface_coulomb_coeff,
                n_start_multithread
            ); 
        }
        cells = cells_new;
        
        // If desired, check that the cell coordinates do not contain any 
        // undefined values 
        #ifdef DEBUG_CHECK_CELL_COORDINATES_NAN
           for (int i = 0; i < cells.rows(); ++i)
           {
               if (cells.row(i).isNaN().any() || cells.row(i).isInf().any())
               {
                   std::cerr << std::setprecision(10);
                   std::cerr << "Iteration " << iter
                             << ": Data for cell " << i << " contains nan" << std::endl;
                   std::cerr << "Timestep: " << dt << std::endl;
                   std::cerr << "Data: (";
                   for (int j = 0; j < cells.cols() - 1; ++j)
                   {
                       std::cerr << cells(i, j) << ", "; 
                   }
                   std::cerr << cells(i, cells.cols() - 1) << ")" << std::endl;
                   throw std::runtime_error("Found nan in cell coordinates"); 
               }
           } 
        #endif

        // Grow the cells
        growCells<T>(cells, dt, R);

        // Update distances between neighboring cells
        updateNeighborDistances<T>(cells, neighbors);

        // If desired, pick out only the cells that overlap with the surface,
        // updating array of neighboring cells whenever a cell is deleted
        if (!no_surface && basal_only)
        { 
            // Since we assume that the z-orientation is always positive, 
            // the maximum cell-surface overlap for each cell occurs at 
            // centerline coordinate -l/2
            //
            // Note that these overlaps can be negative for cells that do
            // not touch the surface 
            Array<T, Dynamic, 1> max_overlaps = (
                R - cells.col(__colidx_rz) + cells.col(__colidx_half_l) * cells.col(__colidx_nz)
            );
            std::vector<int> overlap_idx; 
            for (int j = 0; j < cells.rows(); ++j)
            {
                if (max_overlaps(j) > basal_min_overlap)
                    overlap_idx.push_back(j);
            }
            
            // If there are cells that do not touch the surface, then throw
            // them out 
            if (overlap_idx.size() < cells.rows())
            {
                cells = cells(overlap_idx, Eigen::all).eval();
                neighbors = getCellNeighbors<T>(cells, neighbor_threshold, R, Ldiv);
            }
        }
        
        // Update current time 
        t += dt;
        iter++;

        // Update neighboring cells 
        if (iter % iter_update_neighbors == 0)
            neighbors = getCellNeighbors<T>(cells, neighbor_threshold, R, Ldiv);

        // Switch cells between groups if desired 
        if (n >= n_cells_start_switch && switch_mode == SwitchMode::MARKOV)
        {
            switchGroupsMarkov<T>(
                cells, n_groups, dt, switch_rates, growth_dists, rng, uniform_dist
            );
        }

        // If there is cell switching, update group attributes
        if (switch_mode != SwitchMode::NONE)
        {
            // Switch each group attribute ... 
            for (int k = 0; k < n_attributes; ++k)
            {
                T target1 = attribute_values(0, k);   // Target value for group 1
                T target2 = attribute_values(1, k);   // Target value for group 2
                T rate;                               // Rate of attribute change 
                if (switch_timescale == 0) 
                    rate = std::numeric_limits<T>::infinity(); 
                else 
                    rate = abs(target1 - target2) / switch_timescale;

                // For each cell ...  
                for (int i = 0; i < n; ++i)
                {
                    int group = static_cast<int>(cells(i, __colidx_group));
                    T value = cells(i, group_attributes[k]);
                    if (group == 1)
                    {
                        if (switch_timescale == 0 && value != target1)
                        {
                            cells(i, group_attributes[k]) = target1; 
                        }
                        else if (target1 > target2 && value < target1)
                        {
                            cells(i, group_attributes[k]) += rate * dt;
                            if (cells(i, group_attributes[k]) > target1)
                                cells(i, group_attributes[k]) = target1;  
                        }
                        else if (target1 < target2 && value > target1) 
                        {
                            cells(i, group_attributes[k]) -= rate * dt;
                            if (cells(i, group_attributes[k]) < target1)
                                cells(i, group_attributes[k]) = target1;  
                        }
                        else if (target1 == target2 && value != target1)
                        {
                            cells(i, group_attributes[k]) = target1; 
                        } 
                    }
                    else    // cells(i, __colidx_group == 2)
                    {
                        if (switch_timescale == 0 && value != target2)
                        {
                            cells(i, group_attributes[k]) = target2; 
                        }
                        else if (target2 > target1 && value < target2)
                        {
                            cells(i, group_attributes[k]) += rate * dt; 
                            if (cells(i, group_attributes[k]) > target2)
                                cells(i, group_attributes[k]) = target2;  
                        }
                        else if (target2 < target1 && value > target2) 
                        {
                            cells(i, group_attributes[k]) -= rate * dt;
                            if (cells(i, group_attributes[k]) < target2)
                                cells(i, group_attributes[k]) = target2;  
                        }
                        else if (target1 == target2 && value != target2)
                        {
                            cells(i, group_attributes[k]) = target2; 
                        } 
                    }
                }
            } 

            // ... as well as the surface adhesion energy density, if there
            // is cell-cell adhesion 
            if (adhesion_mode != AdhesionMode::NONE)
            {
                for (int i = 0; i < n; ++i)
                {
                    if (cells(i, __colidx_group) == 1)
                    {
                        if (cells(i, __colidx_gamma) < jkr_data.max_gamma)
                        {
                            cells(i, __colidx_gamma) += jkr_data.gamma_switch_rate * dt; 
                            if (cells(i, __colidx_gamma) > jkr_data.max_gamma)
                                cells(i, __colidx_gamma) = jkr_data.max_gamma;
                        } 
                    }
                    else    // cells(i, __colidx_group == 2)
                    {
                        if (cells(i, __colidx_gamma) > 0)
                        {
                            cells(i, __colidx_gamma) -= jkr_data.gamma_switch_rate * dt; 
                            if (cells(i, __colidx_gamma) < 0)
                                cells(i, __colidx_gamma) = 0; 
                        }                     
                    } 
                }
            }
        }

        // Write the current population to file if the simulation time has 
        // just passed a multiple of dt_write 
        double t_old_factor = std::fmod(t - dt + 1e-12, dt_write);
        double t_new_factor = std::fmod(t + 1e-12, dt_write);  
        if (write && t_old_factor > t_new_factor) 
        {
            auto t_now = std::chrono::system_clock::now();
            std::chrono::duration<double> elapsed = t_now - t_real;
            t_real = t_now;
            if (integration_mode != IntegrationMode::VELOCITY_VERLET)
            { 
                std::cout << "Iteration " << iter << ": " << n << " cells, time = "
                          << t << ", time elapsed = " << elapsed.count() << " sec"
                          << ", max error = " << errors.abs().maxCoeff()
                          << ", avg error = " << errors.abs().sum() / (6 * n)
                          << ", dt = " << dt << std::endl;
            }
            else 
            {
                std::cout << "Iteration " << iter << ": " << n << " cells, time = "
                          << t << ", time elapsed = " << elapsed.count() << " sec"
                          << ", dt = " << dt << std::endl;
            }
            params["t_curr"] = floatToString<T>(t);
            std::stringstream ss; 
            ss << outprefix << "_iter" << iter << ".txt"; 
            std::string filename = ss.str();
            writeCells<T>(cells, params, filename, write_other_cols);
        }
    }

    // Write final population to file
    if (write)
    {
        params["t_curr"] = floatToString<T>(t);
        std::stringstream ss_final; 
        ss_final << outprefix << "_final.txt";
        std::string filename_final = ss_final.str(); 
        writeCells<T>(cells, params, filename_final, write_other_cols);
    }

    // Write complete lineage to file 
    if (write)
    {
        std::stringstream ss_lineage; 
        ss_lineage << outprefix << "_lineage.txt"; 
        writeLineage<T>(parents, ss_lineage.str());
    }

    return std::make_pair(cells, parents);
}

#endif
