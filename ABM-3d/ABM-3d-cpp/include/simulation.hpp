/**
 * Functions for running simulations with flexible initial conditions. 
 *
 * In what follows, a population of N cells is represented as a 2-D array of 
 * size (N, 13), where each row represents a cell and stores the following data:
 * 
 * 0) x-coordinate of cell center
 * 1) y-coordinate of cell center
 * 2) z-coordinate of cell center
 * 3) x-coordinate of cell orientation vector
 * 4) y-coordinate of cell orientation vector
 * 5) z-coordinate of cell orientation vector
 * 6) cell length (excluding caps)
 * 7) half of cell length (excluding caps)
 * 8) timepoint at which the cell was formed
 * 9) cell growth rate
 * 10) cell's ambient viscosity with respect to surrounding fluid
 * 11) cell-surface friction coefficient
 * 12) cell-surface adhesion energy density
 *
 * Authors:
 *     Kee-Myoung Nam
 *
 * Last updated:
 *     2/4/2024
 */

#ifndef BIOFILM_SIMULATIONS_3D_HPP
#define BIOFILM_SIMULATIONS_3D_HPP

#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include <unordered_map>
#include <Eigen/Dense>
#include <boost/random.hpp>
#include <boost/multiprecision/mpfr.hpp>
#include "../include/growth.hpp"
#include "../include/mechanics.hpp"
#include "../include/utils.hpp"

using namespace Eigen;

// Expose math functions for both standard and boost MPFR types
using std::pow;
using boost::multiprecision::pow;
using std::sqrt;
using boost::multiprecision::sqrt;
using std::min;
using boost::multiprecision::min;
using std::max;
using boost::multiprecision::max;
using std::abs;
using boost::multiprecision::abs;

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
 * Run a simulation with the given initial population of cells.
 *
 * TODO Complete this docstring.
 *
 * @param cells_init
 * @param max_iter
 * @param n_cells
 * @param R
 * @param Rcell
 * @param L0
 * @param Ldiv
 * @param E0
 * @param Ecell
 * @param max_stepsize
 * @param iter_write
 * @param iter_update_neighbors
 * @param iter_update_stepsize
 * @param max_error_allowed
 * @param min_error
 * @param max_tries_update_stepsize
 * @param neighbor_threshold
 * @param nz_threshold
 * @param outprefix
 * @param rng_seed
 * @param growth_mean
 * @param growth_std
 * @param daughter_length_std
 * @param daughter_angle_xy_bound
 * @param daughter_angle_z_bound 
 */
template <typename T>
void runSimulation(const Ref<const Array<T, Dynamic, Dynamic> >& cells_init,
                   const int max_iter, const int n_cells, const T R, const T Rcell,
                   const T L0, const T Ldiv, const T E0, const T Ecell,
                   const T max_stepsize, const int iter_write,
                   const int iter_update_neighbors, const int iter_update_stepsize,
                   const T max_error_allowed, const T min_error,
                   const int max_tries_update_stepsize, const T neighbor_threshold,
                   const T nz_threshold, const std::string outprefix,
                   const int rng_seed, const double growth_mean,
                   const double growth_std, const double daughter_length_std,
                   const double daughter_angle_xy_bound,
                   const double daughter_angle_z_bound)
{
    Array<T, Dynamic, Dynamic> cells(cells_init);
    T t = 0;
    T dt = max_stepsize; 
    int iter = 0;
    int n = cells.rows();
    boost::random::mt19937 rng(rng_seed);

    // Define Butcher tableau for order 3(2) Runge-Kutta method by Bogacki
    // and Shampine 
    Array<T, Dynamic, Dynamic> A(4, 4); 
    A << 0,     0,     0,     0,
         1./2., 0,     0,     0,
         0,     3./4., 0,     0,
         2./9., 1./3., 4./9., 0;
    Array<T, Dynamic, 1> b(4);
    b << 2./9., 1./3., 4./9., 0;
    Array<T, Dynamic, 1> bs(4); 
    bs << 7./24., 1./4., 1./3., 1./8.;
    T error_order = 2; 

    // Prefactors for cell-cell interaction forces
    const T sqrtR = sqrt(R); 
    const T powRdiff = pow(R - Rcell, 1.5);
    Array<T, 4, 1> cell_cell_prefactors; 
    cell_cell_prefactors << 2.5 * sqrtR,
                            2.5 * E0 * sqrtR,
                            E0 * powRdiff,
                            Ecell;

    // Compute initial array of neighboring cells
    Array<T, Dynamic, 7> neighbors = getCellNeighbors<T>(cells, neighbor_threshold, R, Ldiv);

    // Initialize velocities to zero
    Array<T, Dynamic, 6> velocities = Array<T, Dynamic, 6>::Zero(n, 6);

    // Growth rate distribution function: normal distribution with given mean
    // and standard deviation
    boost::random::normal_distribution<> growth_dist(growth_mean, growth_std);
    std::function<T(boost::random::mt19937&)> growth_dist_func =
        [&growth_dist](boost::random::mt19937& rng)
        {
            return static_cast<T>(growth_dist(rng)); 
        };

    // Daughter cell length ratio distribution function: normal distribution
    // with mean 0.5 and given standard deviation
    boost::random::normal_distribution<> daughter_length_dist(0.5, daughter_length_std); 
    std::function<T(boost::random::mt19937&)> daughter_length_dist_func =
        [&daughter_length_dist](boost::random::mt19937& rng)
        {
            return static_cast<T>(daughter_length_dist(rng)); 
        };

    // Daughter angle distribution functions: two uniform distributions that 
    // are bounded by the given values 
    boost::random::uniform_01<> uniform_dist; 
    std::function<T(boost::random::mt19937&)> daughter_angle_xy_dist_func = 
        [&uniform_dist, &daughter_angle_xy_bound](boost::random::mt19937& rng)
        {
            return static_cast<T>(
                -daughter_angle_xy_bound + 2 * daughter_angle_xy_bound * uniform_dist(rng)
            );
        };
    std::function<T(boost::random::mt19937&)> daughter_angle_z_dist_func =
        [&uniform_dist, &daughter_angle_z_bound](boost::random::mt19937& rng)
        {
            return static_cast<T>(
                -daughter_angle_z_bound + 2 * daughter_angle_z_bound * uniform_dist(rng)
            );
        }; 

    // Write simulation parameters to a dictionary
    std::unordered_map<std::string, std::string> params;
    const int precision = 10;
    params["n_cells"] = std::to_string(n_cells);
    params["R"] = floatToString<T>(R, precision);
    params["Rcell"] = floatToString<T>(Rcell, precision);
    params["L0"] = floatToString<T>(L0, precision);
    params["Ldiv"] = floatToString<T>(Ldiv, precision);
    params["E0"] = floatToString<T>(E0, precision);
    params["Ecell"] = floatToString<T>(Ecell, precision);
    params["max_stepsize"] = floatToString<T>(max_stepsize, precision);
    params["iter_write"] = std::to_string(iter_write);
    params["iter_update_neighbors"] = std::to_string(iter_update_neighbors);
    params["iter_update_stepsize"] = std::to_string(iter_update_stepsize);
    params["max_error_allowed"] = floatToString<T>(max_error_allowed, precision);
    params["max_tries_update_stepsize"] = std::to_string(max_tries_update_stepsize);
    params["neighbor_threshold"] = floatToString<T>(neighbor_threshold, precision);
    params["nz_threshold"] = floatToString<T>(nz_threshold, precision);
    params["random_seed"] = std::to_string(rng_seed);
    params["growth_mean"] = floatToString<T>(growth_mean, precision); 
    params["growth_std"] = floatToString<T>(growth_std, precision);
    params["daughter_length_std"] = floatToString<T>(daughter_length_std, precision);
    params["daughter_angle_xy_bound"] = floatToString<T>(daughter_angle_xy_bound, precision);
    params["daughter_angle_z_bound"] = floatToString<T>(daughter_angle_z_bound, precision);

    // Write the initial population to file 
    params["t_curr"] = floatToString<T>(t);
    std::stringstream ss_init; 
    ss_init << outprefix << "_init.txt";
    std::string filename_init = ss_init.str(); 
    writeCells<T>(cells, params, filename_init); 
    
    // Define termination criterion, assuming that at least one of n_cells
    // or max_iter is positive
    std::function<bool(int, int)> terminate = [&n_cells, &max_iter](int n, int iter)
    {
        if (n_cells > 0 && max_iter > 0)
            return (n >= n_cells || iter >= max_iter);
        else if (n_cells > 0)    // Decide when to terminate only based on cell count
            return (n >= n_cells); 
        else if (max_iter > 0)   // Decide when to terminate only based on iteration count
            return (iter >= max_iter);
        else    // Otherwise, termination criteria are ill-defined, so always terminate
            return true;
    };

    // Run the simulation ...
    while (!terminate(n, iter))
    {
        // Divide the cells that have reached division length
        Array<int, Dynamic, 1> to_divide = divideMaxLength<T>(cells, Ldiv);
        if (to_divide.sum() > 0)
            std::cout << "... Dividing " << to_divide.sum() << " cells "
                      << "(iteration " << iter << ")" << std::endl;
        cells = divideCells<T>(
            cells, t, R, Rcell, to_divide, growth_dist_func, rng,
            daughter_length_dist_func, daughter_angle_xy_dist_func,
            daughter_angle_z_dist_func
        );

        // Update orientations and neighboring cells if division has occurred
        if (to_divide.sum() > 0)
        {
            normalizeOrientations<T>(cells);
            neighbors = getCellNeighbors<T>(cells, neighbor_threshold, R, Ldiv);
        }

        // Update cell positions and orientations 
        auto result = stepRungeKuttaAdaptiveFromNeighbors<T>(
            A, b, bs, cells, neighbors, dt, R, Rcell, cell_cell_prefactors,
            E0, nz_threshold
        ); 
        Array<T, Dynamic, Dynamic> cells_new = std::get<0>(result);
        Array<T, Dynamic, 6> errors = std::get<1>(result);
        Array<T, Dynamic, 6> velocities_new = std::get<2>(result);

        // If the error is big, retry the step with a smaller stepsize (up to
        // a given maximum number of attempts)
        if (iter % iter_update_stepsize == 0)
        {
            T max_error = max(errors.abs().maxCoeff(), min_error); 
            int j = 0; 
            while (max_error > max_error_allowed && j < max_tries_update_stepsize)
            {
                dt *= pow(max_error_allowed / max_error, 1.0 / (error_order + 1));
                result = stepRungeKuttaAdaptiveFromNeighbors<T>(
                    A, b, bs, cells, neighbors, dt, R, Rcell, cell_cell_prefactors,
                    E0, nz_threshold
                ); 
                cells_new = std::get<0>(result);
                errors = std::get<1>(result);
                velocities_new = std::get<2>(result);
                max_error = max(errors.abs().maxCoeff(), min_error);
                j++;  
            }
            // If the error is small, increase the stepsize up to a maximum stepsize
            if (max_error < max_error_allowed)
                dt = min(dt * pow(max_error_allowed / max_error, 1.0 / (error_order + 1)), max_stepsize);
        }
        cells = cells_new;
        velocities = velocities_new;

        // Grow the cells
        growCells<T>(cells, dt, R);

        // Pick out only the cells that overlap with the surface, updating 
        // array of neighboring cells whenever a cell is deleted 
        Array<T, Dynamic, 1> max_overlaps = R - cells.col(2) - cells.col(7) * cells.col(5);
        std::vector<int> overlap_idx; 
        for (int j = 0; j < cells.rows(); ++j)
        {
            if (max_overlaps(j) > -0.5 * R)    // Allow for a little room
                overlap_idx.push_back(j);
        }
        if (overlap_idx.size() < cells.rows())
        {
            cells = cells(overlap_idx, Eigen::all).eval();
            neighbors = getCellNeighbors<T>(cells, neighbor_threshold, R, Ldiv);
        }

        // Update distances between neighboring cells
        updateNeighborDistances<T>(cells, neighbors);

        // Update current time 
        t += dt;
        iter++;
        n = cells.rows(); 

        // Update neighboring cells 
        if (iter % iter_update_neighbors == 0)
            neighbors = getCellNeighbors<T>(cells, neighbor_threshold, R, Ldiv);
        
        // Write the current population to file
        if (iter % iter_write == 0)
        {
            std::cout << "Iteration " << iter << ": " << n << " cells, time = "
                      << t << ", max error = " << errors.abs().maxCoeff()
                      << ", dt = " << dt << std::endl;
            params["t_curr"] = floatToString<T>(t);
            std::stringstream ss; 
            ss << outprefix << "_iter" << iter << ".txt"; 
            std::string filename = ss.str(); 
            writeCells<T>(cells, params, filename); 
        } 
    }

    // Write final population to file
    params["t_curr"] = floatToString<T>(t);
    std::stringstream ss_final; 
    ss_final << outprefix << "_final.txt";
    std::string filename_final = ss_final.str(); 
    writeCells<T>(cells, params, filename_final);
}

#endif
