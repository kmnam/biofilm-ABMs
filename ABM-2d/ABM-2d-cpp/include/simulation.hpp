/**
 * Functions for running simulations with flexible initial conditions. 
 *
 * In what follows, a population of N cells is represented as a 2-D array of 
 * size (N, 11+), where each row represents a cell and stores the following data:
 * 
 * 0) x-coordinate of cell center
 * 1) y-coordinate of cell center
 * 2) x-coordinate of cell orientation vector
 * 3) y-coordinate of cell orientation vector
 * 4) cell length (excluding caps)
 * 5) half of cell length (excluding caps)
 * 6) timepoint at which the cell was formed
 * 7) cell growth rate
 * 8) cell's ambient viscosity with respect to surrounding fluid
 * 9) cell-surface friction coefficient
 * 10) cell group identity 
 *
 * Authors:
 *     Kee-Myoung Nam
 *
 * Last updated:
 *     7/30/2024
 */

#ifndef BIOFILM_SIMULATIONS_2D_HPP
#define BIOFILM_SIMULATIONS_2D_HPP

#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include <map>
#include <Eigen/Dense>
#include <boost/random.hpp>
#include <boost/multiprecision/mpfr.hpp>
#include "../include/growth.hpp"
#include "../include/mechanics.hpp"
#include "../include/utils.hpp"
#include "../include/switch.hpp"

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
using std::cos; 
using boost::multiprecision::cos;

/**
 * An enum that enumerates the different growth void types. 
 */
enum class GrowthVoidMode
{
    NONE = 0,
    FIXED = 1,
    RATIO = 2
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
 * TODO Update
 *
 * Run a simulation with the given initial population of cells.
 *
 * @param cells_init Initial population of cells. 
 * @param max_iter Maximum number of iterations. 
 * @param n_cells Maximum number of cells. 
 * @param R Cell radius (including the EPS). 
 * @param Rcell Cell radius (excluding the EPS).
 * @param L0 Initial cell length.
 * @param Ldiv Cell division length.
 * @param E0 Elastic modulus of EPS.
 * @param Ecell Elastic modulus of cell.
 * @param sigma0 Cell-surface adhesion energy density.
 * @param max_stepsize Maximum stepsize per iteration. 
 * @param write If true, write simulation output to file(s). 
 * @param outprefix Output filename prefix. 
 * @param iter_write Write cells to file every this many iterations. 
 * @param iter_update_neighbors Update neighboring cells every this many 
 *                              iterations. 
 * @param iter_update_boundary Update peripheral cells every this many 
 *                             iterations (only if `confine` is true).
 * @param iter_update_stepsize Update stepsize every this many iterations. 
 * @param max_error_allowed Maximum Runge-Kutta error allowed per iteration. 
 * @param min_error Minimum Runge-Kutta error. 
 * @param max_tries_update_stepsize Maximum number of tries to update stepsize
 *                                  due to Runge-Kutta error. 
 * @param neighbor_threshold Threshold for distinguishing between neighboring
 *                           and non-neighboring cells.
 * @param rng_seed Random number generator seed. 
 * @param growth_mean Mean growth rate. 
 * @param growth_std Standard deviation of growth rate. 
 * @param daughter_length_std Standard deviation of daughter length ratio 
 *                            distribution. 
 * @param daughter_angle_bound Bound on daughter cell re-orientation angle.
 * @param adhesion_mode Choice of potential used to model cell-cell adhesion.
 *                      Can be NONE (0), KIHARA (1), or GBK (2).
 * @param adhesion_params Parameters required to compute cell-cell adhesion
 *                        forces.
 * @param confine If true, introduce an additional radial confinement on the 
 *                peripheral cells.
 * @param confine_params Parameters required to compute radial confinement
 *                       forces.
 * @param growth_void_mode Choice of growth void to be introduced within the
 *                         biofilm. Can be NONE (0), FIXED (1), or RATIO (2).
 * @param growth_void_params Parameters required to introduce growth void 
 *                           within the biofilm. 
 * @returns Final population of cells.  
 */
template <typename T>
Array<T, Dynamic, Dynamic> runSimulation(const Ref<const Array<T, Dynamic, Dynamic> >& cells_init,
                                         const int max_iter, const int n_cells,
                                         const T R, const T Rcell, const T L0,
                                         const T Ldiv, const T E0, const T Ecell,
                                         const T sigma0, const T max_stepsize,
                                         const bool write,
                                         const std::string outprefix, 
                                         const int iter_write,
                                         const int iter_update_neighbors,
                                         const int iter_update_boundary,
                                         const int iter_update_stepsize,
                                         const T max_error_allowed,
                                         const T min_error,
                                         const int max_tries_update_stepsize,
                                         const T neighbor_threshold,
                                         const int rng_seed,
                                         const T growth_mean,
                                         const T growth_std,
                                         const T daughter_length_std,
                                         const T daughter_angle_bound,
                                         const AdhesionMode adhesion_mode,
                                         std::unordered_map<std::string, T>& adhesion_params,
                                         const bool confine,
                                         std::unordered_map<std::string, T>& confine_params,
                                         const GrowthVoidMode growth_void_mode,
                                         std::unordered_map<std::string, T>& growth_void_params)
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

    // Surface contact area density
    const T surface_contact_density = pow(sigma0 * R * R / (4 * E0), 1. / 3.);

    // Compute initial array of neighboring cells
    Array<T, Dynamic, 6> neighbors = getCellNeighbors<T>(cells, neighbor_threshold, R, Ldiv);

    // Allow adhesion between all pairs of neighboring cells in state 1 
    Array<int, Dynamic, 1> to_adhere = Array<int, Dynamic, 1>::Zero(neighbors.rows());
    for (int k = 0; k < neighbors.rows(); ++k)
    {
        int ni = neighbors(k, 0); 
        int nj = neighbors(k, 1);
        T dist = neighbors(k, Eigen::seq(2, 3)).matrix().norm(); 
        to_adhere(k) = (cells(ni, 10) == 1 && cells(nj, 10) == 1 && dist > R + Rcell && dist < 2 * R); 
    }

    // Initialize velocities to zero
    Array<T, Dynamic, 4> velocities = Array<T, Dynamic, 4>::Zero(n, 4);

    // Growth rate distribution function: normal distribution with given mean
    // and standard deviation
    boost::random::uniform_01<> uniform_dist; 
    std::function<T(boost::random::mt19937&)> growth_dist =
        [growth_mean, growth_std, &uniform_dist](boost::random::mt19937& rng)
        {
            return growth_mean + growth_std * standardNormal<T>(rng, uniform_dist);
        };

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

    // Daughter angle distribution function: uniform distribution that is
    // bounded by the given value 
    std::function<T(boost::random::mt19937&)> daughter_angle_dist = 
        [daughter_angle_bound, &uniform_dist](boost::random::mt19937& rng)
        {
            T r = static_cast<T>(uniform_dist(rng));
            return -daughter_angle_bound + 2 * daughter_angle_bound * r;
        };

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
    params["sigma0"] = floatToString<T>(sigma0, precision);
    params["max_stepsize"] = floatToString<T>(max_stepsize, precision);
    params["iter_write"] = std::to_string(iter_write);
    params["iter_update_neighbors"] = std::to_string(iter_update_neighbors);
    params["iter_update_boundary"] = std::to_string(iter_update_boundary);
    params["iter_update_stepsize"] = std::to_string(iter_update_stepsize);
    params["max_error_allowed"] = floatToString<T>(max_error_allowed, precision);
    params["max_tries_update_stepsize"] = std::to_string(max_tries_update_stepsize);
    params["neighbor_threshold"] = floatToString<T>(neighbor_threshold, precision);
    params["random_seed"] = std::to_string(rng_seed);
    params["growth_mean"] = floatToString<T>(growth_mean, precision); 
    params["growth_std"] = floatToString<T>(growth_std, precision);
    params["daughter_length_std"] = floatToString<T>(daughter_length_std, precision);
    params["daughter_angle_bound"] = floatToString<T>(daughter_angle_bound, precision);
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
    params["confine"] = (confine ? "1" : "0");
    if (confine)
    {
        for (auto&& item : confine_params)
        {
            std::stringstream ss;
            std::string key = item.first;
            T value = item.second;
            ss << "confine_" << key;
            if (key == "find_boundary")                // find_boundary is a boolean value 
                params[ss.str()] = (value == 0 ? "0" : "1");
            else if (key == "mincells_for_boundary")   // mincells_for_boundary is an integer value
                params[ss.str()] = std::to_string(static_cast<int>(value));
            else
                params[ss.str()] = floatToString<T>(value);
        }
    }
    params["growth_void_mode"] = std::to_string(static_cast<int>(growth_void_mode)); 
    if (growth_void_mode != GrowthVoidMode::NONE)
    {
        for (auto&& item : growth_void_params)
        {
            std::stringstream ss; 
            std::string key = item.first; 
            T value = item.second; 
            ss << "growth_void_" << key;
            if (key == "mincells")     // mincells is an integer value 
                params[ss.str()] = std::to_string(static_cast<int>(value));
            else 
                params[ss.str()] = floatToString<T>(value);
        }
    }

    // Get initial subset of peripheral cells (only if confinement is present)
    bool find_boundary = false; 
    int mincells_for_boundary = 0;
    std::vector<int> boundary_idx;
    if (confine)
    {
        find_boundary = (confine_params["find_boundary"] != 0);
        mincells_for_boundary = static_cast<int>(confine_params["mincells_for_boundary"]);
        if (find_boundary)
        {
            boundary_idx = getBoundary<T>(cells, R, mincells_for_boundary);
        }
        else
        {
            boundary_idx.resize(n); 
            std::iota(boundary_idx.begin(), boundary_idx.end(), 0);
        }
    }

    // Variable for keeping track of whether a growth void has been introduced
    bool growth_void_introduced = false;

    // Write the initial population to file
    if (write)
    {
        params["t_curr"] = floatToString<T>(t);
        std::stringstream ss_init; 
        ss_init << outprefix << "_init.txt";
        std::string filename_init = ss_init.str(); 
        if (confine)    // If confinement is present, write indicators for peripheral cells
        {
            Array<T, Dynamic, Dynamic> cells_ = Array<T, Dynamic, Dynamic>::Zero(n, cells.cols() + 1); 
            cells_(Eigen::all, Eigen::seq(0, cells.cols() - 1)) = cells;
            for (const int i : boundary_idx)
                cells_(i, cells.cols()) = 1;
            writeCells<T>(cells_, params, filename_init);
        }
        else 
        {
            writeCells<T>(cells, params, filename_init);
        }
    }
    
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
            cells, t, R, Rcell, to_divide, growth_dist, rng,
            daughter_length_dist, daughter_angle_dist
        );
        n = cells.rows(); 

        // Update neighboring cells and peripheral cells if division has occurred
        if (to_divide.sum() > 0)
        {
            neighbors = getCellNeighbors<T>(cells, neighbor_threshold, R, Ldiv);
            to_adhere.resize(neighbors.rows()); 
            for (int k = 0; k < neighbors.rows(); ++k)
            {
                int ni = neighbors(k, 0); 
                int nj = neighbors(k, 1);
                T dist = neighbors(k, Eigen::seq(2, 3)).matrix().norm(); 
                to_adhere(k) = (cells(ni, 10) == 1 && cells(nj, 10) == 1 && dist > R + Rcell && dist < 2 * R); 
            }
            if (confine)
            {
                if (find_boundary)
                {
                    boundary_idx = getBoundary<T>(cells, R, mincells_for_boundary);
                }
                else
                {
                    boundary_idx.resize(n); 
                    std::iota(boundary_idx.begin(), boundary_idx.end(), 0);
                }
            }
        }

        // Update cell positions and orientations 
        auto result = stepRungeKuttaAdaptive<T>(
            A, b, bs, cells, neighbors, to_adhere, dt, R, Rcell,
            cell_cell_prefactors, surface_contact_density, adhesion_mode, 
            adhesion_params, confine, boundary_idx, confine_params
        ); 
        Array<T, Dynamic, Dynamic> cells_new = std::get<0>(result);
        Array<T, Dynamic, 4> errors = std::get<1>(result);
        Array<T, Dynamic, 4> velocities_new = std::get<2>(result);

        // If the error is big, retry the step with a smaller stepsize (up to
        // a given maximum number of attempts)
        if (iter % iter_update_stepsize == 0)
        {
            // Enforce a composite error of the form e * (1 + y)
            Array<T, Dynamic, 4> scale = max_error_allowed * (
                Array<T, Dynamic, 4>::Ones(n, 4) + cells(Eigen::all, Eigen::seq(0, 3)).abs()
            );
            T error = max(sqrt((errors / scale).pow(2).sum() / (4 * n)), min_error); 

            // Ensure that the updated stepsize is between 0.2 times and 10 times
            // the previous stepsize 
            T factor = 0.9 * pow(1.0 / error, 1.0 / (error_order + 1));
            if (factor >= 10)
                factor = 10;
            else if (factor < 0.2)
                factor = 0.2;
            int j = 0;
            while (error > 1 && j < max_tries_update_stepsize && factor > 0.2 && factor < 10)
            {
                T dt_new = dt * factor; 
                result = stepRungeKuttaAdaptive<T>(
                    A, b, bs, cells, neighbors, to_adhere, dt_new, R, Rcell,
                    cell_cell_prefactors, surface_contact_density, adhesion_mode,
                    adhesion_params, confine, boundary_idx, confine_params
                ); 
                cells_new = std::get<0>(result);
                errors = std::get<1>(result);
                velocities_new = std::get<2>(result);
                error = max(sqrt((errors / scale).pow(2).sum() / (4 * n)), min_error);
                factor *= 0.9 * pow(1.0 / error, 1.0 / (error_order + 1));
                if (factor >= 10)
                    factor = 10;
                else if (factor < 0.2)
                    factor = 0.2;
                j++;  
            }
            
            // Ensure that the proposed stepsize is less than the maximum 
            if (dt * factor > max_stepsize)
                factor = max_stepsize / dt;

            // Re-do the integration with the new stepsize
            dt *= factor;
            result = stepRungeKuttaAdaptive<T>(
                A, b, bs, cells, neighbors, to_adhere, dt, R, Rcell,
                cell_cell_prefactors, surface_contact_density, adhesion_mode,
                adhesion_params, boundary_idx, confine, confine_params
            ); 
            cells_new = std::get<0>(result);
            errors = std::get<1>(result);
            velocities_new = std::get<2>(result);
        }
        // If desired, print a warning message if the error is big
        #ifdef DEBUG_WARN_LARGE_ERROR
            Array<T, Dynamic, 4> scale = max_error_allowed * (
                Array<T, Dynamic, 4>::Ones(n, 4) + cells(Eigen::all, Eigen::seq(0, 3)).abs()
            );
            T error = max(sqrt((errors / scale).pow(2).sum() / (4 * n)), min_error); 
            if (error > 5)
            {
                std::cout << "[WARN] Average error is > 5 times the desired error "
                          << "(absolute tol = relative tol = " << max_error_allowed
                          << ", iteration " << iter << ", time = " << t
                          << ", dt = " << dt << ")" << std::endl;
            }
        #endif
        cells = cells_new;
        velocities = velocities_new;

        // Grow the cells
        growCells<T>(cells, dt, R);

        // Update distances between neighboring cells
        updateNeighborDistances<T>(cells, neighbors);

        // Update current time 
        t += dt;
        iter++;

        // Update neighboring cells 
        if (iter % iter_update_neighbors == 0)
        {
            neighbors = getCellNeighbors<T>(cells, neighbor_threshold, R, Ldiv);
            to_adhere.resize(neighbors.rows()); 
            for (int k = 0; k < neighbors.rows(); ++k)
            {
                int ni = neighbors(k, 0); 
                int nj = neighbors(k, 1);
                T dist = neighbors(k, Eigen::seq(2, 3)).matrix().norm(); 
                to_adhere(k) = (cells(ni, 10) == 1 && cells(nj, 10) == 1 && dist > R + Rcell && dist < 2 * R); 
            }
        }

        // Update peripheral cells
        if (confine && iter % iter_update_boundary == 0)
        {
            if (find_boundary)
            {
                boundary_idx = getBoundary<T>(cells, R, mincells_for_boundary);
            }
            else
            {
                boundary_idx.resize(n); 
                std::iota(boundary_idx.begin(), boundary_idx.end(), 0);
            }
        }

        // Introduce or update growth void
        if ((growth_void_mode == GrowthVoidMode::FIXED && !growth_void_introduced) || growth_void_mode == GrowthVoidMode::RATIO)
        {
            // Have we reached the minimum number of cells? 
            if (cells.rows() >= growth_void_params["mincells"])
            {
                // Find the center of mass of the population  
                Array<T, 2, 1> center; 
                center << cells.col(0).mean(), cells.col(1).mean();

                // Find the radial distance of each cell to the center 
                Array<T, Dynamic, 1> rdists = (cells(Eigen::all, Eigen::seq(0, 1)).rowwise() - center.transpose()).matrix().rowwise().norm().array();

                // Normalize by the maximum radial distance 
                T radius = rdists.maxCoeff();
                rdists /= radius;

                // Identify the innermost fraction of cells whose growth is to
                // be arrested
                for (int i = 0; i < cells.rows(); ++i)
                {
                    if (rdists(i) < growth_void_params["radial_fraction"])
                        cells(i, 7) = 0.0;
                }
                
                // We have now introduced the growth void
                growth_void_introduced = true;
            }
        }
        
        // Write the current population to file
        if (write && (iter % iter_write == 0))
        {
            std::cout << "Iteration " << iter << ": " << n << " cells, time = "
                      << t << ", max error = " << errors.abs().maxCoeff()
                      << ", avg error = " << errors.abs().sum() / (4 * n)
                      << ", dt = " << dt << std::endl;
            params["t_curr"] = floatToString<T>(t);
            std::stringstream ss; 
            ss << outprefix << "_iter" << iter << ".txt"; 
            std::string filename = ss.str(); 
            if (confine)    // If confinement is present, write indicators for peripheral cells
            {
                Array<T, Dynamic, Dynamic> cells_ = Array<T, Dynamic, Dynamic>::Zero(n, cells.cols() + 1); 
                cells_(Eigen::all, Eigen::seq(0, cells.cols() - 1)) = cells;
                for (const int i : boundary_idx)
                    cells_(i, cells.cols()) = 1;
                writeCells<T>(cells_, params, filename);
            }
            else 
            {
                writeCells<T>(cells, params, filename);
            }
        }
    }

    // Write final population to file
    if (write)
    {
        params["t_curr"] = floatToString<T>(t);
        std::stringstream ss_final; 
        ss_final << outprefix << "_final.txt";
        std::string filename_final = ss_final.str(); 
        if (confine)    // If confinement is present, write indicators for peripheral cells
        {
            Array<T, Dynamic, Dynamic> cells_ = Array<T, Dynamic, Dynamic>::Zero(n, cells.cols() + 1); 
            cells_(Eigen::all, Eigen::seq(0, cells.cols() - 1)) = cells;
            for (const int i : boundary_idx)
                cells_(i, cells.cols()) = 1;
            writeCells<T>(cells_, params, filename_final);
        }
        else 
        {
            writeCells<T>(cells, params, filename_final);
        }
    }

    return cells;
}

/**
 * TODO Update
 *
 * Run a simulation with the given initial population of cells.
 *
 * This function runs simulations in which the cells switch between two 
 * groups that differ by growth rate and one or more additional physical
 * attributes. The growth rate and chosen attributes are taken to be normally
 * distributed variables that exhibit a specified mean and standard deviation.
 *
 * @param cells_init Initial population of cells. 
 * @param max_iter Maximum number of iterations. 
 * @param n_cells Maximum number of cells. 
 * @param R Cell radius (including the EPS). 
 * @param Rcell Cell radius (excluding the EPS).
 * @param L0 Initial cell length.
 * @param Ldiv Cell division length.
 * @param E0 Elastic modulus of EPS.
 * @param Ecell Elastic modulus of cell.
 * @param sigma0 Cell-surface adhesion energy density.
 * @param max_stepsize Maximum stepsize per iteration. 
 * @param write If true, write simulation output to file(s). 
 * @param outprefix Output filename prefix. 
 * @param iter_write Write cells to file every this many iterations. 
 * @param iter_update_neighbors Update neighboring cells every this many 
 *                              iterations.
 * @param iter_update_boundary Update peripheral cells every this many 
 *                             iterations (only if `confine` is true).
 * @param iter_update_stepsize Update stepsize every this many iterations. 
 * @param max_error_allowed Maximum Runge-Kutta error allowed per iteration. 
 * @param min_error Minimum Runge-Kutta error. 
 * @param max_tries_update_stepsize Maximum number of tries to update stepsize
 *                                  due to Runge-Kutta error. 
 * @param neighbor_threshold Threshold for distinguishing between neighboring
 *                           and non-neighboring cells.
 * @param rng_seed Random number generator seed. 
 * @param n_groups Number of groups.
 * @param switch_attributes Indices of attributes to change when switching
 *                          groups.
 * @param growth_means Mean growth rate for cells in each group.
 * @param growth_stds Standard deviation of growth rate for cells in each
 *                    group.
 * @param attribute_means Array of mean attribute values for cells in each
 *                        group.
 * @param attribute_stds Array of standard deviations of attributes for cells
 *                       in each group.
 * @param switch_rates Array of between-group switching rates. 
 * @param daughter_length_std Standard deviation of daughter length ratio 
 *                            distribution. 
 * @param daughter_angle_bound Bound on daughter cell re-orientation angle.
 * @param adhesion_mode Choice of potential used to model cell-cell adhesion.
 *                      Can be NONE (0), KIHARA (1), or GBK (2).
 * @param adhesion_params Parameters required to compute cell-cell adhesion
 *                        forces.
 * @param confine If true, introduce an additional radial confinement on the 
 *                peripheral cells.
 * @param confine_params Parameters required to compute radial confinement
 *                       forces.
 * @param growth_void_mode Choice of growth void to be introduced within the
 *                         biofilm. Can be NONE (0), FIXED (1), or RATIO (2).
 * @param growth_void_params Parameters required to introduce growth void 
 *                           within the biofilm. 
 * @returns Final population of cells.  
 */
template <typename T>
Array<T, Dynamic, Dynamic> runSimulation(const Ref<const Array<T, Dynamic, Dynamic> >& cells_init,
                                         const int max_iter, const int n_cells,
                                         const T R, const T Rcell, const T L0,
                                         const T Ldiv, const T E0, const T Ecell,
                                         const T sigma0, const T max_stepsize,
                                         const bool write,
                                         const std::string outprefix, 
                                         const int iter_write,
                                         const int iter_update_neighbors,
                                         const int iter_update_boundary,
                                         const int iter_update_stepsize,
                                         const T max_error_allowed,
                                         const T min_error,
                                         const int max_tries_update_stepsize,
                                         const T neighbor_threshold,
                                         const int rng_seed,
                                         const int n_groups,
                                         std::vector<int>& switch_attributes,
                                         const Ref<const Array<T, Dynamic, 1> >& growth_means,
                                         const Ref<const Array<T, Dynamic, 1> >& growth_stds,
                                         const Ref<const Array<T, Dynamic, Dynamic> >& attribute_means,
                                         const Ref<const Array<T, Dynamic, Dynamic> >& attribute_stds,
                                         const Ref<const Array<T, Dynamic, Dynamic> >& switch_rates,
                                         const T daughter_length_std,
                                         const T daughter_angle_bound, 
                                         const AdhesionMode adhesion_mode, 
                                         std::unordered_map<std::string, T>& adhesion_params,
                                         const bool confine,
                                         std::unordered_map<std::string, T>& confine_params,
                                         const GrowthVoidMode growth_void_mode,
                                         std::unordered_map<std::string, T>& growth_void_params)
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

    // Surface contact area density
    const T surface_contact_density = pow(sigma0 * R * R / (4 * E0), 1. / 3.);

    // Compute initial array of neighboring cells
    Array<T, Dynamic, 6> neighbors = getCellNeighbors<T>(cells, neighbor_threshold, R, Ldiv);

    // Allow adhesion between all pairs of neighboring cells in state 1 
    Array<int, Dynamic, 1> to_adhere = Array<int, Dynamic, 1>::Zero(neighbors.rows());
    for (int k = 0; k < neighbors.rows(); ++k)
    {
        int ni = neighbors(k, 0); 
        int nj = neighbors(k, 1);
        T dist = neighbors(k, Eigen::seq(2, 3)).matrix().norm(); 
        to_adhere(k) = (cells(ni, 10) == 1 && cells(nj, 10) == 1 && dist > R + Rcell && dist < 2 * R); 
    }

    // Initialize velocities to zero
    Array<T, Dynamic, 4> velocities = Array<T, Dynamic, 4>::Zero(n, 4);

    // Growth rate distribution functions: normal distributions with given means
    // and standard deviations
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

    // Attribute distribution functions: normal distributions with given means
    // and standard deviations
    const int n_attributes = switch_attributes.size();
    std::map<std::pair<int, int>, std::function<T(boost::random::mt19937&)> > attribute_dists;
    for (int i = 0; i < n_groups; ++i)
    {
        for (int j = 0; j < n_attributes; ++j)
        {
            T attribute_mean = attribute_means(i, j); 
            T attribute_std = attribute_stds(i, j);
            auto pair = std::make_pair(i, j);
            std::function<T(boost::random::mt19937&)> attribute_dist =
                [attribute_mean, attribute_std, &uniform_dist](boost::random::mt19937& rng)
                {
                    return attribute_mean + attribute_std * standardNormal<T>(rng, uniform_dist);
                };
            attribute_dists[pair] = attribute_dist;
        }
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

    // Daughter angle distribution function: uniform distribution that is
    // bounded by the given value 
    std::function<T(boost::random::mt19937&)> daughter_angle_dist = 
        [daughter_angle_bound, &uniform_dist](boost::random::mt19937& rng)
        {
            T r = static_cast<T>(uniform_dist(rng));
            return -daughter_angle_bound + 2 * daughter_angle_bound * r;
        };

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
    params["sigma0"] = floatToString<T>(sigma0, precision);
    params["max_stepsize"] = floatToString<T>(max_stepsize, precision);
    params["iter_write"] = std::to_string(iter_write);
    params["iter_update_neighbors"] = std::to_string(iter_update_neighbors);
    params["iter_update_boundary"] = std::to_string(iter_update_boundary);
    params["iter_update_stepsize"] = std::to_string(iter_update_stepsize);
    params["max_error_allowed"] = floatToString<T>(max_error_allowed, precision);
    params["max_tries_update_stepsize"] = std::to_string(max_tries_update_stepsize);
    params["neighbor_threshold"] = floatToString<T>(neighbor_threshold, precision);
    params["random_seed"] = std::to_string(rng_seed);
    params["n_groups"] = std::to_string(n_groups);
    for (int i = 0; i < n_attributes; ++i)
    {
        std::stringstream ss; 
        ss << "switch_attribute" << i + 1;
        params[ss.str()] = std::to_string(switch_attributes[i]);
    }
    for (int i = 0; i < n_groups; ++i)
    {
        std::stringstream ss; 
        ss << "growth_mean" << i + 1;
        params[ss.str()] = floatToString<double>(growth_means(i), precision);
        ss.str(std::string());
        ss << "growth_std" << i + 1; 
        params[ss.str()] = floatToString<double>(growth_stds(i), precision);
        ss.str(std::string());
        for (int j = i + 1; j < n_groups; ++j)
        {
            ss << "switch_rate_" << i + 1 << "_" << j + 1;
            params[ss.str()] = floatToString<T>(switch_rates(i, j), precision);
            ss.str(std::string());
            ss << "switch_rate_" << j + 1 << "_" << i + 1;
            params[ss.str()] = floatToString<T>(switch_rates(j, i), precision);
            ss.str(std::string()); 
        }
        for (int j = 0; j < n_attributes; ++j)
        {
            ss << "attribute_mean_" << i + 1 << "_" << j + 1;
            params[ss.str()] = floatToString<T>(attribute_means(i, j), precision);
            ss.str(std::string());
            ss << "attribute_std_" << i + 1 << "_" << j + 1;
            params[ss.str()] = floatToString<T>(attribute_stds(i, j), precision);
            ss.str(std::string());
        }
    }
    params["daughter_length_std"] = floatToString<T>(daughter_length_std, precision);
    params["daughter_angle_bound"] = floatToString<T>(daughter_angle_bound, precision);
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
    params["confine"] = (confine ? "1" : "0");
    if (confine)
    {
        for (auto&& item : confine_params)
        {
            std::stringstream ss;
            std::string key = item.first;
            T value = item.second;
            ss << "confine_" << key; 
            if (key == "find_boundary")                // find_boundary is a boolean value 
                params[ss.str()] = (value == 0 ? "0" : "1");
            else if (key == "mincells_for_boundary")   // mincells_for_boundary is an integer value
                params[ss.str()] = std::to_string(static_cast<int>(value));
            else
                params[ss.str()] = floatToString<T>(value);
        }
    }
    params["growth_void_mode"] = std::to_string(static_cast<int>(growth_void_mode)); 
    if (growth_void_mode != GrowthVoidMode::NONE)
    {
        for (auto&& item : growth_void_params)
        {
            std::stringstream ss; 
            std::string key = item.first; 
            T value = item.second; 
            ss << "growth_void_" << key;
            if (key == "mincells")     // mincells is an integer value 
                params[ss.str()] = std::to_string(static_cast<int>(value));
            else 
                params[ss.str()] = floatToString<T>(value);
        }
    }

    // Get initial subset of peripheral cells (only if confinement is present)
    bool find_boundary = false; 
    int mincells_for_boundary = 0;
    std::vector<int> boundary_idx;
    if (confine)
    {
        find_boundary = (confine_params["find_boundary"] != 0);
        mincells_for_boundary = static_cast<int>(confine_params["mincells_for_boundary"]);
        if (find_boundary)
        {
            boundary_idx = getBoundary<T>(cells, R, mincells_for_boundary);
        }
        else
        {
            boundary_idx.resize(n); 
            std::iota(boundary_idx.begin(), boundary_idx.end(), 0);
        }
    }

    // Variable for keeping track of whether a growth void has been introduced
    bool growth_void_introduced = false;

    // Write the initial population to file
    if (write)
    {
        params["t_curr"] = floatToString<T>(t);
        std::stringstream ss_init; 
        ss_init << outprefix << "_init.txt";
        std::string filename_init = ss_init.str(); 
        if (confine)    // If confinement is on, write indicators for peripheral cells
        {
            Array<T, Dynamic, Dynamic> cells_ = Array<T, Dynamic, Dynamic>::Zero(n, cells.cols() + 1); 
            cells_(Eigen::all, Eigen::seq(0, cells.cols() - 1)) = cells;
            for (const int i : boundary_idx)
                cells_(i, cells.cols()) = 1;
            writeCells<T>(cells_, params, filename_init);
        }
        else 
        {
            writeCells<T>(cells, params, filename_init);
        }
    }
    
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
            cells, t, R, Rcell, to_divide, growth_dists, rng,
            daughter_length_dist, daughter_angle_dist
        );
        n = cells.rows();

        // Update neighboring cells and peripheral cells if division has occurred
        if (to_divide.sum() > 0)
        {
            neighbors = getCellNeighbors<T>(cells, neighbor_threshold, R, Ldiv);
            to_adhere.resize(neighbors.rows());
            for (int k = 0; k < neighbors.rows(); ++k)
            {
                int ni = neighbors(k, 0); 
                int nj = neighbors(k, 1);
                T dist = neighbors(k, Eigen::seq(2, 3)).matrix().norm(); 
                to_adhere(k) = (cells(ni, 10) == 1 && cells(nj, 10) == 1 && dist > R + Rcell && dist < 2 * R); 
            }

            if (confine)
            {
                if (find_boundary)
                {
                    boundary_idx = getBoundary<T>(cells, R, mincells_for_boundary);
                }
                else
                {
                    boundary_idx.resize(n); 
                    std::iota(boundary_idx.begin(), boundary_idx.end(), 0);
                }
            }
        }

        // Update cell positions and orientations
        auto result = stepRungeKuttaAdaptive<T>(
            A, b, bs, cells, neighbors, to_adhere, dt, R, Rcell,
            cell_cell_prefactors, surface_contact_density, adhesion_mode,
            adhesion_params, confine, boundary_idx, confine_params
        ); 
        Array<T, Dynamic, Dynamic> cells_new = std::get<0>(result);
        Array<T, Dynamic, 4> errors = std::get<1>(result);
        Array<T, Dynamic, 4> velocities_new = std::get<2>(result);

        // If the error is big, retry the step with a smaller stepsize (up to
        // a given maximum number of attempts)
        if (iter % iter_update_stepsize == 0)
        {
            // Enforce a composite error of the form e * (1 + y)
            Array<T, Dynamic, 4> scale = max_error_allowed * (
                Array<T, Dynamic, 4>::Ones(n, 4) + cells(Eigen::all, Eigen::seq(0, 3)).abs()
            );
            T error = max(sqrt((errors / scale).pow(2).sum() / (4 * n)), min_error); 

            // Ensure that the updated stepsize is between 0.2 times and 10 times
            // the previous stepsize 
            T factor = 0.9 * pow(1.0 / error, 1.0 / (error_order + 1));
            if (factor >= 10)
                factor = 10;
            else if (factor < 0.2)
                factor = 0.2;
            int j = 0;
            while (error > 1 && j < max_tries_update_stepsize && factor > 0.2 && factor < 10)
            {
                T dt_new = dt * factor; 
                result = stepRungeKuttaAdaptive<T>(
                    A, b, bs, cells, neighbors, to_adhere, dt_new, R, Rcell,
                    cell_cell_prefactors, surface_contact_density, adhesion_mode,
                    adhesion_params, confine, boundary_idx, confine_params
                ); 
                cells_new = std::get<0>(result);
                errors = std::get<1>(result);
                velocities_new = std::get<2>(result);
                error = max(sqrt((errors / scale).pow(2).sum() / (4 * n)), min_error);
                factor *= 0.9 * pow(1.0 / error, 1.0 / (error_order + 1));
                if (factor >= 10)
                    factor = 10;
                else if (factor < 0.2)
                    factor = 0.2;
                j++;  
            }
            
            // Ensure that the proposed stepsize is less than the maximum 
            if (dt * factor > max_stepsize)
                factor = max_stepsize / dt;

            // Re-do the integration with the new stepsize
            dt *= factor;
            result = stepRungeKuttaAdaptive<T>(
                A, b, bs, cells, neighbors, to_adhere, dt, R, Rcell,
                cell_cell_prefactors, surface_contact_density, adhesion_mode,
                adhesion_params, confine, boundary_idx, confine_params
            ); 
            cells_new = std::get<0>(result);
            errors = std::get<1>(result);
            velocities_new = std::get<2>(result);
        }
        #ifdef DEBUG_WARN_LARGE_ERROR
            Array<T, Dynamic, 4> scale = max_error_allowed * (
                Array<T, Dynamic, 4>::Ones(n, 4) + cells(Eigen::all, Eigen::seq(0, 3)).abs()
            );
            T error = max(sqrt((errors / scale).pow(2).sum() / (4 * n)), min_error); 
            if (error > 5)
            {
                std::cout << "[WARN] Average error is > 5 times the desired error "
                          << "(absolute tol = relative tol = " << max_error_allowed
                          << ", iteration " << iter << ", time = " << t
                          << ", dt = " << dt << ")" << std::endl;
            }
        #endif
        cells = cells_new;
        velocities = velocities_new;

        // Grow the cells
        growCells<T>(cells, dt, R);

        // Update distances between neighboring cells
        updateNeighborDistances<T>(cells, neighbors);

        // Update current time 
        t += dt;
        iter++;

        // Update neighboring cells 
        if (iter % iter_update_neighbors == 0)
        {
            neighbors = getCellNeighbors<T>(cells, neighbor_threshold, R, Ldiv);
            to_adhere.resize(neighbors.rows());
            for (int k = 0; k < neighbors.rows(); ++k)
            {
                int ni = neighbors(k, 0); 
                int nj = neighbors(k, 1);
                T dist = neighbors(k, Eigen::seq(2, 3)).matrix().norm(); 
                to_adhere(k) = (cells(ni, 10) == 1 && cells(nj, 10) == 1 && dist > R + Rcell && dist < 2 * R); 
            }
        }

        // Update peripheral cells
        if (confine && iter % iter_update_boundary == 0)
        {
            if (find_boundary)
            {
                boundary_idx = getBoundary<T>(cells, R, mincells_for_boundary);
            }
            else
            {
                boundary_idx.resize(n); 
                std::iota(boundary_idx.begin(), boundary_idx.end(), 0);
            }
        }

        // Switch cells between groups
        switchGroups<T>(
            cells, switch_attributes, n_groups, dt, switch_rates, growth_dists,
            attribute_dists, rng, uniform_dist
        );
        for (int k = 0; k < neighbors.rows(); ++k)
        {
            int ni = neighbors(k, 0); 
            int nj = neighbors(k, 1); 
            T dist = neighbors(k, Eigen::seq(2, 3)).matrix().norm(); 
            to_adhere(k) = (cells(ni, 10) == 1 && cells(nj, 10) == 1 && dist > R + Rcell && dist < 2 * R); 
        }

        // Introduce or update growth void
        if ((growth_void_mode == GrowthVoidMode::FIXED && !growth_void_introduced) || growth_void_mode == GrowthVoidMode::RATIO)
        {
            // Have we reached the minimum number of cells? 
            if (cells.rows() >= growth_void_params["mincells"])
            {
                // Find the center of mass of the population  
                Array<T, 2, 1> center; 
                center << cells.col(0).mean(), cells.col(1).mean();

                // Find the radial distance of each cell to the center 
                Array<T, Dynamic, 1> rdists = (cells(Eigen::all, Eigen::seq(0, 1)).rowwise() - center.transpose()).matrix().rowwise().norm().array();

                // Normalize by the maximum radial distance 
                T radius = rdists.maxCoeff();
                rdists /= radius;

                // Identify the innermost fraction of cells whose growth is to
                // be arrested
                for (int i = 0; i < cells.rows(); ++i)
                {
                    if (rdists(i) < growth_void_params["radial_fraction"])
                        cells(i, 7) = 0.0;
                }
                
                // We have now introduced the growth void
                growth_void_introduced = true;
            }
        }
        
        // Write the current population to file
        if (write && (iter % iter_write == 0))
        {
            std::cout << "Iteration " << iter << ": " << n << " cells, time = "
                      << t << ", max error = " << errors.abs().maxCoeff()
                      << ", avg error = " << errors.abs().sum() / (4 * n)
                      << ", dt = " << dt << std::endl;
            params["t_curr"] = floatToString<T>(t);
            std::stringstream ss; 
            ss << outprefix << "_iter" << iter << ".txt"; 
            std::string filename = ss.str();
            if (confine)    // If confinement is on, write indicators for peripheral cells
            {
                Array<T, Dynamic, Dynamic> cells_ = Array<T, Dynamic, Dynamic>::Zero(n, cells.cols() + 1);
                cells_(Eigen::all, Eigen::seq(0, cells.cols() - 1)) = cells;
                for (const int i : boundary_idx)
                    cells_(i, cells.cols()) = 1;
                writeCells<T>(cells_, params, filename);
            }
            else 
            {
                writeCells<T>(cells, params, filename);
            }
        }
    }

    // Write final population to file
    if (write)
    {
        params["t_curr"] = floatToString<T>(t);
        std::stringstream ss_final; 
        ss_final << outprefix << "_final.txt";
        std::string filename_final = ss_final.str(); 
        if (confine)    // If confinement is on, write indicators for peripheral cells
        {
            Array<T, Dynamic, Dynamic> cells_ = Array<T, Dynamic, Dynamic>::Zero(n, cells.cols() + 1);
            cells_(Eigen::all, Eigen::seq(0, cells.cols() - 1)) = cells;
            for (const int i : boundary_idx)
                cells_(i, cells.cols()) = 1;
            writeCells<T>(cells_, params, filename_final);
        }
        else 
        {
            writeCells<T>(cells, params, filename_final);
        }
    }

    return cells;
}

/**
 * TODO Update
 *
 * Run a simulation with the given initial population of cells.
 *
 * This function runs simulations in which the cells switch between two 
 * groups that differ by growth rate and one or more additional physical
 * attributes, *according to the copy-number of a plasmid*, which is stored
 * in an additional (12th) column in the population data. Switching from
 * group 1 to group 2 occurs only when the plasmid is lost due to asymmetric 
 * partitioning at cell division. 
 * 
 * The growth rate and chosen attributes are taken to be normally distributed
 * variables that exhibit a specified mean and standard deviation.
 *
 * The log-ratio of plasmid copy-numbers in the daughter cells from each
 * division event is taken from a normal distribution centered at 0 with a
 * given standard deviation. 
 *
 * @param cells_init Initial population of cells. 
 * @param max_iter Maximum number of iterations. 
 * @param n_cells Maximum number of cells. 
 * @param R Cell radius (including the EPS). 
 * @param Rcell Cell radius (excluding the EPS).
 * @param L0 Initial cell length.
 * @param Ldiv Cell division length.
 * @param E0 Elastic modulus of EPS.
 * @param Ecell Elastic modulus of cell.
 * @param sigma0 Cell-surface adhesion energy density.
 * @param max_stepsize Maximum stepsize per iteration. 
 * @param write If true, write simulation output to file(s). 
 * @param outprefix Output filename prefix. 
 * @param iter_write Write cells to file every this many iterations. 
 * @param iter_update_neighbors Update neighboring cells every this many 
 *                              iterations.
 * @param iter_update_boundary Update peripheral cells every this many 
 *                             iterations (only if `confine` is true).
 * @param iter_update_stepsize Update stepsize every this many iterations. 
 * @param max_error_allowed Maximum Runge-Kutta error allowed per iteration. 
 * @param min_error Minimum Runge-Kutta error. 
 * @param max_tries_update_stepsize Maximum number of tries to update stepsize
 *                                  due to Runge-Kutta error. 
 * @param neighbor_threshold Threshold for distinguishing between neighboring
 *                           and non-neighboring cells.
 * @param rng_seed Random number generator seed.
 * @param group_default Identifier for the group comprised of cells that 
 *                      contain the plasmid. Should be 1 or 2. 
 * @param switch_attributes Indices of attributes to change when switching
 *                          groups.
 * @param growth_means Mean growth rate for cells in each group.
 * @param growth_stds Standard deviation of growth rate for cells in each
 *                    group.
 * @param attribute_means Array of mean attribute values for cells in each
 *                        group.
 * @param attribute_stds Array of standard deviations of attributes for cells
 *                       in each group.
 * @param partition_logratio_std Standard deviation for the normal distribution
 *                               that determines the log-ratio of plasmid 
 *                               copy-numbers in the daughter cells after
 *                               each division event. 
 * @param daughter_length_std Standard deviation of daughter length ratio 
 *                            distribution. 
 * @param daughter_angle_bound Bound on daughter cell re-orientation angle.
 * @param adhesion_mode Choice of potential used to model cell-cell adhesion.
 *                      Can be NONE (0), KIHARA (1), or GBK (2).
 * @param adhesion_params Parameters required to compute cell-cell adhesion
 *                        forces.
 * @param confine If true, introduce an additional radial confinement on the 
 *                peripheral cells.
 * @param confine_params Parameters required to compute radial confinement
 *                       forces.
 * @param growth_void_mode Choice of growth void to be introduced within the
 *                         biofilm. Can be NONE (0), FIXED (1), or RATIO (2).
 * @param growth_void_params Parameters required to introduce growth void 
 *                           within the biofilm. 
 * @returns Final population of cells.  
 */
template <typename T>
Array<T, Dynamic, Dynamic> runSimulationWithPlasmid(const Ref<const Array<T, Dynamic, Dynamic> >& cells_init,
                                                    const int max_iter,
                                                    const int n_cells,
                                                    const T R,
                                                    const T Rcell,
                                                    const T L0,
                                                    const T Ldiv,
                                                    const T E0,
                                                    const T Ecell,
                                                    const T sigma0,
                                                    const T max_stepsize,
                                                    const bool write,
                                                    const std::string outprefix, 
                                                    const int iter_write,
                                                    const int iter_update_neighbors,
                                                    const int iter_update_boundary,
                                                    const int iter_update_stepsize,
                                                    const T max_error_allowed,
                                                    const T min_error,
                                                    const int max_tries_update_stepsize,
                                                    const T neighbor_threshold,
                                                    const int rng_seed,
                                                    const int group_default,
                                                    std::vector<int>& switch_attributes,
                                                    const Ref<const Array<T, 2, 1> >& growth_means,
                                                    const Ref<const Array<T, 2, 1> >& growth_stds,
                                                    const Ref<const Array<T, 2, Dynamic> >& attribute_means,
                                                    const Ref<const Array<T, 2, Dynamic> >& attribute_stds,
                                                    const T partition_logratio_std,
                                                    const T daughter_length_std,
                                                    const T daughter_angle_bound, 
                                                    const AdhesionMode adhesion_mode, 
                                                    std::unordered_map<std::string, T>& adhesion_params,
                                                    const bool confine,
                                                    std::unordered_map<std::string, T>& confine_params,
                                                    const GrowthVoidMode growth_void_mode,
                                                    std::unordered_map<std::string, T>& growth_void_params)
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

    // Surface contact area density
    const T surface_contact_density = pow(sigma0 * R * R / (4 * E0), 1. / 3.);

    // Compute initial array of neighboring cells
    Array<T, Dynamic, 6> neighbors = getCellNeighbors<T>(cells, neighbor_threshold, R, Ldiv);

    // Allow adhesion between all pairs of neighboring cells in state 1 
    Array<int, Dynamic, 1> to_adhere = Array<int, Dynamic, 1>::Zero(neighbors.rows());
    for (int k = 0; k < neighbors.rows(); ++k)
    {
        int ni = neighbors(k, 0); 
        int nj = neighbors(k, 1);
        T dist = neighbors(k, Eigen::seq(2, 3)).matrix().norm(); 
        to_adhere(k) = (cells(ni, 10) == 1 && cells(nj, 10) == 1 && dist > R + Rcell && dist < 2 * R); 
    }

    // Initialize velocities to zero
    Array<T, Dynamic, 4> velocities = Array<T, Dynamic, 4>::Zero(n, 4);

    // Growth rate distribution functions: normal distributions with given means
    // and standard deviations
    boost::random::uniform_01<> uniform_dist; 
    std::vector<std::function<T(boost::random::mt19937&)> > growth_dists; 
    for (int i = 0; i < 2; ++i)    // Assume there are 2 groups
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

    // Attribute distribution functions: normal distributions with given means
    // and standard deviations
    const int n_attributes = switch_attributes.size();
    std::map<std::pair<int, int>, std::function<T(boost::random::mt19937&)> > attribute_dists;
    for (int i = 0; i < 2; ++i)    // Assume there are 2 groups
    {
        for (int j = 0; j < n_attributes; ++j)
        {
            T attribute_mean = attribute_means(i, j); 
            T attribute_std = attribute_stds(i, j);
            auto pair = std::make_pair(i, j);
            std::function<T(boost::random::mt19937&)> attribute_dist =
                [attribute_mean, attribute_std, &uniform_dist](boost::random::mt19937& rng)
                {
                    return attribute_mean + attribute_std * standardNormal<T>(rng, uniform_dist);
                };
            attribute_dists[pair] = attribute_dist;
        }
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

    // Daughter angle distribution function: uniform distribution that is
    // bounded by the given value 
    std::function<T(boost::random::mt19937&)> daughter_angle_dist = 
        [daughter_angle_bound, &uniform_dist](boost::random::mt19937& rng)
        {
            T r = static_cast<T>(uniform_dist(rng));
            return -daughter_angle_bound + 2 * daughter_angle_bound * r;
        };

    // Plasmid copy-number log-ratio distribution function: normal distribution
    // with mean zero and given standard deviation
    std::function<T(boost::random::mt19937&)> partition_logratio_dist = 
        [partition_logratio_std, &uniform_dist](boost::random::mt19937& rng)
        {
            return partition_logratio_std * standardNormal<T>(rng, uniform_dist);
        };

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
    params["sigma0"] = floatToString<T>(sigma0, precision);
    params["max_stepsize"] = floatToString<T>(max_stepsize, precision);
    params["iter_write"] = std::to_string(iter_write);
    params["iter_update_neighbors"] = std::to_string(iter_update_neighbors);
    params["iter_update_boundary"] = std::to_string(iter_update_boundary);
    params["iter_update_stepsize"] = std::to_string(iter_update_stepsize);
    params["max_error_allowed"] = floatToString<T>(max_error_allowed, precision);
    params["max_tries_update_stepsize"] = std::to_string(max_tries_update_stepsize);
    params["neighbor_threshold"] = floatToString<T>(neighbor_threshold, precision);
    params["random_seed"] = std::to_string(rng_seed);
    params["group_default"] = std::to_string(group_default); 
    for (int i = 0; i < n_attributes; ++i)
    {
        std::stringstream ss; 
        ss << "switch_attribute" << i + 1;
        params[ss.str()] = std::to_string(switch_attributes[i]);
    }
    for (int i = 0; i < 2; ++i)    // Assume there are 2 groups
    {
        std::stringstream ss; 
        ss << "growth_mean" << i + 1;
        params[ss.str()] = floatToString<double>(growth_means(i), precision);
        ss.str(std::string());
        ss << "growth_std" << i + 1; 
        params[ss.str()] = floatToString<double>(growth_stds(i), precision);
        ss.str(std::string());
        for (int j = 0; j < n_attributes; ++j)
        {
            ss << "attribute_mean_" << i + 1 << "_" << j + 1;
            params[ss.str()] = floatToString<T>(attribute_means(i, j), precision);
            ss.str(std::string());
            ss << "attribute_std_" << i + 1 << "_" << j + 1;
            params[ss.str()] = floatToString<T>(attribute_stds(i, j), precision);
            ss.str(std::string());
        }
    }
    params["partition_logratio_std"] = floatToString<T>(partition_logratio_std, precision); 
    params["daughter_length_std"] = floatToString<T>(daughter_length_std, precision);
    params["daughter_angle_bound"] = floatToString<T>(daughter_angle_bound, precision);
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
    params["confine"] = (confine ? "1" : "0");
    if (confine)
    {
        for (auto&& item : confine_params)
        {
            std::stringstream ss;
            std::string key = item.first;
            T value = item.second;
            ss << "confine_" << key; 
            if (key == "find_boundary")                // find_boundary is a boolean value 
                params[ss.str()] = (value == 0 ? "0" : "1");
            else if (key == "mincells_for_boundary")   // mincells_for_boundary is an integer value
                params[ss.str()] = std::to_string(static_cast<int>(value));
            else
                params[ss.str()] = floatToString<T>(value);
        }
    }
    params["growth_void_mode"] = std::to_string(static_cast<int>(growth_void_mode)); 
    if (growth_void_mode != GrowthVoidMode::NONE)
    {
        for (auto&& item : growth_void_params)
        {
            std::stringstream ss; 
            std::string key = item.first; 
            T value = item.second; 
            ss << "growth_void_" << key;
            if (key == "mincells")     // mincells is an integer value 
                params[ss.str()] = std::to_string(static_cast<int>(value));
            else 
                params[ss.str()] = floatToString<T>(value);
        }
    }

    // Get initial subset of peripheral cells (only if confinement is present)
    bool find_boundary = false; 
    int mincells_for_boundary = 0;
    std::vector<int> boundary_idx;
    if (confine)
    {
        find_boundary = (confine_params["find_boundary"] != 0);
        mincells_for_boundary = static_cast<int>(confine_params["mincells_for_boundary"]);
        if (find_boundary)
        {
            boundary_idx = getBoundary<T>(cells, R, mincells_for_boundary);
        }
        else
        {
            boundary_idx.resize(n); 
            std::iota(boundary_idx.begin(), boundary_idx.end(), 0);
        }
    }

    // Variable for keeping track of whether a growth void has been introduced
    bool growth_void_introduced = false;

    // Write the initial population to file
    if (write)
    {
        params["t_curr"] = floatToString<T>(t);
        std::stringstream ss_init; 
        ss_init << outprefix << "_init.txt";
        std::string filename_init = ss_init.str(); 
        if (confine)    // If confinement is on, write indicators for peripheral cells
        {
            Array<T, Dynamic, Dynamic> cells_ = Array<T, Dynamic, Dynamic>::Zero(n, cells.cols() + 1); 
            cells_(Eigen::all, Eigen::seq(0, cells.cols() - 1)) = cells;
            for (const int i : boundary_idx)
                cells_(i, cells.cols()) = 1;
            writeCells<T>(cells_, params, filename_init);
        }
        else 
        {
            writeCells<T>(cells, params, filename_init);
        }
    }
    
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
        cells = divideCellsWithPlasmid<T>(
            cells, t, R, Rcell, to_divide, growth_dists, rng,
            daughter_length_dist, daughter_angle_dist, partition_logratio_dist
        );
        n = cells.rows(); 
        
        // Switch groups for all cells that have lost the plasmid 
        for (int i = 0; i < n; ++i)
        {
            if (cells(i, 11) == 0)
            {
                // Sample the cell's new growth rate and attribute values 
                int group = (group_default == 1 ? 2 : 1);
                int j = group - 1; 
                cells(i, 10) = group;
                T growth_rate = growth_dists[j](rng);
                cells(i, 7) = growth_rate; 
                for (int k = 0; k < n_attributes; ++k)
                {
                    auto pair = std::make_pair(j, k); 
                    T attribute = attribute_dists[pair](rng); 
                    cells(i, switch_attributes[k]) = attribute; 
                }
            }
        }

        // Update neighboring cells and peripheral cells if division has occurred
        if (to_divide.sum() > 0)
        {
            neighbors = getCellNeighbors<T>(cells, neighbor_threshold, R, Ldiv);
            to_adhere.resize(neighbors.rows());
            for (int k = 0; k < neighbors.rows(); ++k)
            {
                int ni = neighbors(k, 0); 
                int nj = neighbors(k, 1);
                T dist = neighbors(k, Eigen::seq(2, 3)).matrix().norm(); 
                to_adhere(k) = (cells(ni, 10) == 1 && cells(nj, 10) == 1 && dist > R + Rcell && dist < 2 * R); 
            }

            if (confine)
            {
                if (find_boundary)
                {
                    boundary_idx = getBoundary<T>(cells, R, mincells_for_boundary);
                }
                else
                {
                    boundary_idx.resize(n); 
                    std::iota(boundary_idx.begin(), boundary_idx.end(), 0);
                }
            }
        }

        // Update cell positions and orientations
        auto result = stepRungeKuttaAdaptive<T>(
            A, b, bs, cells, neighbors, to_adhere, dt, R, Rcell,
            cell_cell_prefactors, surface_contact_density, adhesion_mode,
            adhesion_params, confine, boundary_idx, confine_params
        ); 
        Array<T, Dynamic, Dynamic> cells_new = std::get<0>(result);
        Array<T, Dynamic, 4> errors = std::get<1>(result);
        Array<T, Dynamic, 4> velocities_new = std::get<2>(result);

        // If the error is big, retry the step with a smaller stepsize (up to
        // a given maximum number of attempts)
        if (iter % iter_update_stepsize == 0)
        {
            // Enforce a composite error of the form e * (1 + y)
            Array<T, Dynamic, 4> scale = max_error_allowed * (
                Array<T, Dynamic, 4>::Ones(n, 4) + cells(Eigen::all, Eigen::seq(0, 3)).abs()
            );
            T error = max(sqrt((errors / scale).pow(2).sum() / (4 * n)), min_error); 

            // Ensure that the updated stepsize is between 0.2 times and 10 times
            // the previous stepsize 
            T factor = 0.9 * pow(1.0 / error, 1.0 / (error_order + 1));
            if (factor >= 10)
                factor = 10;
            else if (factor < 0.2)
                factor = 0.2;
            int j = 0;
            while (error > 1 && j < max_tries_update_stepsize && factor > 0.2 && factor < 10)
            {
                T dt_new = dt * factor; 
                result = stepRungeKuttaAdaptive<T>(
                    A, b, bs, cells, neighbors, to_adhere, dt_new, R, Rcell,
                    cell_cell_prefactors, surface_contact_density, adhesion_mode,
                    adhesion_params, confine, boundary_idx, confine_params
                ); 
                cells_new = std::get<0>(result);
                errors = std::get<1>(result);
                velocities_new = std::get<2>(result);
                error = max(sqrt((errors / scale).pow(2).sum() / (4 * n)), min_error);
                factor *= 0.9 * pow(1.0 / error, 1.0 / (error_order + 1));
                if (factor >= 10)
                    factor = 10;
                else if (factor < 0.2)
                    factor = 0.2;
                j++;  
            }
            
            // Ensure that the proposed stepsize is less than the maximum 
            if (dt * factor > max_stepsize)
                factor = max_stepsize / dt;

            // Re-do the integration with the new stepsize
            dt *= factor;
            result = stepRungeKuttaAdaptive<T>(
                A, b, bs, cells, neighbors, to_adhere, dt, R, Rcell,
                cell_cell_prefactors, surface_contact_density, adhesion_mode,
                adhesion_params, confine, boundary_idx, confine_params
            ); 
            cells_new = std::get<0>(result);
            errors = std::get<1>(result);
            velocities_new = std::get<2>(result);
        }
        #ifdef DEBUG_WARN_LARGE_ERROR
            Array<T, Dynamic, 4> scale = max_error_allowed * (
                Array<T, Dynamic, 4>::Ones(n, 4) + cells(Eigen::all, Eigen::seq(0, 3)).abs()
            );
            T error = max(sqrt((errors / scale).pow(2).sum() / (4 * n)), min_error); 
            if (error > 5)
            {
                std::cout << "[WARN] Average error is > 5 times the desired error "
                          << "(absolute tol = relative tol = " << max_error_allowed
                          << ", iteration " << iter << ", time = " << t
                          << ", dt = " << dt << ")" << std::endl;
            }
        #endif
        cells = cells_new;
        velocities = velocities_new;

        // Grow the cells
        growCells<T>(cells, dt, R);

        // Update distances between neighboring cells
        updateNeighborDistances<T>(cells, neighbors);

        // Update current time 
        t += dt;
        iter++;

        // Update neighboring cells 
        if (iter % iter_update_neighbors == 0)
        {
            neighbors = getCellNeighbors<T>(cells, neighbor_threshold, R, Ldiv);
            to_adhere.resize(neighbors.rows());
            for (int k = 0; k < neighbors.rows(); ++k)
            {
                int ni = neighbors(k, 0); 
                int nj = neighbors(k, 1);
                T dist = neighbors(k, Eigen::seq(2, 3)).matrix().norm(); 
                to_adhere(k) = (cells(ni, 10) == 1 && cells(nj, 10) == 1 && dist > R + Rcell && dist < 2 * R); 
            }
        }

        // Update peripheral cells
        if (confine && iter % iter_update_boundary == 0)
        {
            if (find_boundary)
            {
                boundary_idx = getBoundary<T>(cells, R, mincells_for_boundary);
            }
            else
            {
                boundary_idx.resize(n); 
                std::iota(boundary_idx.begin(), boundary_idx.end(), 0);
            }
        }

        // Introduce or update growth void
        if ((growth_void_mode == GrowthVoidMode::FIXED && !growth_void_introduced) || growth_void_mode == GrowthVoidMode::RATIO)
        {
            // Have we reached the minimum number of cells? 
            if (cells.rows() >= growth_void_params["mincells"])
            {
                // Find the center of mass of the population  
                Array<T, 2, 1> center; 
                center << cells.col(0).mean(), cells.col(1).mean();

                // Find the radial distance of each cell to the center 
                Array<T, Dynamic, 1> rdists = (cells(Eigen::all, Eigen::seq(0, 1)).rowwise() - center.transpose()).matrix().rowwise().norm().array();

                // Normalize by the maximum radial distance 
                T radius = rdists.maxCoeff();
                rdists /= radius;

                // Identify the innermost fraction of cells whose growth is to
                // be arrested
                for (int i = 0; i < cells.rows(); ++i)
                {
                    if (rdists(i) < growth_void_params["radial_fraction"])
                        cells(i, 7) = 0.0;
                }
                
                // We have now introduced the growth void
                growth_void_introduced = true;
            }
        }
        
        // Write the current population to file
        if (write && (iter % iter_write == 0))
        {
            std::cout << "Iteration " << iter << ": " << n << " cells, time = "
                      << t << ", max error = " << errors.abs().maxCoeff()
                      << ", avg error = " << errors.abs().sum() / (4 * n)
                      << ", dt = " << dt << std::endl;
            params["t_curr"] = floatToString<T>(t);
            std::stringstream ss; 
            ss << outprefix << "_iter" << iter << ".txt"; 
            std::string filename = ss.str();
            if (confine)    // If confinement is on, write indicators for peripheral cells
            {
                Array<T, Dynamic, Dynamic> cells_ = Array<T, Dynamic, Dynamic>::Zero(n, cells.cols() + 1);
                cells_(Eigen::all, Eigen::seq(0, cells.cols() - 1)) = cells;
                for (const int i : boundary_idx)
                    cells_(i, cells.cols()) = 1;
                writeCells<T>(cells_, params, filename);
            }
            else 
            {
                writeCells<T>(cells, params, filename);
            }
        }
    }

    // Write final population to file
    if (write)
    {
        params["t_curr"] = floatToString<T>(t);
        std::stringstream ss_final; 
        ss_final << outprefix << "_final.txt";
        std::string filename_final = ss_final.str(); 
        if (confine)    // If confinement is on, write indicators for peripheral cells
        {
            Array<T, Dynamic, Dynamic> cells_ = Array<T, Dynamic, Dynamic>::Zero(n, cells.cols() + 1);
            cells_(Eigen::all, Eigen::seq(0, cells.cols() - 1)) = cells;
            for (const int i : boundary_idx)
                cells_(i, cells.cols()) = 1;
            writeCells<T>(cells_, params, filename_final);
        }
        else 
        {
            writeCells<T>(cells, params, filename_final);
        }
    }

    return cells;
}

#endif
