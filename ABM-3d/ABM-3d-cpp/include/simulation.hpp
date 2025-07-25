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
 *     4/23/2025
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
 * @param attribute_means Array of mean attribute values for cells in each
 *                        group.
 * @param attribute_stds Array of standard deviations of attributes for cells
 *                       in each group.
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
 * @param truncate_surface_friction If true, truncate cell-surface friction
 *                                  coefficients according to Coulomb's law
 *                                  of friction.
 * @param surface_coulomb_coeff Friction coefficient that relates the velocity
 *                              of each cell to the normal force due to cell-
 *                              surface repulsion. 
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
 * @param adhesion_map Set of pairs of group IDs (1, 2, ...) for which cells 
 *                     can adhere to each other. 
 * @param adhesion_params Parameters required to compute cell-cell adhesion
 *                        forces.
 * @param no_surface If true, omit the surface from the simulations. 
 * @param n_cells_start_switch Number of cells at which to begin switching.
 *                             All switching is suppressed until this number
 *                             of cells is reached. 
 * @param track_poles If true, keep track of pole birth times.
 * @returns Final population of cells.  
 */
template <typename T>
std::pair<Array<T, Dynamic, Dynamic>, std::vector<int> >
    runSimulationAdaptiveLagrangian(const Ref<const Array<T, Dynamic, Dynamic> >& cells_init,
                                    std::vector<int>& parents_init,
                                    const int max_iter,
                                    const int n_cells,
                                    const T R,
                                    const T Rcell,
                                    const T L0,
                                    const T Ldiv,
                                    const T E0,
                                    const T Ecell,
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
                                    const Ref<const Array<T, Dynamic, Dynamic> >& attribute_means,
                                    const Ref<const Array<T, Dynamic, Dynamic> >& attribute_stds,
                                    const SwitchMode switch_mode,
                                    const Ref<const Array<T, Dynamic, Dynamic> >& switch_rates,
                                    const T daughter_length_std,
                                    const T daughter_angle_xy_bound,
                                    const T daughter_angle_z_bound,
                                    const bool truncate_surface_friction,
                                    const T surface_coulomb_coeff,
                                    const T max_rxy_noise,
                                    const T max_rz_noise,
                                    const T max_nxy_noise,
                                    const T max_nz_noise, 
                                    const bool basal_only,
                                    const T basal_min_overlap, 
                                    const AdhesionMode adhesion_mode, 
                                    std::unordered_set<std::pair<int, int>, boost::hash<std::pair<int, int> > >& adhesion_map, 
                                    std::unordered_map<std::string, T>& adhesion_params,
                                    const bool no_surface = false,
                                    const int n_cells_start_switch = 0,
                                    const bool track_poles = false,
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
    Array<T, 3, 1> cell_cell_prefactors; 
    cell_cell_prefactors << 2.5 * E0 * sqrt(R),
                            2.5 * E0 * sqrt(R) * pow(2 * (R - Rcell), 2.5),
                            2.5 * Ecell * sqrt(Rcell);

    // Compute initial array of neighboring cells
    Array<T, Dynamic, 7> neighbors = getCellNeighbors<T>(cells, neighbor_threshold, R, Ldiv);

    // Allow adhesion between all pairs of neighboring cells according to the
    // given adhesion map 
    Array<int, Dynamic, 1> to_adhere = Array<int, Dynamic, 1>::Zero(neighbors.rows());
    for (int k = 0; k < neighbors.rows(); ++k)
    {
        int ni = neighbors(k, 0); 
        int nj = neighbors(k, 1);
        int gi = cells(ni, __colidx_group); 
        int gj = cells(nj, __colidx_group);
        std::pair<int, int> pair; 
        if (gi < gj)
            pair = std::make_pair(gi, gj); 
        else 
            pair = std::make_pair(gj, gi); 
        T dist = neighbors(k, Eigen::seq(2, 4)).matrix().norm(); 
        to_adhere(k) = (
            adhesion_map.find(pair) != adhesion_map.end() &&
            dist > R + Rcell && dist < 2 * R
        ); 
    }

    // Initialize parent IDs (TODO check that they have been correctly specified)
    std::vector<int> parents(parents_init); 

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
    const int n_attributes = group_attributes.size();
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

    // Check that the cell data array has the correct size if pole ages are 
    // to be tracked 
    if (track_poles && cells.cols() < __ncols_required + 2)
        throw std::runtime_error("Insufficient number of columns for tracking pole ages");  

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
            ss << "attribute_mean_" << i + 1 << "_" << j + 1;
            params[ss.str()] = floatToString<T>(attribute_means(i, j), precision);
            ss.str(std::string());
            ss << "attribute_std_" << i + 1 << "_" << j + 1;
            params[ss.str()] = floatToString<T>(attribute_stds(i, j), precision);
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
    params["daughter_length_std"] = floatToString<T>(daughter_length_std, precision);
    params["daughter_angle_xy_bound"] = floatToString<T>(daughter_angle_xy_bound, precision);
    params["daughter_angle_z_bound"] = floatToString<T>(daughter_angle_z_bound, precision);
    params["truncate_surface_friction"] = (truncate_surface_friction ? "1" : "0"); 
    params["surface_coulomb_coeff"] = floatToString<T>(surface_coulomb_coeff, precision); 
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
    params["track_poles"] = (track_poles ? "1" : "0");
    params["no_surface"] = (no_surface ? "1" : "0");
    params["n_cells_start_switch"] = std::to_string(n_cells_start_switch);  

    // Write the initial population to file
    std::unordered_map<int, int> write_other_cols;
    if (track_poles)
    {
        write_other_cols[__colidx_negpole_t0] = 1;    // Write pole ages as floats
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
        std::vector<std::pair<int, int> > daughter_pairs; 
        if (to_divide.sum() > 0)
            std::cout << "... Dividing " << to_divide.sum() << " cells "
                      << "(iteration " << iter << ")" << std::endl;
        if (track_poles)    // Track poles if desired 
        {
            auto div_result = divideCellsWithPoles<T>(
                cells, parents, t, R, Rcell, to_divide, growth_dists, rng,
                daughter_length_dist, daughter_angle_xy_dist, daughter_angle_z_dist
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
                    cells, daughter_pairs, group_attributes, n_groups,
                    switch_rates, growth_dists, attribute_dists, rng, uniform_dist
                );
            }
            // Update neighboring cells 
            neighbors = getCellNeighbors<T>(cells, neighbor_threshold, R, Ldiv);
            // Update pairs of adhering cells 
            to_adhere.resize(neighbors.rows());
            for (int k = 0; k < neighbors.rows(); ++k)
            {
                int ni = neighbors(k, 0); 
                int nj = neighbors(k, 1);
                int gi = cells(ni, __colidx_group); 
                int gj = cells(nj, __colidx_group);
                std::pair<int, int> pair; 
                if (gi < gj)
                    pair = std::make_pair(gi, gj); 
                else 
                    pair = std::make_pair(gj, gi); 
                T dist = neighbors(k, Eigen::seq(2, 4)).matrix().norm(); 
                to_adhere(k) = (
                    adhesion_map.find(pair) != adhesion_map.end() &&
                    dist > R + Rcell && dist < 2 * R
                ); 
            }
        }

        // Update cell positions and orientations
        #ifdef _OPENMP
            if (n >= n_start_multithread && !started_multithread)
            {
                std::cout << "[NOTE] Started multithreading: detected "
                          << omp_get_max_threads() << " threads" << std::endl;
                started_multithread = true; 
            }
        #endif
        auto result = stepRungeKuttaAdaptive<T>(
            A, b, bs, cells, neighbors, to_adhere, dt, iter, R, Rcell,
            cell_cell_prefactors, E0, nz_threshold, max_rxy_noise,
            max_rz_noise, max_nxy_noise, max_nz_noise, rng, uniform_dist,
            adhesion_mode, adhesion_params, no_surface, n_start_multithread
        ); 
        Array<T, Dynamic, Dynamic> cells_new = result.first;
        Array<T, Dynamic, 6> errors = result.second;

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
                    A, b, bs, cells, neighbors, to_adhere, dt_new, iter, R, Rcell,
                    cell_cell_prefactors, E0, nz_threshold, max_rxy_noise,
                    max_rz_noise, max_nxy_noise, max_nz_noise, rng, uniform_dist,
                    adhesion_mode, adhesion_params, no_surface
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
                A, b, bs, cells, neighbors, to_adhere, dt, iter, R, Rcell,
                cell_cell_prefactors, E0, nz_threshold, max_rxy_noise,
                max_rz_noise, max_nxy_noise, max_nz_noise, rng, uniform_dist,
                adhesion_mode, adhesion_params, no_surface
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
        cells = cells_new;

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
        {
            neighbors = getCellNeighbors<T>(cells, neighbor_threshold, R, Ldiv);
            // Update pairs of adhering cells 
            to_adhere.resize(neighbors.rows());
            for (int k = 0; k < neighbors.rows(); ++k)
            {
                int ni = neighbors(k, 0); 
                int nj = neighbors(k, 1);
                int gi = cells(ni, __colidx_group); 
                int gj = cells(nj, __colidx_group);
                std::pair<int, int> pair; 
                if (gi < gj)
                    pair = std::make_pair(gi, gj); 
                else 
                    pair = std::make_pair(gj, gi); 
                T dist = neighbors(k, Eigen::seq(2, 4)).matrix().norm(); 
                to_adhere(k) = (
                    adhesion_map.find(pair) != adhesion_map.end() &&
                    dist > R + Rcell && dist < 2 * R
                ); 
            }
        }

        // Switch cells between groups if desired 
        if (n >= n_cells_start_switch && switch_mode == SwitchMode::MARKOV)
        {
            // First switch the cells 
            switchGroupsMarkov<T>(
                cells, group_attributes, n_groups, dt, switch_rates, growth_dists,
                attribute_dists, rng, uniform_dist
            );
            // Update pairs of adhering cells 
            for (int k = 0; k < neighbors.rows(); ++k)
            {
                int ni = neighbors(k, 0); 
                int nj = neighbors(k, 1);
                int gi = cells(ni, __colidx_group); 
                int gj = cells(nj, __colidx_group);
                std::pair<int, int> pair; 
                if (gi < gj)
                    pair = std::make_pair(gi, gj); 
                else 
                    pair = std::make_pair(gj, gi); 
                T dist = neighbors(k, Eigen::seq(2, 4)).matrix().norm(); 
                to_adhere(k) = (
                    adhesion_map.find(pair) != adhesion_map.end() &&
                    dist > R + Rcell && dist < 2 * R
                ); 
            }
            // Truncate cell-surface friction coefficients according to Coulomb's law
            if (truncate_surface_friction)
            {
                // TODO Implement this
                throw std::runtime_error("Not implemented");
                /*
                truncateSurfaceFrictionCoeffsCoulomb<T>(
                    cells, R, E0, surface_contact_density, surface_coulomb_coeff
                );
                */
            }
            else    // Otherwise, ensure that friction coefficients are correct after switching 
            {
                for (int i = 0; i < n; ++i)
                    cells(i, __colidx_eta1) = cells(i, __colidx_maxeta1);
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
            std::cout << "Iteration " << iter << ": " << n << " cells, time = "
                      << t << ", time elapsed = " << elapsed.count() << " sec"
                      << ", max error = " << errors.abs().maxCoeff()
                      << ", avg error = " << errors.abs().sum() / (6 * n)
                      << ", dt = " << dt << std::endl;
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
