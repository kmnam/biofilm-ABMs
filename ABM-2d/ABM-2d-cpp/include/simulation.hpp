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
 *     3/5/2025
 */

#ifndef BIOFILM_SIMULATIONS_2D_HPP
#define BIOFILM_SIMULATIONS_2D_HPP

#include <iostream>
#include <fstream>
#include <cmath>
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
#include "confinement.hpp"
#include "growthVoid.hpp"

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
using std::log10; 
using boost::multiprecision::log10; 

/**
 * An enum that enumerates the different growth void types.
 */
enum class GrowthVoidMode
{
    NONE = 0,
    FIXED_CORE = 1,
    FRACTIONAL_ANNULUS = 2
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
 * Run a simulation with the given initial population of cells.
 *
 * This function runs simulations in which the cells switch between multiple
 * groups that differ by growth rate and one or more additional physical
 * attributes. The growth rate and chosen attributes are taken to be normally
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
 * @param sigma0 Cell-surface adhesion energy density.
 * @param max_stepsize Maximum stepsize per iteration.
 * @param min_stepsize Minimum stepsize per iteration. 
 * @param write If true, write simulation output to file(s). 
 * @param outprefix Output filename prefix.
 * @param dt_write Write cells to file during each iteration in which the time
 *                 has passed a multiple of this value. 
 * @param iter_update_neighbors Update neighboring cells every this many 
 *                              iterations.
 * @param iter_update_boundary Update peripheral cells every this many 
 *                             iterations (only if `confine_mode` is not NONE).
 * @param iter_update_stepsize Update stepsize every this many iterations. 
 * @param max_error_allowed Upper bound on maximum Runge-Kutta error allowed
 *                          per iteration.
 * @param min_error Minimum Runge-Kutta error. 
 * @param max_tries_update_stepsize Maximum number of tries to update stepsize
 *                                  due to Runge-Kutta error. 
 * @param neighbor_threshold Threshold for distinguishing between neighboring
 *                           and non-neighboring cells.
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
 * @param daughter_angle_bound Bound on daughter cell re-orientation angle.
 * @param truncate_surface_friction If true, truncate cell-surface friction
 *                                  coefficients according to Coulomb's law
 *                                  of friction.
 * @param surface_coulomb_coeff Friction coefficient that relates the velocity
 *                              of each cell to the normal force due to cell-
 *                              surface repulsion. 
 * @param max_noise Maximum noise to be added to each generalized force used 
 *                  to compute the velocities.
 * @param adhesion_mode Choice of potential used to model cell-cell adhesion.
 *                      Can be NONE (0), KIHARA (1), or GBK (2).
 * @param adhesion_map Set of pairs of group IDs (1, 2, ...) for which cells 
 *                     can adhere to each other. 
 * @param adhesion_params Parameters required to compute cell-cell adhesion
 *                        forces.
 * @param confine_mode Confinement mode. Can be NONE (0), RADIAL (1), or 
 *                     CHANNEL (2). 
 * @param confine_params Parameters required to compute confinement forces. 
 * @param growth_void_mode Choice of growth void to be introduced within the
 *                         biofilm. Can be NONE (0), FIXED_CORE (1), or
 *                         FRACTIONAL_ANNULUS (2).
 * @param growth_void_params Parameters required to introduce growth void 
 *                           within the biofilm.
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
                                    const T sigma0,
                                    const T max_stepsize,
                                    const T min_stepsize,
                                    const bool write,
                                    const std::string outprefix,
                                    const T dt_write,
                                    const int iter_update_neighbors,
                                    const int iter_update_boundary,
                                    const int iter_update_stepsize,
                                    const T max_error_allowed,
                                    const T min_error,
                                    const int max_tries_update_stepsize,
                                    const T neighbor_threshold,
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
                                    const T daughter_angle_bound,
                                    const bool truncate_surface_friction,
                                    const T surface_coulomb_coeff,
                                    const T max_noise,
                                    const AdhesionMode adhesion_mode, 
                                    std::unordered_set<std::pair<int, int>, boost::hash<std::pair<int, int> > >& adhesion_map, 
                                    std::unordered_map<std::string, T>& adhesion_params,
                                    const ConfinementMode confine_mode,
                                    std::unordered_map<std::string, T>& confine_params,
                                    const GrowthVoidMode growth_void_mode,
                                    std::unordered_map<std::string, T>& growth_void_params,
                                    const bool track_poles = false)
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
    Array<T, 3, 1> cell_cell_prefactors; 
    cell_cell_prefactors << 2.5 * E0 * sqrt(R),
                            2.5 * E0 * sqrt(R) * pow(2 * (R - Rcell), 2.5),
                            2.5 * Ecell * sqrt(Rcell);

    // Surface contact area density
    const T surface_contact_density = pow(sigma0 * R * R / (4 * E0), 1. / 3.);

    // Compute initial array of neighboring cells
    Array<T, Dynamic, 6> neighbors = getCellNeighbors<T>(cells, neighbor_threshold, R, Ldiv);

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
        T dist = neighbors(k, Eigen::seq(2, 3)).matrix().norm(); 
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

    // Daughter angle distribution function: uniform distribution that is
    // bounded by the given value 
    std::function<T(boost::random::mt19937&)> daughter_angle_dist = 
        [daughter_angle_bound, &uniform_dist](boost::random::mt19937& rng)
        {
            T r = static_cast<T>(uniform_dist(rng));
            return -daughter_angle_bound + 2 * daughter_angle_bound * r;
        };

    // Peripheral cells are required when either imposing confinement or
    // imposing a growth void 
    //
    // If both are present, then the minimum number of cells for the boundary
    // calculation is set to the *minimum* of confine_params["mincells_for_boundary"]
    // and growth_void_params["mincells"]
    const bool find_boundary = (
        confine_mode != ConfinementMode::NONE || growth_void_mode != GrowthVoidMode::NONE
    );
    int mincells_for_boundary = 0; 
    if (confine_mode != ConfinementMode::NONE && growth_void_mode != GrowthVoidMode::NONE) 
        mincells_for_boundary = min(
            static_cast<int>(confine_params["mincells_for_boundary"]),
            static_cast<int>(growth_void_params["mincells"])
        ); 
    else if (confine_mode != ConfinementMode::NONE)
        mincells_for_boundary = static_cast<int>(confine_params["mincells_for_boundary"]);
    else if (growth_void_mode != GrowthVoidMode::NONE) 
        mincells_for_boundary = static_cast<int>(growth_void_params["mincells"]);

    // Check that the cell data array has the correct size if peripheral cells
    // or pole ages are to be tracked
    if (find_boundary && cells.cols() < __ncols_required + 1)
        throw std::runtime_error("Insufficient number of columns for tracking peripheral cells");  
    if (track_poles && cells.cols() < __ncols_required + 3)
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
    params["sigma0"] = floatToString<T>(sigma0, precision);
    params["max_stepsize"] = floatToString<T>(max_stepsize, precision);
    params["min_stepsize"] = floatToString<T>(min_stepsize, precision); 
    params["dt_write"] = floatToString<T>(dt_write, precision); 
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
    params["daughter_angle_bound"] = floatToString<T>(daughter_angle_bound, precision);
    params["truncate_surface_friction"] = (truncate_surface_friction ? "1" : "0"); 
    params["surface_coulomb_coeff"] = floatToString<T>(surface_coulomb_coeff, precision); 
    params["max_noise"] = floatToString<T>(max_noise, precision); 
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
    params["confine_mode"] = std::to_string(static_cast<int>(confine_mode)); 
    if (confine_mode != ConfinementMode::NONE)
    {
        for (auto&& item : confine_params)
        {
            std::stringstream ss;
            std::string key = item.first;
            T value = item.second;
            ss << "confine_" << key; 
            if (key == "mincells_for_boundary")   // mincells_for_boundary is an integer value
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
            if (key == "mincells")                // mincells is an integer value 
                params[ss.str()] = std::to_string(static_cast<int>(value));
            else 
                params[ss.str()] = floatToString<T>(value);
        }
    }
    params["track_poles"] = (track_poles ? "1" : "0"); 

    // Get initial subset of peripheral cells (only if confinement or a growth
    // void is present)
    std::vector<int> boundary_idx;
    if (find_boundary)
    {
        boundary_idx = getBoundary<T>(cells, R, mincells_for_boundary);
        for (int i = 0; i < n; ++i)
            cells(i, __colidx_boundary) = 0;
        for (const int i : boundary_idx)
            cells(i, __colidx_boundary) = 1;
    }

    // Determine an initial growth void
    //
    // First define the growth void function, which determines whether a cell
    // is in the growth void from its normalized radial distance 
    bool void_introduced = false;
    Array<int, Dynamic, 1> in_void = Array<int, Dynamic, 1>::Zero(n);
    std::function<bool(T)> in_void_func;
    if (growth_void_mode == GrowthVoidMode::NONE)
    {
        in_void_func = [](T x){ return false; };
    } 
    else if (growth_void_mode == GrowthVoidMode::FIXED_CORE)
    {
        in_void_func = [&growth_void_params](T x)
        {
            return x < growth_void_params["core_fraction"];
        };
    }
    else    // growth_void_mode == GrowthVoidMode::FRACTIONAL_ANNULUS)
    {
        in_void_func = [&growth_void_params](T x)
        {
            return x < 1 - growth_void_params["peripheral_fraction"];
        };
    }

    // If a growth void is to be imposed ... 
    if (growth_void_mode != GrowthVoidMode::NONE)
    {
        // Have we reached the minimum number of cells?
        if (n >= growth_void_params["mincells"])
        {
            in_void = inGrowthVoid<T>(cells, boundary_idx, in_void_func);
            void_introduced = (in_void.sum() > 0); 
        }
    }

    // Write the initial population to file
    std::unordered_map<int, int> write_other_cols;
    if (find_boundary)
    {
        write_other_cols[__colidx_boundary] = 0;     // Write boundary indicators as ints
    }
    if (track_poles)
    {
        write_other_cols[__colidx_negpole_t0] = 1;   // Write pole ages as floats
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
                daughter_length_dist, daughter_angle_dist
            );
            cells = div_result.first;
            daughter_pairs = div_result.second; 
        }
        else                // Otherwise, simply divide 
        {
            auto div_result = divideCells<T>(
                cells, parents, t, R, Rcell, to_divide, growth_dists, rng,
                daughter_length_dist, daughter_angle_dist
            );
            cells = div_result.first; 
            daughter_pairs = div_result.second; 
        }
        n = cells.rows();

        // If division has occurred ... 
        if (to_divide.sum() > 0)
        {
            // Switch cells between groups if desired 
            if (switch_mode == SwitchMode::INHERIT)
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
                T dist = neighbors(k, Eigen::seq(2, 3)).matrix().norm(); 
                to_adhere(k) = (
                    adhesion_map.find(pair) != adhesion_map.end() &&
                    dist > R + Rcell && dist < 2 * R
                ); 
            }
            // Update peripheral cells 
            if (find_boundary)
            {
                boundary_idx = getBoundary<T>(cells, R, mincells_for_boundary);
                for (int i = 0; i < n; ++i)
                    cells(i, __colidx_boundary) = 0;
                for (const int i : boundary_idx)
                    cells(i, __colidx_boundary) = 1;
            }
            // Update growth void 
            //
            // Keep track of new daughter cells (which are not in the void, 
            // because they are the progeny of obviously growing cells)
            in_void.conservativeResize(n);
            for (int i = n - to_divide.sum(); i < n; ++i)
                in_void(i) = 0;
        }

        // Update cell positions and orientations
        auto result = stepRungeKuttaAdaptive<T>(
            A, b, bs, cells, neighbors, to_adhere, dt, iter, R, Rcell,
            cell_cell_prefactors, surface_contact_density, max_noise, rng,
            uniform_dist, adhesion_mode, adhesion_params, confine_mode,
            boundary_idx, confine_params
        ); 
        Array<T, Dynamic, Dynamic> cells_new = result.first;
        Array<T, Dynamic, 4> errors = result.second;

        // If the error is big, retry the step with a smaller stepsize (up to
        // a given maximum number of attempts)
        if (iter % iter_update_stepsize == 0)
        {
            // Enforce a composite error of the form tol * (1 + y), for the
            // maximum error
            //
            // Here, y (which determines the scale of the error) is taken to 
            // be the old cell positions and orientations 
            Array<T, Dynamic, 4> z = (
                Array<T, Dynamic, 4>::Ones(n, 4) + cells(Eigen::all, __colseq_coords).abs()
            ); 
            Array<T, Dynamic, 4> max_scale = max_error_allowed * z;
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
                    cell_cell_prefactors, surface_contact_density, max_noise, rng,
                    uniform_dist, adhesion_mode, adhesion_params, confine_mode,
                    boundary_idx, confine_params
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
                cell_cell_prefactors, surface_contact_density, max_noise, rng,
                uniform_dist, adhesion_mode, adhesion_params, confine_mode,
                boundary_idx, confine_params
            ); 
            cells_new = result.first;
            errors = result.second;
        }
        // If desired, print a warning message if the error is big
        #ifdef DEBUG_WARN_LARGE_ERROR
            Array<T, Dynamic, 4> z = (
                Array<T, Dynamic, 4>::Ones(n, 4) + cells(Eigen::all, __colseq_coords).abs()
            );
            Array<T, Dynamic, 4> max_scale = max_error_allowed * z;
            T max_error = max((errors / max_scale).maxCoeff(), min_error);
            if (max_error > 5)
            {
                std::cout << "[WARN] Maximum error is > 5 times the desired error "
                          << "(absolute tol = relative tol = " << max_error_allowed
                          << ", iteration " << iter << ", time = " << t
                          << ", dt = " << dt << ")" << std::endl;
            }
        #endif
        cells = cells_new;

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
                T dist = neighbors(k, Eigen::seq(2, 3)).matrix().norm(); 
                to_adhere(k) = (
                    adhesion_map.find(pair) != adhesion_map.end() &&
                    dist > R + Rcell && dist < 2 * R
                ); 
            }
        }

        // Update peripheral cells
        if (find_boundary && iter % iter_update_boundary == 0)
        {
            boundary_idx = getBoundary<T>(cells, R, mincells_for_boundary);
            for (int i = 0; i < n; ++i)
                cells(i, __colidx_boundary) = 0;
            for (const int i : boundary_idx)
                cells(i, __colidx_boundary) = 1;
        }

        // Switch cells between groups if desired 
        if (switch_mode == SwitchMode::MARKOV)
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
                T dist = neighbors(k, Eigen::seq(2, 3)).matrix().norm(); 
                to_adhere(k) = (
                    adhesion_map.find(pair) != adhesion_map.end() &&
                    dist > R + Rcell && dist < 2 * R
                ); 
            }
            // Correct growth rates for cells within the growth void that have 
            // just switched 
            for (int i = 0; i < n; ++i)
            {
                if (in_void(i) && cells(i, __colidx_growth) > 0)
                    cells(i, __colidx_growth) = 0.0;
            }
            // Truncate cell-surface friction coefficients according to Coulomb's law
            if (truncate_surface_friction)
            {
                truncateSurfaceFrictionCoeffsCoulomb<T>(
                    cells, R, E0, surface_contact_density, surface_coulomb_coeff
                );
            }
            else    // Otherwise, ensure that friction coefficients are correct after switching 
            {
                for (int i = 0; i < n; ++i)
                    cells(i, __colidx_eta1) = cells(i, __colidx_maxeta1);
            }
        }

        // Introduce or update growth void
        if ((growth_void_mode == GrowthVoidMode::FIXED_CORE && !void_introduced) ||
            growth_void_mode == GrowthVoidMode::FRACTIONAL_ANNULUS
        )
        {
            // Have we reached the minimum number of cells? 
            if (n >= growth_void_params["mincells"])
            {
                in_void = inGrowthVoid<T>(cells, boundary_idx, in_void_func);
                void_introduced = (in_void.sum() > 0); 
            }
        }
        
        // Write the current population to file if the simulation time has 
        // just passed a multiple of dt_write 
        double t_old_factor = std::fmod(t - dt + 1e-12, dt_write);
        double t_new_factor = std::fmod(t + 1e-12, dt_write);  
        if (write && t_old_factor > t_new_factor) 
        {
            std::cout << "Iteration " << iter << ": " << n << " cells, time = "
                      << t << ", max error = " << errors.abs().maxCoeff()
                      << ", avg error = " << errors.abs().sum() / (4 * n)
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

/**
 * TODO In progress
 *
 * Run a simulation with the given initial population of cells.
 *
 * This function runs simulations in which the cells switch between two 
 * groups that differ by growth rate and one or more additional physical
 * attributes. The growth rate and chosen attributes are taken to be normally
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
 * @param sigma0 Cell-surface adhesion energy density.
 * @param density Cell density (including the EPS). 
 * @param dt Stepsize per iteration. 
 * @param write If true, write simulation output to file(s). 
 * @param outprefix Output filename prefix.
 * @param dt_write Write cells to file during each iteration in which the time
 *                 has passed a multiple of this value. 
 * @param iter_update_neighbors Update neighboring cells every this many 
 *                              iterations.
 * @param iter_update_boundary Update peripheral cells every this many 
 *                             iterations (only if `confine_mode` is not NONE).
 * @param neighbor_threshold Threshold for distinguishing between neighboring
 *                           and non-neighboring cells.
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
 * @param daughter_angle_bound Bound on daughter cell re-orientation angle.
 * @param truncate_surface_friction If true, truncate cell-surface friction
 *                                  coefficients according to Coulomb's law
 *                                  of friction.
 * @param surface_coulomb_coeff Friction coefficient that relates the velocity
 *                              of each cell to the normal force due to cell-
 *                              surface repulsion. 
 * @param max_noise Maximum noise to be added to each generalized force used 
 *                  to compute the velocities.
 * @param eta_cell_cell Array of cell-cell friction coefficient between cells
 *                      in different groups. 
 * @param adhesion_mode Choice of potential used to model cell-cell adhesion.
 *                      Can be NONE (0), KIHARA (1), or GBK (2).
 * @param adhesion_map Set of pairs of group IDs (1, 2, ...) for which cells 
 *                     can adhere to each other. 
 * @param adhesion_params Parameters required to compute cell-cell adhesion
 *                        forces.
 * @param confine_mode Confinement mode. Can be NONE (0), RADIAL (1), or 
 *                     CHANNEL (2). 
 * @param confine_params Parameters required to compute confinement forces. 
 * @param growth_void_mode Choice of growth void to be introduced within the
 *                         biofilm. Can be NONE (0), FIXED_CORE (1), or
 *                         FRACTIONAL_ANNULUS (2).
 * @param growth_void_params Parameters required to introduce growth void 
 *                           within the biofilm.
 * @param track_poles If true, keep track of pole birth times.
 * @param colidx_negpole_t0 Column index for negative pole birth time. 
 * @param colidx_pospole_t0 Column index for positive pole birth time. 
 * @returns Final population of cells.  
 */
template <typename T>
std::pair<Array<T, Dynamic, Dynamic>, std::vector<int> >
    runSimulationVerletNewtonian(const Ref<const Array<T, Dynamic, Dynamic> >& cells_init,
                                 std::vector<int>& parents_init,
                                 const int max_iter,
                                 const int n_cells,
                                 const T R,
                                 const T Rcell,
                                 const T L0,
                                 const T Ldiv,
                                 const T E0,
                                 const T Ecell,
                                 const T sigma0,
                                 const T density, 
                                 const T dt,
                                 const bool write,
                                 const std::string outprefix,
                                 const T dt_write,
                                 const int iter_update_neighbors,
                                 const int iter_update_boundary,
                                 const T neighbor_threshold,
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
                                 const T daughter_angle_bound,
                                 const bool truncate_surface_friction,
                                 const T surface_coulomb_coeff,
                                 const T max_noise,
                                 const Ref<const Array<T, Dynamic, Dynamic> >& eta_cell_cell, 
                                 const AdhesionMode adhesion_mode, 
                                 std::unordered_set<std::pair<int, int>, boost::hash<std::pair<int, int> > >& adhesion_map, 
                                 std::unordered_map<std::string, T>& adhesion_params,
                                 const ConfinementMode confine_mode,
                                 std::unordered_map<std::string, T>& confine_params,
                                 const GrowthVoidMode growth_void_mode,
                                 std::unordered_map<std::string, T>& growth_void_params,
                                 const bool track_poles = false,
                                 const int colidx_negpole_t0 = __colidx_group + 1,
                                 const int colidx_pospole_t0 = __colidx_group + 2)
{
    Array<T, Dynamic, Dynamic> cells(cells_init);
    T t = 0;
    int iter = 0;
    int n = cells.rows();
    boost::random::mt19937 rng(rng_seed);

    // Prefactors for cell-cell interaction forces
    Array<T, 3, 1> cell_cell_prefactors; 
    cell_cell_prefactors << 2.5 * E0 * sqrt(R),
                            2.5 * E0 * sqrt(R) * pow(2 * (R - Rcell), 2.5),
                            2.5 * Ecell * sqrt(Rcell);

    // Surface contact area density
    const T surface_contact_density = pow(sigma0 * R * R / (4 * E0), 1. / 3.);

    // Compute initial array of neighboring cells
    Array<T, Dynamic, 6> neighbors = getCellNeighbors<T>(cells, neighbor_threshold, R, Ldiv);

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
        T dist = neighbors(k, Eigen::seq(2, 3)).matrix().norm(); 
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
    params["density"] = floatToString<T>(density, precision); 
    params["stepsize"] = floatToString<T>(dt, precision); 
    params["dt_write"] = floatToString<T>(dt_write, precision); 
    params["iter_update_neighbors"] = std::to_string(iter_update_neighbors);
    params["iter_update_boundary"] = std::to_string(iter_update_boundary);
    params["neighbor_threshold"] = floatToString<T>(neighbor_threshold, precision);
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
    params["daughter_angle_bound"] = floatToString<T>(daughter_angle_bound, precision);
    params["truncate_surface_friction"] = (truncate_surface_friction ? "1" : "0"); 
    params["surface_coulomb_coeff"] = floatToString<T>(surface_coulomb_coeff, precision); 
    params["max_noise"] = floatToString<T>(max_noise, precision);
    for (int i = 0; i < n_groups; ++i)
    {
        for (int j = i + 1; j < n_groups; ++j)
        {
            std::stringstream ss; 
            ss << "eta_cell_cell_" << i + 1 << "_" << j + 1; 
            params[ss.str()] = floatToString<T>(eta_cell_cell(i, j), precision); 
        }
    }
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
    params["confine_mode"] = std::to_string(static_cast<int>(confine_mode)); 
    if (confine_mode != ConfinementMode::NONE)
    {
        for (auto&& item : confine_params)
        {
            std::stringstream ss;
            std::string key = item.first;
            T value = item.second;
            ss << "confine_" << key; 
            if (key == "mincells_for_boundary")   // mincells_for_boundary is an integer value
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
            if (key == "mincells")                // mincells is an integer value 
                params[ss.str()] = std::to_string(static_cast<int>(value));
            else 
                params[ss.str()] = floatToString<T>(value);
        }
    }

    // Peripheral cells are required when either imposing confinement or
    // imposing a growth void 
    //
    // If both are present, then the minimum number of cells for the boundary
    // calculation is set to the *minimum* of confine_params["mincells_for_boundary"]
    // and growth_void_params["mincells"]
    const bool find_boundary = (
        confine_mode != ConfinementMode::NONE || growth_void_mode != GrowthVoidMode::NONE
    );
    int mincells_for_boundary = 0; 
    if (confine_mode != ConfinementMode::NONE && growth_void_mode != GrowthVoidMode::NONE) 
        mincells_for_boundary = min(
            static_cast<int>(confine_params["mincells_for_boundary"]),
            static_cast<int>(growth_void_params["mincells"])
        ); 
    else if (confine_mode != ConfinementMode::NONE)
        mincells_for_boundary = static_cast<int>(confine_params["mincells_for_boundary"]);
    else if (growth_void_mode != GrowthVoidMode::NONE) 
        mincells_for_boundary = static_cast<int>(growth_void_params["mincells"]);

    // Get initial subset of peripheral cells (only if confinement or a growth
    // void is present)
    std::vector<int> boundary_idx;
    if (find_boundary)
        boundary_idx = getBoundary<T>(cells, R, mincells_for_boundary);

    // Determine an initial growth void
    //
    // First define the growth void function, which determines whether a cell
    // is in the growth void from its normalized radial distance 
    bool void_introduced = false;
    Array<int, Dynamic, 1> in_void = Array<int, Dynamic, 1>::Zero(n);
    std::function<bool(T)> in_void_func;
    if (growth_void_mode == GrowthVoidMode::NONE)
    {
        in_void_func = [](T x){ return false; };
    } 
    else if (growth_void_mode == GrowthVoidMode::FIXED_CORE)
    {
        in_void_func = [&growth_void_params](T x)
        {
            return x < growth_void_params["core_fraction"];
        };
    }
    else    // growth_void_mode == GrowthVoidMode::FRACTIONAL_ANNULUS)
    {
        in_void_func = [&growth_void_params](T x)
        {
            return x < 1 - growth_void_params["peripheral_fraction"];
        };
    }

    // If a growth void is to be imposed ... 
    if (growth_void_mode != GrowthVoidMode::NONE)
    {
        // Have we reached the minimum number of cells?
        if (n >= growth_void_params["mincells"])
        {
            in_void = inGrowthVoid<T>(cells, boundary_idx, in_void_func);
            void_introduced = (in_void.sum() > 0); 
        }
    }

    // Write the initial population to file
    std::unordered_map<int, int> write_other_cols;
    if (find_boundary)
    {
        write_other_cols[__colidx_group + 1] = 0;
    }
    if (track_poles)
    {
        write_other_cols[colidx_negpole_t0] = 1; 
        write_other_cols[colidx_pospole_t0] = 1;
    }
    if (write)
    {
        params["t_curr"] = floatToString<T>(t);
        std::stringstream ss_init; 
        ss_init << outprefix << "_init.txt";
        std::string filename_init = ss_init.str(); 
        // If desired, write additional indicators for peripheral cells 
        Array<T, Dynamic, Dynamic> cells_(cells);
        if (find_boundary)
        {
            cells_.conservativeResize(n, cells_.cols() + 1);
            cells_.col(cells_.cols() - 1) = Array<T, Dynamic, 1>::Zero(n); 
            for (const int i : boundary_idx)
                cells_(i, cells_.cols() - 1) = 1;
        }
        writeCells<T>(cells_, params, filename_init, write_other_cols);
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
                daughter_length_dist, daughter_angle_dist, colidx_negpole_t0,
                colidx_pospole_t0
            );
            cells = div_result.first;
            daughter_pairs = div_result.second; 
        }
        else                // Otherwise, simply divide 
        {
            auto div_result = divideCells<T>(
                cells, parents, t, R, Rcell, to_divide, growth_dists, rng,
                daughter_length_dist, daughter_angle_dist
            );
            cells = div_result.first; 
            daughter_pairs = div_result.second; 
        }
        n = cells.rows();

        // Update neighboring cells, peripheral cells, and cells within growth
        // void if division has occurred
        if (to_divide.sum() > 0)
        {
            // Switch cells between groups if desired 
            if (switch_mode == SwitchMode::INHERIT)
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
                T dist = neighbors(k, Eigen::seq(2, 3)).matrix().norm(); 
                to_adhere(k) = (
                    adhesion_map.find(pair) != adhesion_map.end() &&
                    dist > R + Rcell && dist < 2 * R
                ); 
            }
            // Update peripheral cells 
            if (find_boundary)
                boundary_idx = getBoundary<T>(cells, R, mincells_for_boundary);
            // Update growth void 
            //
            // Keep track of new daughter cells (which are not in the void, 
            // because they are the progeny of obviously growing cells)
            in_void.conservativeResize(n);
            for (int i = n - to_divide.sum(); i < n; ++i)
                in_void(i) = 0;
        }

        // Update cell positions and orientations
        stepVerlet<T>(
            cells, neighbors, to_adhere, dt, iter, R, Rcell, cell_cell_prefactors,
            density, surface_contact_density, max_noise, rng, uniform_dist,
            eta_cell_cell, adhesion_mode, adhesion_params, confine_mode,
            boundary_idx, confine_params
        );

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
                T dist = neighbors(k, Eigen::seq(2, 3)).matrix().norm(); 
                to_adhere(k) = (
                    adhesion_map.find(pair) != adhesion_map.end() &&
                    dist > R + Rcell && dist < 2 * R
                ); 
            }
        }

        // Update peripheral cells
        if (find_boundary && iter % iter_update_boundary == 0)
            boundary_idx = getBoundary<T>(cells, R, mincells_for_boundary);

        // Switch cells between groups if desired 
        if (switch_mode == SwitchMode::MARKOV)
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
                T dist = neighbors(k, Eigen::seq(2, 3)).matrix().norm(); 
                to_adhere(k) = (
                    adhesion_map.find(pair) != adhesion_map.end() &&
                    dist > R + Rcell && dist < 2 * R
                ); 
            }
            // Correct growth rates for cells within the growth void that have 
            // just switched 
            for (int i = 0; i < n; ++i)
            {
                if (in_void(i) && cells(i, __colidx_growth) > 0)
                    cells(i, __colidx_growth) = 0.0;
            }
            // Truncate cell-surface friction coefficients according to Coulomb's law
            if (truncate_surface_friction)
            {
                truncateSurfaceFrictionCoeffsCoulomb<T>(
                    cells, R, E0, surface_contact_density, surface_coulomb_coeff
                );
            }
            else    // Otherwise, ensure that friction coefficients are correct after switching 
            {
                for (int i = 0; i < n; ++i)
                    cells(i, __colidx_eta1) = cells(i, __colidx_maxeta1);
            }
        }

        // Introduce or update growth void
        if ((growth_void_mode == GrowthVoidMode::FIXED_CORE && !void_introduced) ||
            growth_void_mode == GrowthVoidMode::FRACTIONAL_ANNULUS
        )
        {
            // Have we reached the minimum number of cells? 
            if (n >= growth_void_params["mincells"])
            {
                in_void = inGrowthVoid<T>(cells, boundary_idx, in_void_func);
                void_introduced = (in_void.sum() > 0); 
            }
        }
        
        // Write the current population to file if the simulation time has 
        // just passed a multiple of dt_write 
        double t_old_factor = std::fmod(t - dt + 1e-12, dt_write);
        double t_new_factor = std::fmod(t + 1e-12, dt_write);  
        if (write && t_old_factor > t_new_factor) 
        {
            std::cout << "Iteration " << iter << ": " << n << " cells, time = "
                      << t << ", dt = " << dt << std::endl;
            params["t_curr"] = floatToString<T>(t);
            std::stringstream ss; 
            ss << outprefix << "_iter" << iter << ".txt"; 
            std::string filename = ss.str();
            // If desired, write additional indicators for peripheral cells 
            Array<T, Dynamic, Dynamic> cells_(cells);
            if (find_boundary)
            {
                cells_.conservativeResize(n, cells_.cols() + 1);
                cells_.col(cells_.cols() - 1) = Array<T, Dynamic, 1>::Zero(n); 
                for (const int i : boundary_idx)
                    cells_(i, cells_.cols() - 1) = 1;
            }
            writeCells<T>(cells_, params, filename, write_other_cols);
        }
    }

    // Write final population to file
    if (write)
    {
        params["t_curr"] = floatToString<T>(t);
        std::stringstream ss_final; 
        ss_final << outprefix << "_final.txt";
        std::string filename_final = ss_final.str(); 
        // If desired, write additional indicators for peripheral cells 
        Array<T, Dynamic, Dynamic> cells_(cells);
        if (find_boundary)
        {
            cells_.conservativeResize(n, cells_.cols() + 1);
            cells_.col(cells_.cols() - 1) = Array<T, Dynamic, 1>::Zero(n); 
            for (const int i : boundary_idx)
                cells_(i, cells_.cols() - 1) = 1;
        }
        writeCells<T>(cells_, params, filename_final, write_other_cols);
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

/**
 * TODO This code should be updated 
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
 * @param sigma0 Cell-surface adhesion energy density.
 * @param max_stepsize Maximum stepsize per iteration.
 * @param min_stepsize Minimum stepsize per iteration. 
 * @param write If true, write simulation output to file(s). 
 * @param outprefix Output filename prefix. 
 * @param dt_write Write cells to file during each iteration in which the time
 *                 has passed a multiple of this value. 
 * @param iter_update_neighbors Update neighboring cells every this many 
 *                              iterations.
 * @param iter_update_boundary Update peripheral cells every this many 
 *                             iterations (only if `confine_mode` is not NONE).
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
 * @param colidx_plasmid Column index for plasmid copy-number. 
 * @param partition_logratio_std Standard deviation for the normal distribution
 *                               that determines the log-ratio of plasmid 
 *                               copy-numbers in the daughter cells after
 *                               each division event. 
 * @param daughter_length_std Standard deviation of daughter length ratio 
 *                            distribution. 
 * @param daughter_angle_bound Bound on daughter cell re-orientation angle.
 * @param truncate_surface_friction If true, truncate cell-surface friction
 *                                  coefficients according to Coulomb's law
 *                                  of friction.
 * @param surface_coulomb_coeff Friction coefficient that relates the velocity
 *                              of each cell to the normal force due to cell-
 *                              surface repulsion. 
 * @param max_noise Maximum noise to be added to each generalized force used 
 *                  to compute the velocities.
 * @param adhesion_mode Choice of potential used to model cell-cell adhesion.
 *                      Can be NONE (0), KIHARA (1), or GBK (2).
 * @param adhesion_params Parameters required to compute cell-cell adhesion
 *                        forces.
 * @param confine_mode Confinement mode. Can be NONE (0), RADIAL (1), or 
 *                     CHANNEL (2). 
 * @param confine_params Parameters required to compute confinement forces. 
 * @param growth_void_mode Choice of growth void to be introduced within the
 *                         biofilm. Can be NONE (0), FIXED_CORE (1), or
 *                         FRACTIONAL_ANNULUS (2).
 * @param growth_void_params Parameters required to introduce growth void 
 *                           within the biofilm. 
 * @returns Final population of cells.  
 */
template <typename T>
std::pair<Array<T, Dynamic, Dynamic>, std::vector<int> >
    runSimulationWithPlasmid(const Ref<const Array<T, Dynamic, Dynamic> >& cells_init,
                             std::vector<int>& parents_init,
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
                             const T min_stepsize,
                             const bool write,
                             const std::string outprefix,
                             const T dt_write,
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
                             const int colidx_plasmid, 
                             const T partition_logratio_std,
                             const T daughter_length_std,
                             const T daughter_angle_bound,
                             const bool truncate_surface_friction,
                             const T surface_coulomb_coeff,
                             const T max_noise,
                             const AdhesionMode adhesion_mode, 
                             std::unordered_map<std::string, T>& adhesion_params,
                             const ConfinementMode confine_mode, 
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
    Array<T, 3, 1> cell_cell_prefactors; 
    cell_cell_prefactors << 2.5 * E0 * sqrt(R),
                            2.5 * E0 * sqrt(R) * pow(2 * (R - Rcell), 2.5),
                            2.5 * Ecell * sqrt(Rcell);

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
        to_adhere(k) = (
            cells(ni, __colidx_group) == 1 && cells(nj, __colidx_group) == 1 &&
            dist > R + Rcell && dist < 2 * R
        ); 
    }

    // Initialize parent IDs (TODO check that they have been correctly specified)
    std::vector<int> parents(parents_init); 

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
    params["min_stepsize"] = floatToString<T>(min_stepsize, precision); 
    params["dt_write"] = floatToString<T>(dt_write, precision); 
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
    params["truncate_surface_friction"] = (truncate_surface_friction ? "1" : "0"); 
    params["surface_coulomb_coeff"] = floatToString<T>(surface_coulomb_coeff, precision); 
    params["max_noise"] = floatToString<T>(max_noise, precision); 
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
    params["confine_mode"] = std::to_string(static_cast<int>(confine_mode)); 
    if (confine_mode != ConfinementMode::NONE)
    {
        for (auto&& item : confine_params)
        {
            std::stringstream ss;
            std::string key = item.first;
            T value = item.second;
            ss << "confine_" << key; 
            if (key == "mincells_for_boundary")   // mincells_for_boundary is an integer value
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
            if (key == "mincells")                // mincells is an integer value 
                params[ss.str()] = std::to_string(static_cast<int>(value));
            else 
                params[ss.str()] = floatToString<T>(value);
        }
    }

    // Peripheral cells are required when either imposing confinement or
    // imposing a growth void 
    //
    // If both are present, then the minimum number of cells for the boundary
    // calculation is set to the *minimum* of confine_params["mincells_for_boundary"]
    // and growth_void_params["mincells"]
    const bool find_boundary = (
        confine_mode != ConfinementMode::NONE || growth_void_mode != GrowthVoidMode::NONE
    );
    int mincells_for_boundary = 0; 
    if (confine_mode != ConfinementMode::NONE && growth_void_mode != GrowthVoidMode::NONE) 
        mincells_for_boundary = min(
            static_cast<int>(confine_params["mincells_for_boundary"]),
            static_cast<int>(growth_void_params["mincells"])
        ); 
    else if (confine_mode != ConfinementMode::NONE)
        mincells_for_boundary = static_cast<int>(confine_params["mincells_for_boundary"]);
    else if (growth_void_mode != GrowthVoidMode::NONE) 
        mincells_for_boundary = static_cast<int>(growth_void_params["mincells"]);

    // Get initial subset of peripheral cells (only if confinement or a growth
    // void is present)
    std::vector<int> boundary_idx;
    if (find_boundary)
        boundary_idx = getBoundary<T>(cells, R, mincells_for_boundary);

    // Determine an initial growth void
    //
    // First define the growth void function, which determines whether a cell
    // is in the growth void from its normalized radial distance 
    bool void_introduced = false;
    Array<int, Dynamic, 1> in_void = Array<int, Dynamic, 1>::Zero(n);
    std::function<bool(T)> in_void_func;
    if (growth_void_mode == GrowthVoidMode::NONE)
    {
        in_void_func = [](T x){ return false; };
    } 
    else if (growth_void_mode == GrowthVoidMode::FIXED_CORE)
    {
        in_void_func = [&growth_void_params](T x)
        {
            return x < growth_void_params["core_fraction"];
        };
    }
    else    // growth_void_mode == GrowthVoidMode::FRACTIONAL_ANNULUS)
    {
        in_void_func = [&growth_void_params](T x)
        {
            return x < 1 - growth_void_params["peripheral_fraction"];
        };
    }

    // If a growth void is to be imposed ... 
    if (growth_void_mode != GrowthVoidMode::NONE)
    {
        // Have we reached the minimum number of cells?
        if (n >= growth_void_params["mincells"])
        {
            in_void = inGrowthVoid<T>(cells, boundary_idx, in_void_func);
            void_introduced = (in_void.sum() > 0); 
        }
    } 

    // Write the initial population to file
    std::unordered_map<int, int> write_other_cols { {colidx_plasmid, 0} }; 
    if (write)
    {
        params["t_curr"] = floatToString<T>(t);
        std::stringstream ss_init; 
        ss_init << outprefix << "_init.txt";
        std::string filename_init = ss_init.str(); 
        // If desired, write additional indicators for peripheral cells 
        Array<T, Dynamic, Dynamic> cells_(cells);
        if (find_boundary)
        {
            cells_.conservativeResize(n, cells_.cols() + 1);
            cells_.col(cells_.cols() - 1) = Array<T, Dynamic, 1>::Zero(n); 
            for (const int i : boundary_idx)
                cells_(i, cells_.cols() - 1) = 1;
        }
        writeCells<T>(cells_, params, filename_init, write_other_cols);
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

    // Check that the plasmid copy-number column index is valid 
    #ifdef DEBUG_CHECK_COLUMN_INDICES
        if (colidx_plasmid >= cells.cols() || colidx_plasmid <= __colidx_group)
            throw std::runtime_error("Invalid column index for plasmid copy-number");
    #endif

    // Run the simulation ...
    while (!terminate(n, iter))
    {
        // Divide the cells that have reached division length
        Array<int, Dynamic, 1> to_divide = divideMaxLength<T>(cells, Ldiv);
        if (to_divide.sum() > 0)
            std::cout << "... Dividing " << to_divide.sum() << " cells "
                      << "(iteration " << iter << ")" << std::endl;
        cells = divideCellsWithPlasmid<T>(
            cells, parents, t, R, Rcell, to_divide, growth_dists, rng,
            daughter_length_dist, daughter_angle_dist, colidx_plasmid, 
            partition_logratio_dist
        );
        n = cells.rows(); 
        
        // Switch groups for all cells that have lost the plasmid 
        for (int i = 0; i < n; ++i)
        {
            if (cells(i, colidx_plasmid) == 0)
            {
                // Sample the cell's new growth rate and attribute values 
                int group = (group_default == 1 ? 2 : 1);
                int j = group - 1; 
                cells(i, __colidx_group) = group;
                T growth_rate = growth_dists[j](rng);
                cells(i, __colidx_growth) = growth_rate; 
                for (int k = 0; k < n_attributes; ++k)
                {
                    auto pair = std::make_pair(j, k); 
                    T attribute = attribute_dists[pair](rng); 
                    cells(i, switch_attributes[k]) = attribute; 
                }
            }
        }

        // Update neighboring cells, peripheral cells, and cells within growth
        // void if division has occurred
        if (to_divide.sum() > 0)
        {
            // Update neighboring cells 
            neighbors = getCellNeighbors<T>(cells, neighbor_threshold, R, Ldiv);
            to_adhere.resize(neighbors.rows());
            for (int k = 0; k < neighbors.rows(); ++k)
            {
                int ni = neighbors(k, 0); 
                int nj = neighbors(k, 1);
                T dist = neighbors(k, Eigen::seq(2, 3)).matrix().norm(); 
                to_adhere(k) = (
                    cells(ni, __colidx_group) == 1 && cells(nj, __colidx_group) == 1 &&
                    dist > R + Rcell && dist < 2 * R
                ); 
            }
            // Update peripheral cells 
            if (find_boundary)
                boundary_idx = getBoundary<T>(cells, R, mincells_for_boundary);
            // Update growth void 
            //
            // Keep track of new daughter cells (which are not in the void, 
            // because they are the progeny of obviously growing cells)
            in_void.conservativeResize(n);
            for (int i = n - to_divide.sum(); i < n; ++i)
                in_void(i) = 0;
        }

        // Update cell positions and orientations
        auto result = stepRungeKuttaAdaptive<T>(
            A, b, bs, cells, neighbors, to_adhere, dt, iter, R, Rcell,
            cell_cell_prefactors, surface_contact_density, max_noise, rng,
            uniform_dist, adhesion_mode, adhesion_params, confine_mode,
            boundary_idx, confine_params
        ); 
        Array<T, Dynamic, Dynamic> cells_new = result.first;
        Array<T, Dynamic, 4> errors = result.second;

        // If the error is big, retry the step with a smaller stepsize (up to
        // a given maximum number of attempts)
        if (iter % iter_update_stepsize == 0)
        {
            // Enforce a composite error of the form tol * (1 + y), for the
            // maximum error
            //
            // Here, y (which determines the scale of the error) is taken to 
            // be the old cell positions and orientations 
            Array<T, Dynamic, 4> z = (
                Array<T, Dynamic, 4>::Ones(n, 4) + cells(Eigen::all, __colseq_coords).abs()
            ); 
            Array<T, Dynamic, 4> max_scale = max_error_allowed * z;
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
                    cell_cell_prefactors, surface_contact_density, max_noise, rng,
                    uniform_dist, adhesion_mode, adhesion_params, confine_mode,
                    boundary_idx, confine_params
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
                cell_cell_prefactors, surface_contact_density, max_noise, rng,
                uniform_dist, adhesion_mode, adhesion_params, confine_mode,
                boundary_idx, confine_params
            ); 
            cells_new = result.first;
            errors = result.second;
        }
        // If desired, print a warning message if the error is big
        #ifdef DEBUG_WARN_LARGE_ERROR
            Array<T, Dynamic, 4> z = (
                Array<T, Dynamic, 4>::Ones(n, 4) + cells(Eigen::all, __colseq_coords).abs()
            );
            Array<T, Dynamic, 4> max_scale = max_error_allowed * z;
            T max_error = max((errors / max_scale).maxCoeff(), min_error);
            if (max_error > 5)
            {
                std::cout << "[WARN] Maximum error is > 5 times the desired error "
                          << "(absolute tol = relative tol = " << max_error_allowed
                          << ", iteration " << iter << ", time = " << t
                          << ", dt = " << dt << ")" << std::endl;
            }
        #endif
        cells = cells_new;

        // Truncate cell-surface friction coefficients according to Coulomb's law
        if (truncate_surface_friction)
        {
            truncateSurfaceFrictionCoeffsCoulomb<T>(
                cells, R, E0, surface_contact_density, surface_coulomb_coeff
            );
        }

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
                to_adhere(k) = (
                    cells(ni, __colidx_group) == 1 && cells(nj, __colidx_group) == 1 &&
                    dist > R + Rcell && dist < 2 * R
                ); 
            }
        }

        // Update peripheral cells
        if (find_boundary && iter % iter_update_boundary == 0)
            boundary_idx = getBoundary<T>(cells, R, mincells_for_boundary);

        // Truncate cell-surface friction coefficients according to Coulomb's law
        if (truncate_surface_friction)
        {
            truncateSurfaceFrictionCoeffsCoulomb<T>(
                cells, R, E0, surface_contact_density, surface_coulomb_coeff
            );
        }
        else    // Otherwise, ensure that friction coefficients are correct after switching 
        {
            for (int i = 0; i < n; ++i)
                cells(i, __colidx_eta1) = cells(i, __colidx_maxeta1);
        }

        // Introduce or update growth void
        if ((growth_void_mode == GrowthVoidMode::FIXED_CORE && !void_introduced) ||
            growth_void_mode == GrowthVoidMode::FRACTIONAL_ANNULUS
        )
        {
            // Have we reached the minimum number of cells? 
            if (n >= growth_void_params["mincells"])
            {
                in_void = inGrowthVoid<T>(cells, boundary_idx, in_void_func);
                void_introduced = (in_void.sum() > 0); 
            }
        }
        
        // Write the current population to file if the simulation time has 
        // just passed a multiple of dt_write 
        double t_old_factor = std::fmod(t - dt + 1e-12, dt_write);
        double t_new_factor = std::fmod(t + 1e-12, dt_write);  
        if (write && t_old_factor > t_new_factor)
        {
            std::cout << "Iteration " << iter << ": " << n << " cells, time = "
                      << t << ", max error = " << errors.abs().maxCoeff()
                      << ", avg error = " << errors.abs().sum() / (4 * n)
                      << ", dt = " << dt << std::endl;
            params["t_curr"] = floatToString<T>(t);
            std::stringstream ss; 
            ss << outprefix << "_iter" << iter << ".txt"; 
            std::string filename = ss.str();
            // If desired, write additional indicators for peripheral cells 
            Array<T, Dynamic, Dynamic> cells_(cells);
            if (find_boundary)
            {
                cells_.conservativeResize(n, cells_.cols() + 1);
                cells_.col(cells_.cols() - 1) = Array<T, Dynamic, 1>::Zero(n); 
                for (const int i : boundary_idx)
                    cells_(i, cells_.cols() - 1) = 1;
            }
            writeCells<T>(cells_, params, filename, write_other_cols);
        }
    }

    // Write final population to file
    if (write)
    {
        params["t_curr"] = floatToString<T>(t);
        std::stringstream ss_final; 
        ss_final << outprefix << "_final.txt";
        std::string filename_final = ss_final.str(); 
        // If desired, write additional indicators for peripheral cells 
        Array<T, Dynamic, Dynamic> cells_(cells);
        if (find_boundary)
        {
            cells_.conservativeResize(n, cells_.cols() + 1);
            cells_.col(cells_.cols() - 1) = Array<T, Dynamic, 1>::Zero(n); 
            for (const int i : boundary_idx)
                cells_(i, cells_.cols() - 1) = 1;
        }
        writeCells<T>(cells_, params, filename_final, write_other_cols);
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
