/**
 * An agent-based model of 2-D biofilm growth that switches cells between two
 * states that exhibit different friction coefficients. 
 *
 * In what follows, a population of N cells is represented as a 2-D array 
 * with N rows, whose columns are as specified in `include/indices.hpp`.
 *
 * Authors:
 *     Kee-Myoung Nam
 *
 * Last updated:
 *     1/23/2025
 */

#include <Eigen/Dense>
#include "../include/indices.hpp"
#include "../include/simulation.hpp"
#include "../include/utils.hpp"
#include "../include/prepattern.hpp"

using namespace Eigen;

// Define floating-point type to be used 
typedef double T;

// Maximum number of attempts to control stepsize per Runge-Kutta iteration 
const int max_tries_update_stepsize = 3;

// Minimum error per Runge-Kutta iteration
const T min_error = static_cast<T>(1e-30);

int main(int argc, char** argv)
{
    // Parse input json file 
    std::string json_filename = argv[1];
    boost::json::object json_data = parseConfigFile(json_filename).as_object();

    // Define required input parameters
    const T R = static_cast<T>(json_data["R"].as_double());
    const T Rcell = static_cast<T>(json_data["Rcell"].as_double());
    const T L0 = static_cast<T>(json_data["L0"].as_double());
    const T Ldiv = 2 * L0 + 2 * R;
    const T E0 = static_cast<T>(json_data["E0"].as_double());
    const T Ecell = static_cast<T>(json_data["Ecell"].as_double()); 
    const T sigma0 = static_cast<T>(json_data["sigma0"].as_double()); 
    const T eta_ambient = static_cast<T>(json_data["eta_ambient"].as_double()); 
    const T max_stepsize = static_cast<T>(json_data["max_stepsize"].as_double());
    const T min_stepsize = static_cast<T>(json_data["min_stepsize"].as_double()); 
    const T dt_write = static_cast<T>(json_data["dt_write"].as_double()); 
    const int iter_update_stepsize = json_data["iter_update_stepsize"].as_int64(); 
    const int iter_update_neighbors = json_data["iter_update_neighbors"].as_int64();
    const int iter_update_boundary = 0;
    const T neighbor_threshold = 2 * (2 * R + L0);
    const int max_iter = json_data["max_iter"].as_int64();
    const int n_cells = json_data["n_cells"].as_int64();
    const T growth_mean = static_cast<T>(json_data["growth_mean"].as_double());
    const T growth_std = static_cast<T>(json_data["growth_std"].as_double());
    const T eta_mean1 = static_cast<T>(json_data["eta_mean1"].as_double());
    const T eta_std1 = static_cast<T>(json_data["eta_std1"].as_double()); 
    const T eta_mean2 = static_cast<T>(json_data["eta_mean2"].as_double());
    const T eta_std2 = static_cast<T>(json_data["eta_std2"].as_double());
    const T lifetime_mean1 = static_cast<T>(json_data["lifetime_mean1"].as_double()); 
    const T lifetime_mean2 = static_cast<T>(json_data["lifetime_mean2"].as_double()); 
    const T daughter_length_std = static_cast<T>(json_data["daughter_length_std"].as_double());
    const T daughter_angle_bound = static_cast<T>(json_data["daughter_angle_bound"].as_double());
    const T max_noise = static_cast<T>(json_data["max_noise"].as_double()); 
    const T max_error_allowed = static_cast<T>(json_data["max_error_allowed"].as_double());
    const bool truncate_surface_friction = json_data["truncate_surface_friction"].as_int64();
    const T surface_coulomb_coeff = (
        truncate_surface_friction ? static_cast<T>(json_data["surface_coulomb_coeff"].as_double()) : 0.0
    );
    const AdhesionMode adhesion_mode = AdhesionMode::NONE;           // No cell-cell adhesion
    std::unordered_set<std::pair<int, int>, boost::hash<std::pair<int, int> > > adhesion_map; 
    std::unordered_map<std::string, T> adhesion_params;
    const ConfinementMode confine_mode = ConfinementMode::NONE;      // No confinement forces 
    std::unordered_map<std::string, T> confine_params; 
    const GrowthVoidMode growth_void_mode = GrowthVoidMode::NONE;    // No growth void 
    std::unordered_map<std::string, T> growth_void_params;

    // Parse pre-patterning parameters
    const int n_cells_init = json_data["n_cells_init"].as_int64();
    const PrepatternMode prepattern_mode = static_cast<PrepatternMode>(json_data["prepattern_mode"].as_int64());
    const double prepattern_switch_fraction = json_data["prepattern_switch_fraction"].as_double();

    // Vectors of growth rate means and standard deviations (identical for
    // both groups) 
    Array<T, Dynamic, 1> growth_means(2);
    Array<T, Dynamic, 1> growth_stds(2); 
    growth_means << growth_mean, growth_mean; 
    growth_stds << growth_std, growth_std; 

    // Vectors of friction coefficient means and standard deviations
    std::vector<int> group_attributes { __colidx_maxeta1 };
    Array<T, Dynamic, Dynamic> attribute_means(2, 1);
    Array<T, Dynamic, Dynamic> attribute_stds(2, 1);
    attribute_means << eta_mean1, eta_mean2;
    attribute_stds << eta_std1, eta_std2;

    // Vectors of dummy friction coefficient means and standard deviations
    // (identical for both groups) 
    Array<T, Dynamic, Dynamic> dummy_means(2, 1); 
    Array<T, Dynamic, Dynamic> dummy_stds(2, 1);
    dummy_means << eta_mean1, eta_mean1; 
    dummy_stds << eta_std1, eta_std1; 

    // Switching rates between groups 1 and 2
    Array<T, Dynamic, Dynamic> switch_rates(2, 2); 
    switch_rates << 0.0, 1.0 / lifetime_mean1,
                    1.0 / lifetime_mean2, 0.0;

    // Output file prefix
    std::string outprefix = argv[2];
    std::stringstream ss; 
    ss << outprefix << "_pre"; 
    std::string outprefix_pre = ss.str();  

    // Random seed
    const int rng_seed = std::stoi(argv[3]);

    // Initialize simulation ...
    //
    // Define a founder cell at the origin at time zero, parallel to x-axis, 
    // with zero velocity, mean growth rate, and default viscosity and friction
    // coefficients
    Array<T, Dynamic, Dynamic> cells(1, __ncols_required);
    cells << 0, 0, 0, 1, 0, 0, 0, 0, 0, L0, L0 / 2, 0, growth_mean, eta_ambient,
             eta_mean1, eta_mean1, 1;

    // Initialize parent IDs 
    std::vector<int> parents; 
    parents.push_back(-1);

    // Run the first stage of the simulation
    auto result = runSimulationAdaptiveLagrangian<T>(
        cells, parents, max_iter, n_cells_init, R, Rcell, L0, Ldiv, E0, Ecell, 
        sigma0, max_stepsize, min_stepsize, true, outprefix_pre, dt_write,
        iter_update_neighbors, iter_update_boundary, iter_update_stepsize, 
        max_error_allowed, min_error, max_tries_update_stepsize, neighbor_threshold,
        rng_seed, 2, group_attributes, growth_means, growth_stds, dummy_means,
        dummy_stds, SwitchMode::NONE, switch_rates, daughter_length_std,
        daughter_angle_bound, truncate_surface_friction, surface_coulomb_coeff,
        max_noise, adhesion_mode, adhesion_map, adhesion_params, confine_mode,
        confine_params, growth_void_mode, growth_void_params
    ); 
    cells = result.first; 
    parents = result.second;

    // Growth rate distribution functions: normal distributions with given means
    // and standard deviations
    boost::random::uniform_01<> uniform_dist; 
    std::vector<std::function<T(boost::random::mt19937&)> > growth_dists; 
    for (int i = 0; i < 2; ++i)    // There are two groups 
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
    for (int i = 0; i < 2; ++i)    // There are two groups 
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

    // Prepattern the cells
    boost::random::mt19937 rng(rng_seed); 
    prepattern<T>(
        cells, prepattern_mode, prepattern_switch_fraction, group_attributes,
        growth_dists, attribute_dists, rng
    );

    // Ensure that each cell has the right cell-surface friction coefficient 
    for (int i = 0; i < cells.rows(); ++i)
        cells(i, __colidx_eta1) = cells(i, __colidx_maxeta1); 
    
    // Run the second stage of the simulation
    runSimulationAdaptiveLagrangian<T>(
        cells, parents, max_iter, n_cells, R, Rcell, L0, Ldiv, E0, Ecell, sigma0, 
        max_stepsize, min_stepsize, true, outprefix, dt_write, iter_update_neighbors,
        iter_update_boundary, iter_update_stepsize, max_error_allowed,
        min_error, max_tries_update_stepsize, neighbor_threshold, rng_seed, 2,
        group_attributes, growth_means, growth_stds, attribute_means, 
        attribute_stds, SwitchMode::MARKOV, switch_rates, daughter_length_std,
        daughter_angle_bound, truncate_surface_friction, surface_coulomb_coeff,
        max_noise, adhesion_mode, adhesion_map, adhesion_params, confine_mode,
        confine_params, growth_void_mode, growth_void_params
    ); 
    
    return 0; 
}
