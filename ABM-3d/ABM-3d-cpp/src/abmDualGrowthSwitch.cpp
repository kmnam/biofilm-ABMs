/**
 * An agent-based model of 3-D biofilm growth that switches cells between two
 * states that exhibit differential growth. 
 *
 * In what follows, a population of N cells is represented as a 2-D array 
 * with N rows, whose columns are as specified in `include/indices.hpp`.
 *
 * Authors:
 *     Kee-Myoung Nam
 *
 * Last updated:
 *     3/14/2025
 */

#include <Eigen/Dense>
#include "../include/indices.hpp"
#include "../include/simulation.hpp"
#include "../include/utils.hpp"

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
    const T eta_surface = static_cast<T>(json_data["eta_surface"].as_double()); 
    const T max_stepsize = static_cast<T>(json_data["max_stepsize"].as_double());
    const T min_stepsize = static_cast<T>(json_data["min_stepsize"].as_double()); 
    const T dt_write = static_cast<T>(json_data["dt_write"].as_double()); 
    const int iter_update_stepsize = json_data["iter_update_stepsize"].as_int64(); 
    const int iter_update_neighbors = json_data["iter_update_neighbors"].as_int64();
    //const T neighbor_threshold = 2 * (2 * R + L0);
    const T neighbor_threshold = 0.5 * (2 * R + Ldiv); 
    const int max_iter = json_data["max_iter"].as_int64();
    const int n_cells = json_data["n_cells"].as_int64();
    const T growth_mean1 = static_cast<T>(json_data["growth_mean1"].as_double());
    const T growth_std1 = static_cast<T>(json_data["growth_std1"].as_double());
    const T growth_mean2 = static_cast<T>(json_data["growth_mean2"].as_double());
    const T growth_std2 = static_cast<T>(json_data["growth_std2"].as_double());
    const T lifetime_mean1 = static_cast<T>(json_data["lifetime_mean1"].as_double()); 
    const T lifetime_mean2 = static_cast<T>(json_data["lifetime_mean2"].as_double()); 
    const T daughter_length_std = static_cast<T>(json_data["daughter_length_std"].as_double());
    const T daughter_angle_xy_bound = static_cast<T>(json_data["daughter_angle_xy_bound"].as_double());
    const T daughter_angle_z_bound = static_cast<T>(json_data["daughter_angle_z_bound"].as_double());
    const T nz_threshold = static_cast<T>(json_data["nz_threshold"].as_double()); 
    const T max_rxy_noise = static_cast<T>(json_data["max_rxy_noise"].as_double()); 
    const T max_rz_noise = static_cast<T>(json_data["max_rz_noise"].as_double());
    const T max_nxy_noise = static_cast<T>(json_data["max_nxy_noise"].as_double()); 
    const T max_nz_noise = static_cast<T>(json_data["max_nz_noise"].as_double()); 
    const T max_error_allowed = static_cast<T>(json_data["max_error_allowed"].as_double());
    const bool truncate_surface_friction = json_data["truncate_surface_friction"].as_int64();
    const T surface_coulomb_coeff = (
        truncate_surface_friction ? static_cast<T>(json_data["surface_coulomb_coeff"].as_double()) : 0.0
    );
    const bool basal_only = json_data["basal_only"].as_int64(); 
    const T basal_min_overlap = (
        basal_only ? static_cast<T>(json_data["basal_min_overlap"].as_double()) : 0.0
    ); 
    
    // No cell-cell adhesion
    AdhesionMode adhesion_mode = AdhesionMode::NONE; 
    std::unordered_set<std::pair<int, int>, boost::hash<std::pair<int, int> > > adhesion_map;
    std::unordered_map<std::string, T> adhesion_params;

    // Vectors of growth rate means and standard deviations (identical for
    // both groups) 
    Array<T, Dynamic, 1> growth_means(2);
    Array<T, Dynamic, 1> growth_stds(2); 
    growth_means << growth_mean1, growth_mean2; 
    growth_stds << growth_std1, growth_std2; 

    // Vectors of friction coefficient means and standard deviations (identical
    // for both groups)
    std::vector<int> group_attributes { __colidx_maxeta1 };
    Array<T, Dynamic, Dynamic> attribute_means(2, 1);
    Array<T, Dynamic, Dynamic> attribute_stds(2, 1);
    attribute_means << eta_surface, eta_surface; 
    attribute_stds << 0.0, 0.0;

    // Switching rates between groups 1 and 2
    Array<T, Dynamic, Dynamic> switch_rates(2, 2); 
    switch_rates << 0.0, 1.0 / lifetime_mean1,
                    1.0 / lifetime_mean2, 0.0;

    // Output file prefix
    std::string outprefix = argv[2];

    // Random seed
    const int rng_seed = std::stoi(argv[3]);

    // Initialize simulation ...
    //
    // Define a founder cell at the origin at time zero, parallel to x-axis, 
    // with zero velocity, mean growth rate, and default viscosity and friction
    // coefficients
    Array<T, Dynamic, Dynamic> cells(1, __ncols_required);
    cells << 0, 0, 0, 0.96 * R, 1, 0, 0, 0, 0, 0, 0, 0, 0, L0, L0 / 2, 0, growth_mean1,
             eta_ambient, eta_surface, eta_surface, sigma0, 1;

    // Initialize parent IDs 
    std::vector<int> parents; 
    parents.push_back(-1); 
    
    // Run the simulation
    runSimulationAdaptiveLagrangian<T>(
        cells, parents, max_iter, n_cells, R, Rcell, L0, Ldiv, E0, Ecell, 
        max_stepsize, min_stepsize, true, outprefix, dt_write, iter_update_neighbors,
        iter_update_stepsize, max_error_allowed, min_error, max_tries_update_stepsize,
        neighbor_threshold, nz_threshold, rng_seed, 2, group_attributes, growth_means,
        growth_stds, attribute_means, attribute_stds, SwitchMode::MARKOV, switch_rates,
        daughter_length_std, daughter_angle_xy_bound, daughter_angle_z_bound,
        truncate_surface_friction, surface_coulomb_coeff, max_rxy_noise, max_rz_noise,
        max_nxy_noise, max_nz_noise, basal_only, basal_min_overlap, adhesion_mode,
        adhesion_map, adhesion_params
    ); 
    
    return 0; 
}
