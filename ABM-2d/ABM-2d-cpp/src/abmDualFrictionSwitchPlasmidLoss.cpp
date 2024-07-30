/**
 * TODO Update
 *
 * An agent-based model of 2-D biofilm growth that switches cells between two
 * states that exhibit different friction coefficients. 
 *
 * In what follows, a population of N cells is represented as a 2-D array of 
 * size (N, 11), where each row represents a cell and stores the following data:
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
 * 10) cell group identifier (1 for high friction, 2 for low friction)
 * 11) plasmid copy-number
 *
 * Authors:
 *     Kee-Myoung Nam
 *
 * Last updated:
 *     7/30/2024
 */

#include <Eigen/Dense>
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
    const T max_stepsize = static_cast<T>(json_data["max_stepsize"].as_double()); 
    const int iter_write = json_data["iter_write"].as_int64(); 
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
    const int group_default = json_data["state_default"].as_int64(); 
    const T partition_logratio_std = static_cast<T>(json_data["partition_logratio_std"].as_double());
    const T daughter_length_std = static_cast<T>(json_data["daughter_length_std"].as_double());
    const T daughter_angle_bound = static_cast<T>(json_data["daughter_angle_bound"].as_double());
    const T max_error_allowed = static_cast<T>(json_data["max_error_allowed"].as_double());
    const AdhesionMode adhesion_mode = AdhesionMode::NONE;    // No cell-cell adhesion
    std::unordered_map<std::string, T> adhesion_params;
    const bool confine = false;    // No radial confinement forces 
    std::unordered_map<std::string, T> confine_params; 
    const GrowthVoidMode growth_void_mode = GrowthVoidMode::NONE;    // No growth void 
    std::unordered_map<std::string, T> growth_void_params; 

    // Vectors of growth rate means and standard deviations (identical for
    // both groups) 
    Array<T, 2, 1> growth_means;
    Array<T, 2, 1> growth_stds;
    growth_means << growth_mean, growth_mean; 
    growth_stds << growth_std, growth_std; 

    // Vectors of friction coefficient means and standard deviations
    std::vector<int> switch_attributes { 9 };
    Array<T, 2, Dynamic> attribute_means(2, 1);
    Array<T, 2, Dynamic> attribute_stds(2, 1);
    attribute_means << eta_mean1, eta_mean2;
    attribute_stds << eta_std1, eta_std2;

    // Output file prefix
    std::string outprefix = argv[2];

    // Random seed
    const int rng_seed = std::stoi(argv[3]);

    // Initialize simulation ...
    //
    // Define a founder cell at the origin at time zero, parallel to x-axis, 
    // with mean growth rate and default viscosity and friction coefficients
    Array<T, Dynamic, Dynamic> cells(1, 12);
    cells << 0, 0, 1, 0, L0, L0 / 2, 0, growth_mean, eta_ambient, eta_mean1,
             static_cast<T>(group_default), 1;
    
    // Run the simulation
    runSimulationWithPlasmid<T>(
        cells, max_iter, n_cells, R, Rcell, L0, Ldiv, E0, Ecell, sigma0, 
        max_stepsize, true, outprefix, iter_write, iter_update_neighbors,
        iter_update_boundary, iter_update_stepsize, max_error_allowed,
        min_error, max_tries_update_stepsize, neighbor_threshold, rng_seed,
        group_default, switch_attributes, growth_means, growth_stds,
        attribute_means, attribute_stds, partition_logratio_std,
        daughter_length_std, daughter_angle_bound, adhesion_mode,
        adhesion_params, confine, confine_params, growth_void_mode,
        growth_void_params
    ); 
    
    return 0; 
}