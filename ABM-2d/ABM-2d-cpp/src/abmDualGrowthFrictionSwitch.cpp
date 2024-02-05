/**
 * An agent-based model of 2-D biofilm growth that switches cells between two
 * states that exhibit different growth rates and friction coefficients.
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
 * 10) cell group identifier (1 for slow-growing/high-friction, 2 for
 *     fast-growing/low-friction)
 *
 * Authors:
 *     Kee-Myoung Nam
 *
 * Last updated:
 *     2/5/2024
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
    const T neighbor_threshold = 2 * (2 * R + L0);
    const int max_iter = json_data["max_iter"].as_int64();
    const int n_cells = json_data["n_cells"].as_int64();
    const double growth_mean1 = json_data["growth_mean1"].as_double();
    const double growth_std1 = json_data["growth_std1"].as_double();
    const double growth_mean2 = json_data["growth_mean2"].as_double(); 
    const double growth_std2 = json_data["growth_std2"].as_double();
    const double eta_mean1 = json_data["eta_mean1"].as_double();
    const double eta_std1 = json_data["eta_std1"].as_double(); 
    const double eta_mean2 = json_data["eta_mean2"].as_double();
    const double eta_std2 = json_data["eta_std2"].as_double();
    const T lifetime_mean1 = static_cast<T>(json_data["lifetime_mean1"].as_double()); 
    const T lifetime_mean2 = static_cast<T>(json_data["lifetime_mean2"].as_double()); 
    const double daughter_length_std = json_data["daughter_length_std"].as_double();
    const double daughter_angle_bound = json_data["daughter_angle_bound"].as_double();
    const T max_error_allowed = static_cast<T>(json_data["max_error_allowed"].as_double());

    // Vectors of growth rate means and standard deviations 
    std::vector<double> growth_means { growth_mean1, growth_mean2 };
    std::vector<double> growth_stds  { growth_std1, growth_std2 };

    // Vectors of friction coefficient means and standard deviations
    const int switch_attribute = 9;
    std::vector<double> attribute_means { eta_mean1, eta_mean2 };
    std::vector<double> attribute_stds  { eta_std1, eta_std2 };

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
    // with mean growth rate and default viscosity and friction coefficients
    Array<T, Dynamic, Dynamic> cells(1, 11);
    cells << 0, 0, 1, 0, L0, L0 / 2, 0, growth_mean1, eta_ambient, eta_mean1, 1;
    
    // Run the simulation
    runSimulation<T>(
        cells, max_iter, n_cells, R, Rcell, L0, Ldiv, E0, Ecell, sigma0, 
        max_stepsize, true, outprefix, iter_write, iter_update_neighbors,
        iter_update_stepsize, max_error_allowed, min_error,
        max_tries_update_stepsize, neighbor_threshold, rng_seed, 2,
        switch_attribute, growth_means, growth_stds, attribute_means, 
        attribute_stds, switch_rates, daughter_length_std, daughter_angle_bound
    ); 
    
    return 0; 
}
