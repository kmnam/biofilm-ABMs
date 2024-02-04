/**
 * An agent-based model that simulates the growth of the basal layer of a 
 * 3-D biofilm. 
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

#define EIGEN_DONT_PARALLELIZE    // Disable internal parallelization within Eigen

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
    const double growth_mean = json_data["growth_mean"].as_double();
    const double growth_std = json_data["growth_std"].as_double(); 
    const T E0 = static_cast<T>(json_data["E0"].as_double());
    const T Ecell = static_cast<T>(json_data["Ecell"].as_double()); 
    const T sigma0 = static_cast<T>(json_data["sigma0"].as_double()); 
    const T eta_ambient = static_cast<T>(json_data["eta_ambient"].as_double()); 
    const T eta_surface = static_cast<T>(json_data["eta_surface"].as_double()); 
    const T max_stepsize = static_cast<T>(json_data["max_stepsize"].as_double());
    const int iter_write = json_data["iter_write"].as_int64(); 
    const int iter_update_stepsize = json_data["iter_update_stepsize"].as_int64(); 
    const int iter_update_neighbors = json_data["iter_update_neighbors"].as_int64(); 
    const T neighbor_threshold = 2 * (2 * R + L0);
    const int n_cells = json_data["n_cells"].as_int64();
    const double daughter_length_std = json_data["daughter_length_std"].as_double();
    const double daughter_angle_xy_bound = json_data["daughter_angle_xy_bound"].as_double();
    const double daughter_angle_z_bound = json_data["daughter_angle_z_bound"].as_double();
    const T nz_threshold = static_cast<T>(json_data["nz_threshold"].as_double());
    const T max_error_allowed = static_cast<T>(json_data["max_error_allowed"].as_double());

    // Output file prefix
    std::string prefix = argv[2];

    // Random seed
    const int rng_seed = std::stoi(argv[3]);

    // Initialize simulation ...
    //
    // Define a founder cell at the origin at time zero, parallel to x-axis 
    // and xy-plane, with mean growth rate and default viscosity and friction
    // coefficients
    Array<T, Dynamic, Dynamic> cells(1, 13);
    cells << 0, 0, 0.99 * R, 1, 0, 0, L0, L0 / 2, 0, growth_mean, eta_ambient, eta_surface, sigma0;

    // Run the simulation
    runSimulation<T>(
        cells, n_cells, R, Rcell, L0, Ldiv, E0, Ecell, max_stepsize, iter_write,
        iter_update_stepsize, max_error_allowed, min_error, max_tries_update_stepsize,
        neighbor_threshold, nz_threshold, outprefix, rng_seed, growth_mean, 
        growth_std, daughter_length_std, daughter_angle_xy_bound, daughter_angle_z_bound
    );
    
    return 0; 
}
