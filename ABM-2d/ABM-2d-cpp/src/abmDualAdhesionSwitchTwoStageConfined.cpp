/**
 * An agent-based model that switches cells between two states that exhibit
 * different friction coefficients. 
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
 * 10) cell group identifier (1 for self-adhering, 2 for non-adhering)
 *
 * Authors:
 *     Kee-Myoung Nam
 *
 * Last updated:
 *     7/23/2024
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
    const int iter_write = json_data["iter_write"].as_int64(); 
    const int iter_update_stepsize = json_data["iter_update_stepsize"].as_int64(); 
    const int iter_update_neighbors = json_data["iter_update_neighbors"].as_int64();
    const int iter_update_boundary = json_data["iter_update_boundary"].as_int64();
    const T neighbor_threshold = 2 * (2 * R + L0);
    const int max_iter = json_data["max_iter"].as_int64(); 
    const int n_seed = json_data["n_cells_seed"].as_int64();
    const int n_cells = json_data["n_cells_total"].as_int64();
    const T growth_mean = static_cast<T>(json_data["growth_mean"].as_double());
    const T growth_std = static_cast<T>(json_data["growth_std"].as_double());
    const T lifetime_mean1_stage1 = static_cast<T>(json_data["lifetime_mean1_stage1"].as_double()); 
    const T lifetime_mean2_stage1 = static_cast<T>(json_data["lifetime_mean2_stage1"].as_double());
    const T lifetime_mean1_stage2 = static_cast<T>(json_data["lifetime_mean1_stage2"].as_double()); 
    const T lifetime_mean2_stage2 = static_cast<T>(json_data["lifetime_mean2_stage2"].as_double()); 
    const T daughter_length_std = static_cast<T>(json_data["daughter_length_std"].as_double());
    const T daughter_angle_bound = static_cast<T>(json_data["daughter_angle_bound"].as_double());
    const T max_error_allowed = static_cast<T>(json_data["max_error_allowed"].as_double());
    const AdhesionMode adhesion_mode = static_cast<AdhesionMode>(json_data["adhesion_mode"].as_int64()); 
    std::unordered_map<std::string, T> adhesion_params;
    adhesion_params["strength"] = static_cast<T>(json_data["adhesion_strength"].as_double());
    adhesion_params["distance_exp"] = static_cast<T>(json_data["adhesion_distance_exp"].as_double()); 
    adhesion_params["mindist"] = static_cast<T>(json_data["adhesion_mindist"].as_double()); 
    if (adhesion_mode == GBK) 
    {
        adhesion_params["anisotropy_exp1"] = static_cast<T>(json_data["adhesion_anisotropy_exp1"].as_double()); 
        adhesion_params["anisotropy_exp2"] = static_cast<T>(json_data["adhesion_anisotropy_exp2"].as_double());
        adhesion_params["well_depth_delta"] = static_cast<T>(json_data["adhesion_well_depth_delta"].as_double()); 
    }
    const bool confine = true; 
    std::unordered_map<std::string, T> confine_params; 
    confine_params["find_boundary"] = static_cast<T>(json_data["confine_find_boundary"].as_int64());
    confine_params["mincells_for_boundary"] = 20;
    confine_params["rest_radius_factor"] = static_cast<T>(json_data["confine_rest_radius_factor"].as_double()); 
    confine_params["spring_const"] = static_cast<T>(json_data["confine_spring_const"].as_double()); 

    // Surface contact area density and powers of cell radius 
    const T surface_contact_density = std::pow(sigma0 * R * R / (4 * E0), 1. / 3.);
    const T sqrtR = std::sqrt(R); 
    const T powRdiff = std::pow(R - Rcell, 1.5);
    Array<T, 4, 1> cell_cell_prefactors;
    cell_cell_prefactors << 2.5 * sqrtR,
                            2.5 * E0 * sqrtR,
                            E0 * powRdiff,
                            Ecell;

    // Vectors of growth rate means and standard deviations (identical for
    // both groups) 
    Array<T, Dynamic, 1> growth_means(2);
    Array<T, Dynamic, 1> growth_stds(2); 
    growth_means << growth_mean, growth_mean; 
    growth_stds << growth_std, growth_std; 

    // Vectors of friction coefficient means and standard deviations (identical
    // for both groups)
    std::vector<int> switch_attributes { 9 };
    Array<T, Dynamic, Dynamic> attribute_means(2, 1);
    Array<T, Dynamic, Dynamic> attribute_stds(2, 1);
    attribute_means << eta_surface, eta_surface;
    attribute_stds << 0.0, 0.0;

    // Switching rates between groups 1 and 2 for the two stages
    Array<T, Dynamic, Dynamic> switch_rates1(2, 2), switch_rates2(2, 2); 
    switch_rates1 << 0.0, 1.0 / lifetime_mean1_stage1,
                     1.0 / lifetime_mean2_stage1, 0.0;
    switch_rates2 << 0.0, 1.0 / lifetime_mean1_stage2,
                     1.0 / lifetime_mean2_stage2, 0.0;


    // Output file prefix
    std::string outprefix = argv[2];

    // Random seed
    const int rng_seed = std::stoi(argv[3]);

    // Initialize simulation ...
    //
    // Define a founder cell at the origin at time zero, parallel to x-axis, 
    // with mean growth rate and default viscosity and friction coefficients
    Array<T, Dynamic, Dynamic> cells(1, 11);
    cells << 0, 0, 1, 0, L0, L0 / 2, 0, growth_mean, eta_ambient, eta_surface, 1;
    
    // Run the first stage of the simulation ... 
    std::stringstream ss1, ss2; 
    ss1 << outprefix << "_stage1"; 
    cells = runSimulation<T>(
        cells, max_iter, n_seed, R, Rcell, L0, Ldiv, E0, Ecell, sigma0, 
        max_stepsize, true, ss1.str(), iter_write, iter_update_neighbors,
        iter_update_boundary, iter_update_stepsize, max_error_allowed,
        min_error, max_tries_update_stepsize, neighbor_threshold, rng_seed, 2,
        switch_attributes, growth_means, growth_stds, attribute_means, 
        attribute_stds, switch_rates1, daughter_length_std, daughter_angle_bound,
        adhesion_mode, adhesion_params, confine, confine_params
    ); 
  
    // ... then run the second stage of the simulation
    ss2 << outprefix << "_stage2";
    runSimulation<T>(
        cells, max_iter, n_cells, R, Rcell, L0, Ldiv, E0, Ecell, sigma0, 
        max_stepsize, true, ss2.str(), iter_write, iter_update_neighbors,
        iter_update_boundary, iter_update_stepsize, max_error_allowed,
        min_error, max_tries_update_stepsize, neighbor_threshold, rng_seed, 2,
        switch_attributes, growth_means, growth_stds, attribute_means, 
        attribute_stds, switch_rates2, daughter_length_std, daughter_angle_bound,
        adhesion_mode, adhesion_params, confine, confine_params
    ); 

    return 0; 
}
