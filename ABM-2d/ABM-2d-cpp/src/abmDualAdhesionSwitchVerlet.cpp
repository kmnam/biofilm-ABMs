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
 *     1/14/2025
 */

#include <Eigen/Dense>
#include "../include/indices.hpp"
#include "../include/simulation.hpp"
#include "../include/utils.hpp"

using namespace Eigen;

// Define floating-point type to be used 
typedef double T;

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
	const T density = static_cast<T>(json_data["density"].as_double()); 
    const T eta_ambient = static_cast<T>(json_data["eta_ambient"].as_double()); 
    const T dt = static_cast<T>(json_data["stepsize"].as_double()); 
    const T dt_write = static_cast<T>(json_data["dt_write"].as_double()); 
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
    const bool truncate_surface_friction = json_data["truncate_surface_friction"].as_int64();
    const T surface_coulomb_coeff = (
        truncate_surface_friction ? static_cast<T>(json_data["surface_coulomb_coeff"].as_double()) : 0.0
    );

	// Parse cell-cell adhesion parameters
    AdhesionMode adhesion_mode;
	const int token = json_data["adhesion_mode"].as_int64(); 
	if (token == 0)
	    adhesion_mode = AdhesionMode::NONE;
	else if (token == 1)
	    adhesion_mode = AdhesionMode::KIHARA; 
	else if (token == 2)
	    adhesion_mode = AdhesionMode::GBK;
	else 
	    throw std::runtime_error("Invalid cell-cell adhesion mode specified");
	std::unordered_set<std::pair<int, int>, boost::hash<std::pair<int, int> > > adhesion_map;
	adhesion_map.insert(std::make_pair(1, 1)); 
    std::unordered_map<std::string, T> adhesion_params;
    adhesion_params["strength"] = static_cast<T>(json_data["adhesion_strength"].as_double());
    adhesion_params["distance_exp"] = static_cast<T>(json_data["adhesion_distance_exp"].as_double()); 
    adhesion_params["mindist"] = static_cast<T>(json_data["adhesion_mindist"].as_double()); 
    if (adhesion_mode == AdhesionMode::GBK) 
        adhesion_params["anisotropy_exp1"] = static_cast<T>(json_data["adhesion_anisotropy_exp1"].as_double());

	// Parse cell-cell friction coefficient, which is assumed to be the same 
	// across all groups
	Array<T, Dynamic, Dynamic> eta_cell_cell = Array<T, Dynamic, Dynamic>::Zero(2, 2);
	for (int i = 0; i < 2; ++i)
	{
	    for (int j = i; j < 2; ++j)
		{
		    std::stringstream ss; 
			ss << "eta_cc" << i + 1 << j + 1; 
			T eta = static_cast<T>(json_data[ss.str()].as_double()); 
			eta_cell_cell(i, j) = eta;
			if (j > i)
			    eta_cell_cell(j, i) = eta;
		}
	}

    // No confinement forces or growth void 
    const ConfinementMode confine_mode = ConfinementMode::NONE; 
    std::unordered_map<std::string, T> confine_params; 
    const GrowthVoidMode growth_void_mode = GrowthVoidMode::NONE;
    std::unordered_map<std::string, T> growth_void_params;

    // Vectors of growth rate means and standard deviations (identical for
    // both groups) 
    Array<T, Dynamic, 1> growth_means(2);
    Array<T, Dynamic, 1> growth_stds(2); 
    growth_means << growth_mean, growth_mean; 
    growth_stds << growth_std, growth_std; 

    // Vectors of friction coefficient means and standard deviations
    std::vector<int> switch_attributes { __colidx_maxeta1 };
    Array<T, Dynamic, Dynamic> attribute_means(2, 1);
    Array<T, Dynamic, Dynamic> attribute_stds(2, 1);
    attribute_means << eta_mean1, eta_mean2;
    attribute_stds << eta_std1, eta_std2;

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
    cells << 0, 0, 0, 1, 0, 0, 0, 0, 0, L0, L0 / 2, 0, growth_mean, eta_ambient,
             eta_mean1, eta_mean1, 1;

    // Initialize parent IDs 
    std::vector<int> parents; 
    parents.push_back(-1); 
    
    // Run the simulation
    runSimulationVerletNewtonian<T>(
        cells, parents, max_iter, n_cells, R, Rcell, L0, Ldiv, E0, Ecell, sigma0, 
        density, dt, true, outprefix, dt_write, iter_update_neighbors,
		iter_update_boundary, neighbor_threshold, rng_seed, 2, switch_attributes,
		growth_means, growth_stds, attribute_means, attribute_stds, switch_rates,
		daughter_length_std, daughter_angle_bound, truncate_surface_friction,
		surface_coulomb_coeff, max_noise, eta_cell_cell, adhesion_mode,
		adhesion_map, adhesion_params, confine_mode, confine_params,
		growth_void_mode, growth_void_params
    ); 
    
    return 0; 
}