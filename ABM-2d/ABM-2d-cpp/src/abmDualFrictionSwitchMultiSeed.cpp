/**
 * An agent-based model of 2-D biofilm growth that switches cells between two
 * states that exhibit different friction coefficients.
 *
 * This simulates biofilms with multiple cells in the seed population, whose
 * coordinates are given in a second input file. 
 *
 * In what follows, a population of N cells is represented as a 2-D array 
 * with N rows, whose columns are as specified in `include/indices.hpp`.
 *
 * Authors:
 *     Kee-Myoung Nam
 *
 * Last updated:
 *     1/21/2025
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

    // Switching rates between groups 1 and 2
    Array<T, Dynamic, Dynamic> switch_rates(2, 2); 
    switch_rates << 0.0, 1.0 / lifetime_mean1,
                    1.0 / lifetime_mean2, 0.0;

    // Parse input coordinates file 
    std::string seed_filename = argv[2];
    Array<T, Dynamic, 4> seed_coords = Array<T, Dynamic, 4>::Zero(1, 4);
    std::ifstream infile(seed_filename); 
    std::string line;
    int i = 0; 
    while (std::getline(infile, line))
    {
        seed_coords.conservativeResize(i + 1, 4); 
        int j = 0;
        std::stringstream ss;
        std::string token; 
        ss << line; 
        while (std::getline(ss, token, '\t'))
        {
            if (j >= 4)
                throw std::runtime_error("Invalid coordinates for seed population");
            seed_coords(i, j) = static_cast<T>(std::stod(token));
            j++; 
        }
        i++; 
    }

    // Output file prefix
    std::string outprefix = argv[3];

    // Random seed
    const int rng_seed = std::stoi(argv[4]);

    // Initialize simulation ...
    //
    // Define each cell within the seed population at time zero with the given
    // coordinates, zero velocity, randomly sampled growth rate, and default 
    // viscosity and friction coefficients 
    Array<T, Dynamic, Dynamic> cells = Array<T, Dynamic, Dynamic>::Zero(seed_coords.rows(), __ncols_required);
    boost::random::mt19937 rng(rng_seed);
    boost::random::uniform_01<> uniform_dist; 
    for (int i = 0; i < seed_coords.rows(); ++i)
    {
        T growth_rate = growth_mean + growth_std * standardNormal<T>(rng, uniform_dist); 
        cells(i, __colidx_id) = i;
        cells(i, __colseq_coords) = seed_coords.row(i);
        cells(i, __colidx_l) = L0; 
        cells(i, __colidx_half_l) = L0 / 2;
        cells(i, __colidx_growth) = growth_rate; 
        cells(i, __colidx_eta0) = eta_ambient; 
        cells(i, __colidx_eta1) = eta_mean1;
        cells(i, __colidx_maxeta1) = eta_mean1; 
        cells(i, __colidx_group) = 1;
    }

    // Initialize parent IDs 
    std::vector<int> parents; 
    parents.push_back(-1); 
    
    // Run the simulation
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
