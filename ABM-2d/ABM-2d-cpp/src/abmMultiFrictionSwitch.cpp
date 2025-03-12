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
    const T neighbor_threshold = 2 * R + Ldiv; 
    const int max_iter = json_data["max_iter"].as_int64();
    const int n_cells = json_data["n_cells"].as_int64();
    const T growth_mean = static_cast<T>(json_data["growth_mean"].as_double());
    const T growth_std = static_cast<T>(json_data["growth_std"].as_double());
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

    // Parse friction coefficients for each group 
    int n_groups = 0;
    std::vector<T> eta_means, eta_stds; 
    while (true)
    {
        std::stringstream ss1, ss2; 
        ss1 << "eta_mean" << n_groups + 1; 
        ss2 << "eta_std" << n_groups + 1;
        T eta_mean, eta_std;
        // Parse friction coefficient mean for next group
        if (json_data.if_contains(ss1.str()))
            eta_mean = static_cast<T>(json_data[ss1.str()].as_double());
        else
            break;
        // Parse friction coefficient SD for next group 
        if (json_data.if_contains(ss2.str()))
            eta_std = static_cast<T>(json_data[ss2.str()].as_double());
        else
            throw std::runtime_error(
                "Improperly specified cell group (mean only, no SD, for friction "
                "coefficient)"
            );
        n_groups++;
        eta_means.push_back(eta_mean); 
        eta_stds.push_back(eta_std); 
    }

    // Has at least one group been specified? 
    if (n_groups == 0)
        throw std::runtime_error("No cell groups specified");

    // Parse group lifetimes
    std::vector<T> lifetimes; 
    for (int i = 0; i < n_groups; ++i)
    {
        std::stringstream ss; 
        ss << "lifetime_mean" << i + 1;
        T lifetime; 
        if (json_data.if_contains(ss.str()))    // Parse lifetime for next group 
        {
            lifetime = static_cast<T>(json_data[ss.str()].as_double());
        }
        else
        {
            std::stringstream ss_err; 
            ss_err << "Unspecified lifetime for cell group " << i + 1; 
            throw std::runtime_error(ss_err.str()); 
        }
        lifetimes.push_back(lifetime); 
    }

    // Sort the groups by descending friction coefficient
    std::vector<std::tuple<T, T, T> > groups_combined;
    for (int i = 0; i < n_groups; ++i)
        groups_combined.push_back(std::make_tuple(eta_means[i], eta_stds[i], lifetimes[i]));
    std::sort(
        groups_combined.begin(), groups_combined.end(),
        [](std::tuple<T, T, T>& a, std::tuple<T, T, T>& b)
        {
            return std::get<0>(a) > std::get<0>(b);
        }
    );

    // Vectors of growth rate means and standard deviations (identical for
    // all groups) 
    Array<T, Dynamic, 1> growth_means = growth_mean * Array<T, Dynamic, 1>::Ones(n_groups);
    Array<T, Dynamic, 1> growth_stds = growth_std * Array<T, Dynamic, 1>::Ones(n_groups);

    // Vectors of friction coefficient means and standard deviations
    std::vector<int> group_attributes { __colidx_maxeta1 };
    Array<T, Dynamic, Dynamic> attribute_means(n_groups, 1);
    Array<T, Dynamic, Dynamic> attribute_stds(n_groups, 1);
    for (int i = 0; i < n_groups; ++i)
    {
        attribute_means(i, 0) = std::get<0>(groups_combined[i]); 
        attribute_stds(i, 0) = std::get<1>(groups_combined[i]);
    }

    // Switching rates between consecutive groups 
    Array<T, Dynamic, Dynamic> switch_rates(n_groups, n_groups);
    for (int i = 0; i < n_groups; ++i)
    {
        if (i == 0)
        {
            switch_rates(i, i + 1) = 1.0 / std::get<2>(groups_combined[i]);
        }
        else if (i == n_groups - 1)
        {
            switch_rates(i, i - 1) = 1.0 / std::get<2>(groups_combined[i]); 
        }
        else 
        {
            switch_rates(i, i + 1) = 1.0 / (2 * std::get<2>(groups_combined[i])); 
            switch_rates(i, i - 1) = 1.0 / (2 * std::get<2>(groups_combined[i])); 
        }
    }

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
             std::get<0>(groups_combined[0]), std::get<0>(groups_combined[0]), 1;

    // Initialize parent IDs 
    std::vector<int> parents; 
    parents.push_back(-1); 
    
    // Run the simulation
    runSimulationAdaptiveLagrangian<T>(
        cells, parents, max_iter, n_cells, R, Rcell, L0, Ldiv, E0, Ecell, sigma0, 
        max_stepsize, min_stepsize, true, outprefix, dt_write, iter_update_neighbors,
        iter_update_boundary, iter_update_stepsize, max_error_allowed,
        min_error, max_tries_update_stepsize, neighbor_threshold, rng_seed, n_groups,
        group_attributes, growth_means, growth_stds, attribute_means, 
        attribute_stds, SwitchMode::MARKOV, switch_rates, daughter_length_std,
        daughter_angle_bound, truncate_surface_friction, surface_coulomb_coeff,
        max_noise, adhesion_mode, adhesion_map, adhesion_params, confine_mode,
        confine_params, growth_void_mode, growth_void_params
    ); 
    
    return 0; 
}
