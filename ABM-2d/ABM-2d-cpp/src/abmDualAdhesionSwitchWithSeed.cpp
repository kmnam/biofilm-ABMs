/**
 * TODO Update
 *
 * An agent-based model in which cells exhibit two states but do not switch
 * between each other. Two cells in state 1 can adhere to each other.
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
 *     6/16/2024
 */

#include <algorithm>
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

// Strategies for seeding group 2 cells
enum SeedStrategy
{
    RANDOM, 
    INNERMOST, 
    OUTERMOST
}; 

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
    const T neighbor_threshold = 2 * (2 * R + L0);
    const int max_iter = json_data["max_iter"].as_int64();
    const int n_cells_init = json_data["n_cells_init"].as_int64();
    const int n_cells_total = json_data["n_cells_total"].as_int64();
    const T growth_mean = static_cast<T>(json_data["growth_mean"].as_double());
    const T growth_std = static_cast<T>(json_data["growth_std"].as_double());
    const T lifetime_mean1 = static_cast<T>(json_data["lifetime_mean1"].as_double()); 
    const T lifetime_mean2 = static_cast<T>(json_data["lifetime_mean2"].as_double()); 
    const T daughter_length_std = static_cast<T>(json_data["daughter_length_std"].as_double());
    const T daughter_angle_bound = static_cast<T>(json_data["daughter_angle_bound"].as_double());
    const T max_error_allowed = static_cast<T>(json_data["max_error_allowed"].as_double());
    const AdhesionMode adhesion_mode = static_cast<AdhesionMode>(json_data["adhesion_mode"].as_int64()); 
    std::unordered_map<std::string, T> adhesion_params;
    adhesion_params["strength"] = static_cast<T>(json_data["adhesion_strength"].as_double()); 
    if (adhesion_mode == GBK) 
    {
        adhesion_params["anisotropy_exp1"] = static_cast<T>(json_data["adhesion_anisotropy_exp1"].as_double()); 
        adhesion_params["anisotropy_exp2"] = static_cast<T>(json_data["adhesion_anisotropy_exp2"].as_double()); 
        adhesion_params["well_depth_delta"] = static_cast<T>(json_data["adhesion_well_depth_delta"].as_double()); 
    }
    else if (adhesion_mode == LJ)
    {
        adhesion_params["cos_eps_parallel"] = static_cast<T>(std::cos(json_data["adhesion_eps_parallel"].as_double()));
        adhesion_params["eps_collinear"] = static_cast<T>(json_data["adhesion_eps_collinear"].as_double());
        adhesion_params["delta"] = static_cast<T>(json_data["adhesion_delta"].as_double()); 
    }
    const double seed_fraction = json_data["seed_fraction"].as_double();
    const SeedStrategy seed_strategy = static_cast<SeedStrategy>(json_data["seed_strategy"].as_int64()); 

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

    // Switching rates between groups 1 and 2
    Array<T, Dynamic, Dynamic> switch_rates(2, 2); 
    switch_rates << 0.0, 1.0 / lifetime_mean1,
                    1.0 / lifetime_mean2, 0.0;

    // Output file prefix
    std::string outprefix = argv[2];
    std::stringstream ss1, ss2; 
    ss1 << outprefix << "_preseed";
    ss2 << outprefix << "_postseed"; 

    // Random seed
    const int rng_seed = std::stoi(argv[3]);

    // Initialize simulation ...
    //
    // Define a founder cell at the origin at time zero, parallel to x-axis, 
    // with mean growth rate and default viscosity and friction coefficients
    Array<T, Dynamic, Dynamic> cells(1, 11);
    cells << 0, 0, 1, 0, L0, L0 / 2, 0, growth_mean, eta_ambient, eta_surface, 2;
    
    // Run the simulation up to the desired number of initial cells ...
    //
    // Every cell belongs to group 2 and exhibits all the same attributes
    cells = runSimulation<T>(
        cells, max_iter, n_cells_init, R, Rcell, L0, Ldiv, E0, Ecell, sigma0, 
        max_stepsize, true, ss1.str(), iter_write, iter_update_neighbors,
        iter_update_stepsize, max_error_allowed, min_error,
        max_tries_update_stepsize, neighbor_threshold, rng_seed,
        growth_mean, growth_std, daughter_length_std, daughter_angle_bound,
        adhesion_mode, adhesion_params
    );

    // Then introduce a fraction of group 1 cells
    const int n_seed = static_cast<int>(n_cells_init * seed_fraction);
    boost::random::mt19937 rng(rng_seed);
    if (n_seed > 0)
    {
        if (seed_strategy == RANDOM)
        {
            std::vector<int> sample = sampleWithoutReplacement(n_cells_init, n_seed, rng);
            for (auto it = sample.begin(); it != sample.end(); ++it)
            {
                int k = *it;
                cells(k, 10) = 1;   // Note that no attributes are explicitly changed 
            }
        }
        else    // seed_strategy == INNERMOST or seed_strategy == OUTERMOST
        {
            // Get the center of mass of the population 
            T rx_mean = cells.col(0).sum() / n_cells_init; 
            T ry_mean = cells.col(1).sum() / n_cells_init;
            Matrix<T, 1, 2> r_mean; 
            r_mean << rx_mean, ry_mean; 

            // Get the distance of each cell center to the center of mass 
            std::vector<T> dists; 
            for (int i = 0; i < n_cells_init; ++i)
                dists.push_back((cells(i, Eigen::seq(0, 1)).matrix() - r_mean).norm());

            // Sort the distances and identify the cutoff distance by which 
            // to change the innermost/outermost cells to group 2
            if (seed_strategy == INNERMOST)
            {
                // Sort the distances in ascending order
                std::vector<T> dists2(dists); 
                std::sort(dists2.begin(), dists2.end());

                // Get the cutoff distance
                T cutoff = std::numeric_limits<T>::infinity(); 
                if (n_seed < n_cells_init)
                    cutoff = dists2[n_seed];

                // Find all cells with distance less than the cutoff 
                for (int i = 0; i < n_cells_init; ++i)
                {
                    if (dists[i] < cutoff)
                        cells(i, 10) = 1;    // Note that no attributes are explicitly changed
                }
            }
            else    // seed_strategy == OUTERMOST
            {
                // Sort the distances in descending order
                std::vector<T> dists2(dists); 
                std::sort(dists2.begin(), dists2.end(), [](T a, T b){ return a > b; });

                // Get the cutoff distance
                T cutoff = 0.0;
                if (n_seed < n_cells_init)
                    cutoff = dists2[n_seed];

                // Find all cells with distance greater than the cutoff
                for (int i = 0; i < n_cells_init; ++i)
                {
                    if (dists[i] > cutoff)
                        cells(i, 10) = 1;   // Note that no attributes are explicitly changed
                }
            }
        }
    }
    
    // Now run the simulation up to the desired number of total cells ...
    //
    // Now, switching between the two groups is allowed 
    runSimulation<T>(
        cells, max_iter, n_cells_total, R, Rcell, L0, Ldiv, E0, Ecell, sigma0, 
        max_stepsize, true, ss2.str(), iter_write, iter_update_neighbors,
        iter_update_stepsize, max_error_allowed, min_error,
        max_tries_update_stepsize, neighbor_threshold, rng_seed, 2,
        switch_attributes, growth_means, growth_stds, attribute_means, 
        attribute_stds, switch_rates, daughter_length_std, daughter_angle_bound,
        adhesion_mode, adhesion_params
    );
   
    return 0; 
}
