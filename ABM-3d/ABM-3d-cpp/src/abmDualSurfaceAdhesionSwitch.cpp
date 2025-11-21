/**
 * An agent-based model of 3-D biofilm growth that switches cells between two
 * states that exhibit different cell-surface adhesion energies and/or friction
 * coefficients. 
 *
 * In what follows, a population of N cells is represented as a 2-D array 
 * with N rows, whose columns are as specified in `include/indices.hpp`.
 *
 * Authors:
 *     Kee-Myoung Nam
 *
 * Last updated:
 *     11/20/2025
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

    // Check if the simulation varies just the cell-surface adhesion energy
    // density (default), the cell-surface friction coefficient (--coef), or 
    // both (--combined)
    int surface_friction_mode = 0; 
    if (argc > 4)
    {
        std::string arg = argv[4];
        if (arg == "--eta")
            surface_friction_mode = 1;
        else if (arg == "--combined")
            surface_friction_mode = 2; 
        else
            throw std::runtime_error(
                "Unrecognized input argument for friction mode"
            );  
    }

    // Define required input parameters
    const T R = static_cast<T>(json_data["R"].as_double());
    const T Rcell = static_cast<T>(json_data["Rcell"].as_double());
    const T L0 = static_cast<T>(json_data["L0"].as_double());
    const T Ldiv = 2 * L0 + 2 * R;
    const T E0 = static_cast<T>(json_data["E0"].as_double());
    const T Ecell = static_cast<T>(json_data["Ecell"].as_double()); 
    const T eta_ambient = static_cast<T>(json_data["eta_ambient"].as_double()); 
    const T max_stepsize = static_cast<T>(json_data["max_stepsize"].as_double());
    const T min_stepsize = static_cast<T>(json_data["min_stepsize"].as_double()); 
    const T dt_write = static_cast<T>(json_data["dt_write"].as_double()); 
    const int iter_update_stepsize = json_data["iter_update_stepsize"].as_int64(); 
    const int iter_update_neighbors = json_data["iter_update_neighbors"].as_int64();
    //const T neighbor_threshold = 2 * (2 * R + L0);
    const T neighbor_threshold = 0.5 * (2 * R + Ldiv); 
    const int max_iter = json_data["max_iter"].as_int64();
    const int n_cells = json_data["n_cells"].as_int64();
    const T max_time = static_cast<T>(json_data["max_time"].as_double());  
    const T growth_mean = static_cast<T>(json_data["growth_mean"].as_double());
    const T growth_std = static_cast<T>(json_data["growth_std"].as_double());
    T sigma0_mean1, sigma0_mean2, eta1_mean1, eta1_mean2;
    if (surface_friction_mode == 0 || surface_friction_mode == 2)
    { 
        sigma0_mean1 = static_cast<T>(json_data["sigma0_mean1"].as_double()); 
        sigma0_mean2 = static_cast<T>(json_data["sigma0_mean2"].as_double());
    }
    else    // surface_friction_mode == 1 
    {
        sigma0_mean1 = static_cast<T>(json_data["sigma0"].as_double());
        sigma0_mean2 = 0.0; 
    }
    if (surface_friction_mode == 1 || surface_friction_mode == 2)
    {
        eta1_mean1 = static_cast<T>(json_data["eta1_mean1"].as_double()); 
        eta1_mean2 = static_cast<T>(json_data["eta1_mean2"].as_double());
    }
    else    // surface_friction_mode == 0 
    {
        eta1_mean1 = static_cast<T>(json_data["eta_surface"].as_double());
        eta1_mean2 = 0.0; 
    }
    const T lifetime_mean1 = static_cast<T>(json_data["lifetime_mean1"].as_double()); 
    const T lifetime_mean2 = static_cast<T>(json_data["lifetime_mean2"].as_double());
    const T switch_timescale = static_cast<T>(json_data["switch_timescale"].as_double()); 
    const T daughter_length_std = static_cast<T>(json_data["daughter_length_std"].as_double());
    const T daughter_angle_xy_bound = static_cast<T>(json_data["daughter_angle_xy_bound"].as_double());
    const T daughter_angle_z_bound = static_cast<T>(json_data["daughter_angle_z_bound"].as_double());
    const T nz_threshold = static_cast<T>(json_data["nz_threshold"].as_double()); 
    const T max_rxy_noise = static_cast<T>(json_data["max_rxy_noise"].as_double()); 
    const T max_rz_noise = static_cast<T>(json_data["max_rz_noise"].as_double());
    const T max_nxy_noise = static_cast<T>(json_data["max_nxy_noise"].as_double()); 
    const T max_nz_noise = static_cast<T>(json_data["max_nz_noise"].as_double()); 
    const T max_error_allowed = static_cast<T>(json_data["max_error_allowed"].as_double());
    const bool basal_only = json_data["basal_only"].as_int64(); 
    const T basal_min_overlap = (
        basal_only ? static_cast<T>(json_data["basal_min_overlap"].as_double()) : 0.0
    ); 

    // No cell-cell adhesion
    const AdhesionMode adhesion_mode = AdhesionMode::NONE;
    std::unordered_map<std::string, T> adhesion_params;

    // No modulation of ambient viscosity 
    const bool modulate_local_viscosity = false; 
    Array<T, Dynamic, 2> viscosity_lims(2, 2); 
    viscosity_lims << eta_ambient, eta_ambient, 
                      eta_ambient, eta_ambient; 
    const int max_coordination_number = 1; 

    // Parse cell-cell friction parameters
    FrictionMode friction_mode; 
    const int token = json_data["cell_cell_friction_mode"].as_int64(); 
    if (token == 0)
        friction_mode = FrictionMode::NONE; 
    else if (token == 1)
        friction_mode = FrictionMode::KINETIC; 
    else 
        throw std::runtime_error("Invalid cell-cell friction mode specified");

    // Parse cell-cell friction coefficient
    T eta_cell_cell = 0.0; 
    if (friction_mode == FrictionMode::KINETIC)
        eta_cell_cell = static_cast<T>(json_data["eta_cell_cell"].as_double());

    // Set the Coulomb coefficient for cell-cell and cell-surface friction 
    T cell_cell_coulomb_coeff = 1.0;
    T cell_surface_coulomb_coeff = 1.0;
    try 
    {
        cell_surface_coulomb_coeff = static_cast<T>(
            json_data["cell_surface_coulomb_coeff"].as_double()
        ); 
    }
    catch (boost::wrapexcept<boost::system::system_error>& e) { }
    if (friction_mode == FrictionMode::KINETIC)
    {
        try
        {
            cell_cell_coulomb_coeff = static_cast<T>(
                json_data["cell_cell_coulomb_coeff"].as_double()
            ); 
        }
        catch (boost::wrapexcept<boost::system::system_error>& e) { } 
    }

    // Use Verlet integration, if desired 
    IntegrationMode integration_mode = IntegrationMode::HEUN_EULER; 
    T M0 = 0.0; 
    try
    {
        const int token2 = json_data["integration_mode"].as_int64(); 
        if (token2 == 0)
            integration_mode = IntegrationMode::VELOCITY_VERLET; 
        else if (token2 == 1)
            integration_mode = IntegrationMode::HEUN_EULER;
        else if (token2 == 2)
            integration_mode = IntegrationMode::BOGACKI_SHAMPINE; 
        else if (token2 == 3)
            integration_mode = IntegrationMode::RUNGE_KUTTA_FEHLBERG; 
        else if (token2 == 4) 
            integration_mode = IntegrationMode::DORMAND_PRINCE; 
        else 
            throw std::runtime_error("Invalid integration mode specified");
    }
    catch (boost::wrapexcept<boost::system::system_error>& e) { }
    if (integration_mode == IntegrationMode::VELOCITY_VERLET)
    {
        M0 = static_cast<T>(json_data["M0"].as_double());
    } 

    // Parse minimum number of cells at which to start switching, if given
    int n_cells_start_switch = 0; 
    try
    {
        n_cells_start_switch = json_data["n_cells_start_switch"].as_int64(); 
    }
    catch (boost::wrapexcept<boost::system::system_error>& e) { }

    // Vectors of growth rate means and standard deviations (identical for
    // both groups) 
    Array<T, Dynamic, 1> growth_means(2);
    Array<T, Dynamic, 1> growth_stds(2); 
    growth_means << growth_mean, growth_mean; 
    growth_stds << growth_std, growth_std; 

    // Vectors of cell-surface adhesion energy density and/or friction
    // coefficient means and standard deviations
    std::vector<int> group_attributes; 
    Array<T, Dynamic, Dynamic> attribute_values;
    if (surface_friction_mode == 0)
    {
        group_attributes.push_back(__colidx_sigma0);
        attribute_values.resize(2, 1); 
        attribute_values << sigma0_mean1,
                            sigma0_mean2;
    } 
    else if (surface_friction_mode == 1)
    {
        group_attributes.push_back(__colidx_eta1);
        attribute_values.resize(2, 1); 
        attribute_values << eta1_mean1,
                            eta1_mean2;
    }
    else    // surface_friction_mode == 2 
    {
        group_attributes.push_back(__colidx_eta1);
        group_attributes.push_back(__colidx_sigma0); 
        attribute_values.resize(2, 2); 
        attribute_values << eta1_mean1, sigma0_mean1,
                            eta1_mean2, sigma0_mean2;
    }

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
    // Check whether an initial population was specified
    Array<T, Dynamic, Dynamic> cells;
    int ncols = (friction_mode != FrictionMode::NONE ? __ncols_required + 1 : __ncols_required);
    std::string init_filename = "";
    std::string lineage_filename = "";  
    try
    {
        init_filename = json_data["init_filename"].as_string().c_str();
    }
    catch (boost::wrapexcept<boost::system::system_error>& e) { }
    try
    {
        lineage_filename = json_data["lineage_filename"].as_string().c_str(); 
    }
    catch (boost::wrapexcept<boost::system::system_error>& e) { }

    // If an initial population was specified ...
    std::vector<int> parents; 
    if (init_filename.size() > 0)
    {
        auto result = readCells<T>(init_filename); 
        cells = result.first; 
        if (cells.cols() != ncols)
        {
            throw std::runtime_error(
                "File specifying initial population contains incorrect number of "
                "columns"
            ); 
        }
        
        // In this case, check whether a lineage was specified
        if (lineage_filename.size() > 0)
        {
            std::unordered_map<int, int> lineage = readLineage(lineage_filename);
            
            // In this case, check that the cells in the initial population
            // are specified in the lineage 
            for (int i = 0; i < cells.rows(); ++i)
            {
                int id = static_cast<int>(cells(i, __colidx_id)); 
                if (lineage.find(id) == lineage.end())
                {
                    throw std::runtime_error(
                        "Input lineage file does not specify parent of cell "
                        "in initial population"
                    ); 
                } 
            }

            // Define the vector of parent IDs
            parents.resize(lineage.size());  
            for (const auto& pair : lineage)
            {
                int child_id = pair.first; 
                int parent_id = pair.second; 
                parents[child_id] = parent_id; 
            } 
        }
        else 
        {
            throw std::runtime_error(
                "File specifying initial population was specified, but without "
                "accompanying lineage file"
            ); 
        }
    }
    // Otherwise ... 
    else
    { 
        // Define a founder cell at the origin at time zero, parallel to x-axis, 
        // with zero velocity, mean growth rate, and default viscosity and friction
        // coefficients
        cells.resize(1, ncols); 
        T rz = R - pow(sigma0_mean1 * sqrt(R) / (4 * E0), 2. / 3.);
        if (friction_mode != FrictionMode::NONE)
            cells << 0, 0, 0, rz, 1, 0, 0, 0, 0, 0, 0, 0, 0, L0, L0 / 2, 0, growth_mean,
                     eta_ambient, eta1_mean1, sigma0_mean1, 1, eta_cell_cell;
        else
            cells << 0, 0, 0, rz, 1, 0, 0, 0, 0, 0, 0, 0, 0, L0, L0 / 2, 0, growth_mean,
                     eta_ambient, eta1_mean1, sigma0_mean1, 1;
        parents.push_back(-1); 
    }
 
    // Run the simulation
    runSimulation<T>(
        cells, parents, integration_mode, max_iter, n_cells, max_time, R, Rcell,
        L0, Ldiv, E0, Ecell, M0, max_stepsize, min_stepsize, true, outprefix,
        dt_write, iter_update_neighbors, iter_update_stepsize, max_error_allowed,
        min_error, max_tries_update_stepsize, neighbor_threshold, nz_threshold,
        rng_seed, 2, group_attributes, growth_means, growth_stds, attribute_values,
        SwitchMode::MARKOV, switch_rates, switch_timescale, daughter_length_std,
        daughter_angle_xy_bound, daughter_angle_z_bound, max_rxy_noise,
        max_rz_noise, max_nxy_noise, max_nz_noise, basal_only, basal_min_overlap,
        adhesion_mode, adhesion_params, "", "", friction_mode,
        modulate_local_viscosity, viscosity_lims, max_coordination_number, false,
        n_cells_start_switch, false, cell_cell_coulomb_coeff,
        cell_surface_coulomb_coeff, 50 
    ); 
    
    return 0; 
}
