/**
 * An agent-based model that switches cells between two states that exhibit
 * different distributions of growth rates. 
 *
 * In what follows, a population of N cells is represented as a 2-D array of 
 * size (N, 10), where each row represents a cell and stores the following data:
 * 
 * 0) x-coordinate of cell center
 * 1) y-coordinate of cell center
 * 2) x-coordinate of cell orientation vector
 * 3) y-coordinate of cell orientation vector
 * 4) cell length (excluding caps) 
 * 5) timepoint at which the cell was formed
 * 6) cell growth rate
 * 7) cell's ambient viscosity with respect to surrounding fluid
 * 8) cell-surface friction coefficient
 * 9) cell group identifier (1 for high friction, 2 for low friction)
 *
 * Authors:
 *     Kee-Myoung Nam
 *
 * Last updated:
 *     10/26/2023
 */

#include <iostream>
#include <fstream>
#include <stdexcept>
#include <cmath>
#include <Eigen/Dense>
#include <boost/random.hpp>
#include "../include/growth.hpp"
#include "../include/mechanics.hpp"
#include "../include/switch.hpp"
#include "../include/utils.hpp"

using namespace Eigen;

// Define floating-point type to be used 
typedef double T; 

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
    const T growth_mean = static_cast<T>(json_data["growth_mean"].as_double());
    const T growth_std = static_cast<T>(json_data["growth_std"].as_double());
    const T eta_mean1 = static_cast<T>(json_data["eta_mean1"].as_double());
    const T eta_std1 = static_cast<T>(json_data["eta_std1"].as_double()); 
    const T eta_mean2 = static_cast<T>(json_data["eta_mean2"].as_double());
    const T eta_std2 = static_cast<T>(json_data["eta_std2"].as_double());
    const T lifetime_mean1 = static_cast<T>(json_data["lifetime_mean1"].as_double()); 
    const T lifetime_mean2 = static_cast<T>(json_data["lifetime_mean2"].as_double()); 
    const T E0 = static_cast<T>(json_data["E0"].as_double());
    const T Ecell = static_cast<T>(json_data["Ecell"].as_double()); 
    const T sigma0 = static_cast<T>(json_data["sigma0"].as_double()); 
    const T eta_ambient = static_cast<T>(json_data["eta_ambient"].as_double()); 
    T dt = static_cast<T>(json_data["dt"].as_double());    // Can be changed 
    const int iter_write = json_data["iter_write"].as_int64(); 
    const int iter_update_stepsize = json_data["iter_update_stepsize"].as_int64(); 
    const int iter_update_neighbors = json_data["iter_update_neighbors"].as_int64(); 
    const T neighbor_threshold = 2 * (2 * R + L0);
    const int n_cells = json_data["n_cells"].as_int64();
    const T daughter_length_std = static_cast<T>(json_data["daughter_length_std"].as_double());
    const T orientation_conc = static_cast<T>(json_data["orientation_conc"].as_double());

    // Surface contact area density 
    const T surface_contact_density = std::pow(sigma0 * R * R / (4 * E0), 1. / 3.);

    // Growth rate distribution function: normal distribution with given mean
    // and standard deviation
    boost::random::normal_distribution<> growth_dist(growth_mean, growth_std);
    std::function<T(boost::random::mt19937&)> growth_dist_func =
        [&growth_dist](boost::random::mt19937& rng)
        {
            return growth_dist(rng); 
        };

    // Friction coefficient distribution functions: normal distributions with
    // given mean and standard deviation 
    boost::random::normal_distribution<> eta_dist1(eta_mean1, eta_std1);
    boost::random::normal_distribution<> eta_dist2(eta_mean2, eta_std2); 
    std::function<T(boost::random::mt19937&)> eta_dist1_func =
        [&eta_dist1](boost::random::mt19937& rng)
        {
            return eta_dist1(rng); 
        };
    std::function<T(boost::random::mt19937&)> eta_dist2_func =
        [&eta_dist2](boost::random::mt19937& rng)
        {
            return eta_dist2(rng);
        };

    // Daughter cell length ratio distribution function: normal distribution
    // with mean 0.5 and given standard deviation
    boost::random::normal_distribution<> daughter_length_dist(0.5, daughter_length_std); 
    std::function<T(boost::random::mt19937&)> daughter_length_dist_func =
        [&daughter_length_dist](boost::random::mt19937& rng)
        {
            return daughter_length_dist(rng); 
        };

    // Daughter angle distribution function: von Mises distribution with 
    // mean 0 and given concentration parameter
    boost::random::uniform_01<> uniform_dist;  
    std::function<T(boost::random::mt19937&)> daughter_angle_dist_func =
        [&orientation_conc, &uniform_dist](boost::random::mt19937& rng)
        {
            return vonMises(0.0, orientation_conc, rng, uniform_dist); 
        };

    // Rates of switching from group 1 to group 2 and vice versa 
    T rate_12 = 1.0 / lifetime_mean1; 
    T rate_21 = 1.0 / lifetime_mean2;

    // Output file prefix
    std::string prefix = argv[2];

    // Maximum number of attempts to control stepsize per iteration 
    int max_tries = 3;

    // Initialize simulation ...
    //
    // Define a founder cell at the origin at time zero, parallel to x-axis, 
    // with mean growth rate and default viscosity and friction coefficients
    boost::random::mt19937 rng(std::stoi(argv[3]));
    T t = 0; 
    int i = 0;
    int n = 1;
    Array<T, Dynamic, Dynamic> cells(n, 10);
    cells << 0, 0, 1, 0, L0, 0, growth_mean, eta_ambient, eta_mean1, 1;
    
    // Compute initial array of neighboring cells (should be empty)
    Array<T, Dynamic, 6> neighbors = getCellNeighbors<T>(cells, neighbor_threshold, R, Ldiv);

    // Write the founder cell to file
    json_data["t_curr"] = t;
    std::stringstream ss_init; 
    ss_init << prefix << "_init.txt";
    std::string filename_init = ss_init.str(); 
    writeCells<T>(cells, json_data, filename_init); 
    
    // Run the simulation ... 
    while (n < n_cells)
    {
        // Divide the cells that have reached division length
        Array<int, Dynamic, 1> to_divide = divideMaxLength<T>(cells, Ldiv);
        cells = divideCells<T>(
            cells, t, R, to_divide, growth_dist_func, rng, daughter_length_dist_func,
            daughter_angle_dist_func
        );

        // Update the neighboring cells if division has occurred
        if (to_divide.sum() > 0)
            neighbors = getCellNeighbors<T>(cells, neighbor_threshold, R, Ldiv);

        // Update cell positions and orientations 
        auto result = stepRungeKuttaAdaptiveFromNeighbors<T>(
            A, b, bs, cells, neighbors, dt, R, Rcell, E0, Ecell,
            surface_contact_density
        ); 
        Array<T, Dynamic, Dynamic> cells_new = result.first; 
        Array<T, Dynamic, 4> errors = result.second;

        // If the error is big, retry the step with a smaller stepsize (up to
        // a given maximum number of attempts)
        if (i % iter_update_stepsize == 0)
        {
            T max_error = std::max(errors.abs().maxCoeff(), 1e-100); 
            int j = 0; 
            while (max_error > 1e-8 && j < max_tries)
            {
                dt *= std::pow(1e-8 / max_error, 1.0 / (error_order + 1));
                result = stepRungeKuttaAdaptiveFromNeighbors<T>(
                    A, b, bs, cells, neighbors, dt, R, Rcell, E0, Ecell,
                    surface_contact_density
                ); 
                cells_new = result.first; 
                errors = result.second;
                max_error = std::max(errors.abs().maxCoeff(), 1e-100); 
                j++;  
            }
            // If the error is small, increase the stepsize up to a maximum stepsize
            if (max_error < 1e-8)
                dt = std::min(dt * std::pow(1e-8 / max_error, 1.0 / (error_order + 1)), 1e-5);
        }
        cells = cells_new;

        // Grow the cells
        growCells<T>(cells, dt, R);

        // Update distances between neighboring cells
        updateNeighborDistances<T>(cells, neighbors);

        // Update current time 
        t += dt;
        i += 1;
        n = cells.rows(); 

        // Update neighboring cells 
        if (i % iter_update_neighbors == 0)
            neighbors = getCellNeighbors<T>(cells, neighbor_threshold, R, Ldiv);

        // Switch cells between groups at the given rates
        Array<int, Dynamic, 1> to_switch = chooseCellsToSwitch<T>(
            cells, rate_12, rate_21, dt, rng, uniform_dist
        );
        switchGroups<T>(cells, 8, to_switch, eta_dist1, eta_dist2, rng);

        // Check for any NaN's or infinities
        if (cells.isNaN().any() || cells.isInf().any())
        {
            // Write final population to file and terminate  
            json_data["t_curr"] = t;
            std::stringstream ss_final; 
            ss_final << prefix << "_finalexcept.txt";
            std::string filename_final = ss_final.str(); 
            writeCells<T>(cells, json_data, filename_final); 
            throw std::runtime_error("Encountered NaN and/or infinity");  
        }

        // Write the current population to file
        if (i % iter_write == 0)
        {
            std::cout << "Iteration " << i << ": " << n << " cells, time = "
                      << t << ", max error = " << errors.abs().maxCoeff()
                      << ", dt = " << dt << std::endl;
            json_data["t_curr"] = t;
            std::stringstream ss; 
            ss << prefix << "_iter" << i << ".txt"; 
            std::string filename = ss.str(); 
            writeCells<T>(cells, json_data, filename); 
        } 
    }

    // Write final population to file
    json_data["t_curr"] = t;
    std::stringstream ss_final; 
    ss_final << prefix << "_final.txt";
    std::string filename_final = ss_final.str(); 
    writeCells<T>(cells, json_data, filename_final); 
    
    return 0; 
}
