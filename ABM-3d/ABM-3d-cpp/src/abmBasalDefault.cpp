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
 *     2/2/2024
 */

#define EIGEN_DONT_PARALLELIZE    // Disable internal parallelization within Eigen

#include <iostream>
#include <fstream>
#include <stdexcept>
#include <cmath>
#include <omp.h>
#include <Eigen/Dense>
#include <boost/random.hpp>
#include <boost/multiprecision/mpfr.hpp>    // TODO Trying this out?
#include "../include/growth.hpp"
#include "../include/mechanics.hpp"
#include "../include/utils.hpp"

using namespace Eigen;

// Expose math functions for both standard and boost MPFR types
using std::pow;
using boost::multiprecision::pow;
using std::sqrt;
using boost::multiprecision::sqrt;
using std::min;
using boost::multiprecision::min;
using std::max;
using boost::multiprecision::max;
using std::abs;
using boost::multiprecision::abs;

// Define floating-point type to be used 
//typedef boost::multiprecision::number<boost::multiprecision::mpfr_float_backend<20> > T;
typedef double T;

// Maximum number of attempts to control stepsize per Runge-Kutta iteration 
const int max_tries = 3;

// Minimum error per Runge-Kutta iteration
const T min_error = static_cast<T>(1e-30);

// Minimum distance between neighboring cells
const T min_dist = static_cast<T>(1e-8);

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
    const double growth_mean = json_data["growth_mean"].as_double();
    const double growth_std = json_data["growth_std"].as_double(); 
    const T E0 = static_cast<T>(json_data["E0"].as_double());
    const T Ecell = static_cast<T>(json_data["Ecell"].as_double()); 
    const T sigma0 = static_cast<T>(json_data["sigma0"].as_double()); 
    const T eta_ambient = static_cast<T>(json_data["eta_ambient"].as_double()); 
    const T eta_surface = static_cast<T>(json_data["eta_surface"].as_double()); 
    T dt = static_cast<T>(json_data["max_dt"].as_double());    // Can be changed
    const T max_stepsize = dt; 
    const int iter_write = json_data["iter_write"].as_int64(); 
    const int iter_update_stepsize = json_data["iter_update_stepsize"].as_int64(); 
    const int iter_update_neighbors = json_data["iter_update_neighbors"].as_int64(); 
    const T neighbor_threshold = 2 * (2 * R + L0);
    const int n_cells = json_data["n_cells"].as_int64();
    const double daughter_length_std = json_data["daughter_length_std"].as_double();
    const double orientation_conc = json_data["orientation_conc"].as_double();
    const double theta_xy_bound = json_data["max_orientation_angle_xy"].as_double();
    const double theta_z_bound = json_data["max_orientation_angle_z"].as_double();
    const T nz_threshold = static_cast<T>(json_data["nz_threshold"].as_double());
    const T max_error_allowed = static_cast<T>(json_data["max_rungekutta_error"].as_double());

    // Prefactors for cell-cell interaction forces
    const T sqrtR = sqrt(R); 
    const T powRdiff = pow(R - Rcell, 1.5);
    Array<T, 4, 1> cell_cell_prefactors; 
    cell_cell_prefactors << 2.5 * sqrtR,
                            2.5 * E0 * sqrtR,
                            E0 * powRdiff,
                            Ecell;

    // Growth rate distribution function: normal distribution with given mean
    // and standard deviation
    boost::random::normal_distribution<> growth_dist(growth_mean, growth_std);
    std::function<T(boost::random::mt19937&)> growth_dist_func =
        [&growth_dist](boost::random::mt19937& rng)
        {
            return static_cast<T>(growth_dist(rng)); 
        };

    // Daughter cell length ratio distribution function: normal distribution
    // with mean 0.5 and given standard deviation
    boost::random::normal_distribution<> daughter_length_dist(0.5, daughter_length_std); 
    std::function<T(boost::random::mt19937&)> daughter_length_dist_func =
        [&daughter_length_dist](boost::random::mt19937& rng)
        {
            return static_cast<T>(daughter_length_dist(rng)); 
        };

    // Daughter angle distribution functions: two von Mises distributions with 
    // mean 0 and given concentration parameter that are bounded by the given 
    // values 
    boost::random::uniform_01<> uniform_dist; 
    std::function<T(boost::random::mt19937&)> daughter_angle_xy_dist_func,
                                              daughter_angle_z_dist_func;
    if (theta_xy_bound == 0.0)
    {
        daughter_angle_xy_dist_func = [](boost::random::mt19937& rng){ return 0; };
    }
    else
    {
        daughter_angle_xy_dist_func = 
            [&orientation_conc, &uniform_dist, &theta_xy_bound](boost::random::mt19937& rng)
            {
                double theta = vonMises(0.0, orientation_conc, rng, uniform_dist);
                while (theta > theta_xy_bound || theta < -theta_xy_bound)
                    theta = vonMises(0.0, orientation_conc, rng, uniform_dist);
                return static_cast<T>(theta);
            };
    }
    if (theta_z_bound == 0.0)
    {
        daughter_angle_z_dist_func = [](boost::random::mt19937& rng){ return 0; };
    }
    else
    {
        daughter_angle_z_dist_func = 
            [&orientation_conc, &uniform_dist, &theta_z_bound](boost::random::mt19937& rng)
            {
                double theta = vonMises(0.0, orientation_conc, rng, uniform_dist);
                while (theta > theta_z_bound || theta < -theta_z_bound)
                    theta = vonMises(0.0, orientation_conc, rng, uniform_dist);
                return static_cast<T>(theta);
            };
    }

    // Output file prefix
    std::string prefix = argv[2];

    // Initialize simulation ...
    //
    // Define a founder cell at the origin at time zero, parallel to x-axis 
    // and xy-plane, with mean growth rate and default viscosity and friction
    // coefficients
    boost::random::mt19937 rng(std::stoi(argv[3]));
    T t = 0; 
    int i = 0;
    int n = 1;
    Array<T, Dynamic, Dynamic> cells(n, 13);
    cells << 0, 0, 0.99 * R, 1, 0, 0, L0, L0 / 2, 0, growth_mean, eta_ambient, eta_surface, sigma0;
    Array<T, Dynamic, 6> velocities(n, 6);
    velocities << 0, 0, 0, 0, 0, 0;
    
    // Compute initial array of neighboring cells (should be empty)
    Array<T, Dynamic, 7> neighbors = getCellNeighbors<T>(cells, neighbor_threshold, R, Ldiv);

    // Write the founder cell to file
    json_data["t_curr"] = static_cast<double>(t);
    std::stringstream ss_init; 
    ss_init << prefix << "_init.txt";
    std::string filename_init = ss_init.str(); 
    writeCells<T>(cells, json_data, filename_init); 
    
    // Run the simulation ... 
    while (n < n_cells)
    {
        // Divide the cells that have reached division length
        Array<int, Dynamic, 1> to_divide = divideMaxLength<T>(cells, Ldiv);
        if (to_divide.sum() > 0)
            std::cout << "... Dividing " << to_divide.sum() << " cells (iteration " << i
                      << ")" << std::endl;
        cells = divideCells<T>(
            cells, t, R, Rcell, to_divide, growth_dist_func, rng,
            daughter_length_dist_func, daughter_angle_xy_dist_func,
            daughter_angle_z_dist_func
        );

        // Update orientations and neighboring cells if division has occurred
        if (to_divide.sum() > 0)
        {
            normalizeOrientations<T>(cells);
            neighbors = getCellNeighbors<T>(cells, neighbor_threshold, R, Ldiv);
        }

        // Update cell positions and orientations 
        auto result = stepRungeKuttaAdaptiveFromNeighbors<T>(
            A, b, bs, cells, neighbors, dt, R, Rcell, cell_cell_prefactors,
            E0, nz_threshold
        ); 
        Array<T, Dynamic, Dynamic> cells_new = std::get<0>(result);
        Array<T, Dynamic, 6> errors = std::get<1>(result);
        Array<T, Dynamic, 6> velocities_new = std::get<2>(result);

        // If the error is big, retry the step with a smaller stepsize (up to
        // a given maximum number of attempts)
        if (i % iter_update_stepsize == 0)
        {
            T max_error = max(errors.abs().maxCoeff(), min_error); 
            int j = 0; 
            while (max_error > max_error_allowed && j < max_tries)
            {
                dt *= pow(max_error_allowed / max_error, 1.0 / (error_order + 1));
                result = stepRungeKuttaAdaptiveFromNeighbors<T>(
                    A, b, bs, cells, neighbors, dt, R, Rcell, cell_cell_prefactors,
                    E0, nz_threshold
                ); 
                cells_new = std::get<0>(result);
                errors = std::get<1>(result);
                velocities_new = std::get<2>(result);
                max_error = max(errors.abs().maxCoeff(), min_error);
                j++;  
            }
            // If the error is small, increase the stepsize up to a maximum stepsize
            if (max_error < max_error_allowed)
                dt = min(dt * pow(max_error_allowed / max_error, 1.0 / (error_order + 1)), max_stepsize);
        }
        cells = cells_new;
        velocities = velocities_new;

        // Grow the cells
        growCells<T>(cells, dt, R);

        // Pick out only the cells that overlap with the surface, updating 
        // array of neighboring cells whenever a cell is deleted 
        Array<T, Dynamic, 1> max_overlaps = R - cells.col(2) - cells.col(7) * cells.col(5);
        std::vector<int> overlap_idx; 
        for (int j = 0; j < cells.rows(); ++j)
        {
            if (max_overlaps(j) > -0.5 * R)    // Allow for a little room
                overlap_idx.push_back(j);
        }
        if (overlap_idx.size() < cells.rows())
        {
            cells = cells(overlap_idx, Eigen::all).eval();
            neighbors = getCellNeighbors<T>(cells, neighbor_threshold, R, Ldiv);
        }

        // If there is more than one cell, for each cell whose vertical orientation
        // is less than the threshold, randomly fix the vertical orientation with a
        // small probability
        if (cells.rows() > 1)
        {
            for (int j = 0; j < cells.rows(); ++j)
	    {
	        if (abs(cells(j, 5)) < nz_threshold)
		{
		    double r1 = uniform_dist(rng); 
                    double r2 = uniform_dist(rng);
                    T theta = boost::math::constants::pi<T>() / 180;
		    if (r1 < 0.01)
                    {
                        if (r2 < 0.5)
		            cells(j, 5) = -theta;
                        else
                            cells(j, 5) = theta;
                        T norm = cells(j, 3) * cells(j, 3) + cells(j, 4) * cells(j, 4);
		        T alpha = (1 - cells(j, 5) * cells(j, 5)) / norm;
		        cells(j, 3) *= sqrt(alpha);
		        cells(j, 4) *= sqrt(alpha);
                    }
		}
            }
            normalizeOrientations<T>(cells);
        }

        // Update distances between neighboring cells
        updateNeighborDistances<T>(cells, neighbors);

        // Update current time 
        t += dt;
        i += 1;
        n = cells.rows(); 

        // Update neighboring cells 
        if (i % iter_update_neighbors == 0)
            neighbors = getCellNeighbors<T>(cells, neighbor_threshold, R, Ldiv);
        
        // Write the current population to file
        if (i % iter_write == 0)
        {
            std::cout << "Iteration " << i << ": " << n << " cells, time = "
                      << t << ", max error = " << errors.abs().maxCoeff()
                      << ", dt = " << dt << std::endl;
            json_data["t_curr"] = static_cast<double>(t);
            std::stringstream ss; 
            ss << prefix << "_iter" << i << ".txt"; 
            std::string filename = ss.str(); 
            writeCells<T>(cells, json_data, filename); 
        } 
    }

    // Write final population to file
    json_data["t_curr"] = static_cast<double>(t);
    std::stringstream ss_final; 
    ss_final << prefix << "_final.txt";
    std::string filename_final = ss_final.str(); 
    writeCells<T>(cells, json_data, filename_final); 
    
    return 0; 
}
