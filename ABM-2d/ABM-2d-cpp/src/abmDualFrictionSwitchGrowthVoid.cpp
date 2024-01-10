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
 * 10) cell group identifier (1 for high friction, 2 for low friction)
 *
 * Authors:
 *     Kee-Myoung Nam
 *
 * Last updated:
 *     1/10/2024
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

// Upper bound on daughter cell orientation angle 
const double theta_bound = boost::math::constants::pi<double>() / 90; 

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

    // Surface contact area density and powers of cell radius  
    const T surface_contact_density = std::pow(sigma0 * R * R / (4 * E0), 1. / 3.);
    const T sqrtR = std::sqrt(R); 
    const T powRdiff = std::pow(R - Rcell, 1.5);
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
            return growth_dist(rng); 
        };

    // Growth rate distribution function for dead cells 
    std::function<T(boost::random::mt19937&)> void_dist_func = 
        [](boost::random::mt19937& rng)
        {
            return 0;
        };

    // Vector of growth rate distribution functions 
    std::vector<std::function<T(boost::random::mt19937&)> > growth_dist_funcs {
        growth_dist_func, growth_dist_func, void_dist_func
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
            T theta = vonMises<T>(0.0, orientation_conc, rng, uniform_dist);
            while (theta > theta_bound || theta < -theta_bound)
                theta = vonMises<T>(0.0, orientation_conc, rng, uniform_dist);
            return theta;
        };

    // Rates of switching from group 1 to group 2 and vice versa 
    T rate_12 = 1.0 / lifetime_mean1; 
    T rate_21 = 1.0 / lifetime_mean2;

    // Output file prefix
    std::string prefix = argv[2];

    // Maximum number of attempts to control stepsize per iteration 
    int max_tries = 3;

    // Minimum error 
    T min_error = static_cast<T>(1e-30); 

    // Initialize simulation ...
    //
    // Define a founder cell at the origin at time zero, parallel to x-axis, 
    // with mean growth rate and default viscosity and friction coefficients
    boost::random::mt19937 rng(std::stoi(argv[3]));
    T t = 0; 
    int i = 0;
    int n = 1;
    Array<T, Dynamic, Dynamic> cells(n, 11);
    cells << 0, 0, 1, 0, L0, L0 / 2, 0, growth_mean, eta_ambient, eta_mean1, 1;
    Array<T, Dynamic, 4> velocities(n, 4);
    velocities << 0, 0, 0, 0;
    
    // Compute initial array of neighboring cells (should be empty)
    Array<T, Dynamic, 6> neighbors = getCellNeighbors<T>(cells, neighbor_threshold, R, Ldiv);

    // Write the founder cell to file
    json_data["t_curr"] = t;
    std::stringstream ss_init; 
    ss_init << prefix << "_init.txt";
    std::string filename_init = ss_init.str(); 
    writeCells<T>(cells, json_data, filename_init);

    // Indicator for whether the growth void has been introduced
    bool void_introduced = false;

    // Number of cells at which to introduce the growth void 
    int n_cells_introduce_void = static_cast<int>(n_cells / 2); 

    // Number of cells to include within the void 
    int n_cells_within_void = static_cast<int>(n_cells / 4); 
    
    // Run the simulation ... 
    while (n < n_cells)
    {
        // If the void has not yet been introduced and the critical cell 
        // number has been reached ...
        if (n >= n_cells_introduce_void && !void_introduced)
        {
            // Find the center of mass of the biofilm 
            T center_x = cells.col(0).sum() / n;
            T center_y = cells.col(1).sum() / n;

            // Get the distance of each cell to the center of mass
            Array<T, 2, 1> center; 
            center << center_x, center_y;
            std::vector<T> dists_to_center;
            std::vector<int> idx; 
            for (int j = 0; j < n; ++j)
            {
                dists_to_center.push_back((cells(j, Eigen::seq(0, 1)) - center.transpose()).matrix().norm());
                idx.push_back(j);
            }

            // Sort the cells in order of distance to center of mass 
            std::sort(
                idx.begin(), idx.end(),
                [&dists_to_center](int p, int q)
                {
                    return dists_to_center[p] < dists_to_center[q]; 
                }
            );

            // Kill the cells that are nearest to the center of mass 
            for (int j = 0; j < n_cells_within_void; ++j)
            {
                cells(idx[j], 7) = 0;
                cells(idx[j], 10) = 3;
            }

            void_introduced = true;
        } 

        // Divide the cells that have reached division length
        Array<int, Dynamic, 1> to_divide = divideMaxLength<T>(cells, Ldiv);
        if (to_divide.sum() > 0)
            std::cout << "... Dividing " << to_divide.sum() << " cells (iteration " << i
                      << ")" << std::endl;
        cells = divideCells<T>(
            cells, t, R, Rcell, to_divide, growth_dist_funcs, rng, daughter_length_dist_func,
            daughter_angle_dist_func
        );

        // Update the neighboring cells if division has occurred
        if (to_divide.sum() > 0)
        {
            neighbors = getCellNeighbors<T>(cells, neighbor_threshold, R, Ldiv);
            // Check that none of the distances are near zero 
            if ((neighbors(Eigen::all, Eigen::seq(2, 3)).matrix().rowwise().norm().array() < 1e-8).any())
            {
                // Write final population to file and terminate 
                std::stringstream ss_final, ss_error;
                ss_final << prefix << "_finalexcept.txt";
                std::string filename_final = ss_final.str();
                json_data["t_curr"] = t;
                writeCells<T>(cells, json_data, filename_final);
                ss_error << "Encountered near-zero cell-cell distance (iteration " << i << ")" << std::endl;
                throw std::runtime_error(ss_error.str()); 
            }
        }

        // Update cell positions and orientations 
        auto result = stepRungeKuttaAdaptiveFromNeighbors<T>(
            A, b, bs, cells, neighbors, dt, R, Rcell, cell_cell_prefactors,
            surface_contact_density
        ); 
        Array<T, Dynamic, Dynamic> cells_new = std::get<0>(result); 
        Array<T, Dynamic, 4> errors = std::get<1>(result);
        Array<T, Dynamic, 4> velocities_new = std::get<2>(result); 

        // If the error is big, retry the step with a smaller stepsize (up to
        // a given maximum number of attempts)
        if (i % iter_update_stepsize == 0)
        {
            T max_error = std::max(errors.abs().maxCoeff(), min_error); 
            int j = 0; 
            while (max_error > 1e-8 && j < max_tries)
            {
                dt *= std::pow(1e-8 / max_error, 1.0 / (error_order + 1));
                result = stepRungeKuttaAdaptiveFromNeighbors<T>(
                    A, b, bs, cells, neighbors, dt, R, Rcell, cell_cell_prefactors,
                    surface_contact_density
                ); 
                cells_new = std::get<0>(result); 
                errors = std::get<1>(result);
                velocities_new = std::get<2>(result); 
                max_error = std::max(errors.abs().maxCoeff(), min_error); 
                j++;  
            }
            // If the error is small, increase the stepsize up to a maximum stepsize
            if (max_error < 1e-8)
                dt = std::min(dt * std::pow(1e-8 / max_error, 1.0 / (error_order + 1)), 1e-5);
        }
        // Check for any NaN's or infinities
        if (cells_new.isNaN().any() || cells_new.isInf().any())
        {
            // Write final population to file and terminate 
            std::stringstream ss_prev, ss_final, ss_error;
            ss_prev << prefix << "_finalexceptprev.txt"; 
            ss_final << prefix << "_finalexcept.txt";
            std::string filename_prev = ss_prev.str(); 
            std::string filename_final = ss_final.str();
            writeCells<T>(cells, json_data, filename_prev);
            json_data["t_curr"] = t;
            writeCells<T>(cells_new, json_data, filename_final);
            ss_error << "Encountered NaN and/or infinity (iteration " << i << ")" << std::endl; 
            throw std::runtime_error(ss_error.str());
        }
        cells = cells_new;
        velocities = velocities_new;

        // Grow the cells
        growCells<T>(cells, dt, R);

        // Update distances between neighboring cells (and check that no distances
        // are near zero)
        updateNeighborDistances<T>(cells, neighbors);
        if ((neighbors(Eigen::all, Eigen::seq(2, 3)).matrix().rowwise().norm().array() < 1e-8).any())
        {
            // Write final population to file and terminate 
            std::stringstream ss_final, ss_error;
            ss_final << prefix << "_finalexcept.txt";
            std::string filename_final = ss_final.str();
            json_data["t_curr"] = t;
            writeCells<T>(cells, json_data, filename_final);
            ss_error << "Encountered near-zero cell-cell distance (iteration " << i << ")" << std::endl;
            throw std::runtime_error(ss_error.str()); 
        }

        // Update current time 
        t += dt;
        i += 1;
        n = cells.rows(); 

        // Update neighboring cells 
        if (i % iter_update_neighbors == 0)
        {
            neighbors = getCellNeighbors<T>(cells, neighbor_threshold, R, Ldiv);
            // Check that none of the distances are near zero 
            if ((neighbors(Eigen::all, Eigen::seq(2, 3)).matrix().rowwise().norm().array() < 1e-8).any())
            {
                // Write final population to file and terminate 
                std::stringstream ss_final, ss_error;
                ss_final << prefix << "_finalexcept.txt";
                std::string filename_final = ss_final.str();
                json_data["t_curr"] = t;
                writeCells<T>(cells, json_data, filename_final);
                ss_error << "Encountered near-zero cell-cell distance (iteration " << i << ")" << std::endl;
                throw std::runtime_error(ss_error.str()); 
            }
        }

        // Switch cells between groups at the given rates, while preventing dead 
        // cells from switching
        Array<int, Dynamic, 1> to_switch = chooseCellsToSwitch<T>(
            cells, rate_12, rate_21, dt, rng, uniform_dist
        );
        for (int j = 0; j < n; ++j)
        {
            if (cells(j, 10) == 3 && to_switch(j) == 1)
                to_switch(j) = 0;
        }
        switchGroups<T>(cells, 9, to_switch, eta_dist1_func, eta_dist2_func, rng);

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

            // Write the velocities to file as additional attributes
            Array<T, Dynamic, Dynamic> cells_to_write(cells.rows(), cells.cols() + 4);
            cells_to_write(Eigen::all, Eigen::seq(0, cells.cols() - 1)) = cells; 
            cells_to_write(Eigen::all, Eigen::seq(cells.cols(), cells.cols() + 3)) = velocities;
            writeCells<T>(cells_to_write, json_data, filename); 
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
