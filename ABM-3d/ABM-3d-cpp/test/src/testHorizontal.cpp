/**
 * In what follows, a population of N cells is represented as a 2-D array of 
 * size (N, 12), where each row represents a cell and stores the following data:
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
 *
 * Authors:
 *     Kee-Myoung Nam
 *
 * Last updated:
 *     1/28/2024
 */

#include <iostream>
#include <stdexcept>
#include <cmath>
#include <Eigen/Dense>
#include <boost/math/constants/constants.hpp>
#include <boost/random.hpp>
#include "../../include/mechanics.hpp"
#include "../../include/utils.hpp"

int main()
{
    // Define system and simulation parameters
    const double half_l = 0.5;
    const double R = 0.8;
    const double Rcell = 0.35;
    const double E0 = 3900.0;
    const double Ecell = 390000.0;
    const double eps = 0.01;
    const double growth_rate = 1.5;
    const double eta_ambient = 0.072;
    const double eta_surface = 720.0;
    const double sigma0 = 100.0;
    double Ldiv = 2 * R + 4 * half_l;
    double neighbor_threshold = 4 * R + 4 * half_l;

    // Define Butcher tableau for order 3(2) Runge-Kutta method by Bogacki
    // and Shampine 
    Array<double, Dynamic, Dynamic> A(4, 4); 
    A << 0,     0,     0,     0,
         1./2., 0,     0,     0,
         0,     3./4., 0,     0,
         2./9., 1./3., 4./9., 0;
    Array<double, Dynamic, 1> b(4);
    b << 2./9., 1./3., 4./9., 0;
    Array<double, Dynamic, 1> bs(4); 
    bs << 7./24., 1./4., 1./3., 1./8.;
    double error_order = 2;

    // Define additional simulation parameters
    const double min_error = 1e-30;
    const double max_error_allowed = 1e-10;
    const double max_tries = 3;
    const double max_stepsize = 1e-6;
    const double nz_threshold = 1e-10;
    double dt = 1e-6;

    // Define a horizontal cell in which the cell is just touching the surface
    Array<double, 3, 1> r, n;
    r << 0.0, 0.0, 0.95 * R;
    n << 1.0, 0.0, 0.0;
    Array<double, Dynamic, Dynamic> cells(1, 12);
    cells << r(0), r(1), r(2), n(0), n(1), n(2),
             2 * half_l, half_l, 0.0, growth_rate, eta_ambient, eta_surface,
    std::cout << cells << std::endl;
    Array<double, Dynamic, 6> velocities = Array<double, Dynamic, 6>::Zero(1, 6);

    // Compute initial array of neighboring cells (should be empty)
    Array<double, Dynamic, 7> neighbors = getCellNeighbors<double>(cells, neighbor_threshold, R, Ldiv);

    // Prefactors for cell-cell interaction forces
    const double sqrtR = std::sqrt(R); 
    const double powRdiff = std::pow(R - Rcell, 1.5);
    Array<double, 4, 1> cell_cell_prefactors; 
    cell_cell_prefactors << 2.5 * sqrtR,
                            2.5 * E0 * sqrtR,
                            E0 * powRdiff,
                            Ecell;

    // Net force noise distribution: uniform distribution centered at zero
    boost::random::uniform_01<> uniform_dist;
    std::function<double(boost::random::mt19937&)> noise_dist_func =
        [&uniform_dist](boost::random::mt19937& rng)
        {
            return -1 + 2 * uniform_dist(rng);
        };

    // Run 10000 Runge-Kutta iterations ...
    double t = 0;
    boost::random::mt19937 rng(1234567890);
    for (int i = 0; i < 10000; ++i)
    {
        // Update cell positions and orientations 
        auto result = stepRungeKuttaAdaptiveFromNeighbors<double>(
            A, b, bs, cells, neighbors, dt, R, Rcell, cell_cell_prefactors,
            E0, sigma0, nz_threshold, rng, noise_dist_func, 0.0
        ); 
        Array<double, Dynamic, Dynamic> cells_new = std::get<0>(result);
        Array<double, Dynamic, 6> errors = std::get<1>(result);
        Array<double, Dynamic, 6> velocities_new = std::get<2>(result);

        // If the error is big, retry the step with a smaller stepsize (up to
        // a given maximum number of attempts)
        double max_error = std::max(errors.abs().maxCoeff(), min_error); 
        int j = 0; 
        while (max_error > max_error_allowed && j < max_tries)
        {
            dt *= std::pow(max_error_allowed / max_error, 1.0 / (error_order + 1));
            result = stepRungeKuttaAdaptiveFromNeighbors<double>(
                A, b, bs, cells, neighbors, dt, R, Rcell, cell_cell_prefactors,
                E0, sigma0, nz_threshold, rng, noise_dist_func, 0.0
            ); 
            cells_new = std::get<0>(result);
            errors = std::get<1>(result);
            velocities_new = std::get<2>(result);
            max_error = std::max(errors.abs().maxCoeff(), min_error);
            j++;  
        }
        // If the error is small, increase the stepsize up to a maximum stepsize
        if (max_error < max_error_allowed)
            dt = std::min(dt * std::pow(max_error_allowed / max_error, 1.0 / (error_order + 1)), max_stepsize);
        cells = cells_new;
        velocities = velocities_new;
        if (cells.isNaN().any())
        {
            std::stringstream ss;
            ss << "iteration " << i << ": found nan\n";
            throw std::runtime_error(ss.str()); 
        }

        // Update current time 
        t += dt;

        // Print iteration details
        if (i % 1000 == 0)
        {
            std::cout << "Iteration " << i << ": " << cells.rows() << " cells, time = "
                      << t << ", max error = " << errors.abs().maxCoeff()
                      << ", dt = " << dt << std::endl;
            std::cout << "z-coordinate of cell center: " << cells(0, 2) << std::endl;
        }
    }

    std::cout << "Final z-coordinate of cell center: " << cells(0, 2) << std::endl;
    double adhesion = sigma0 / (R * E0);
    double theoretical = R * (1 - std::pow(0.25 * adhesion, 2./3.));
    std::cout << "Theoretical value: " << theoretical << std::endl;
}
