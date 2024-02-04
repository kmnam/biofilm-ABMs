/**
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
 *     2/4/2024
 */

#include <iostream>
#include <Eigen/Dense>
#include <boost/math/constants/constants.hpp>
#include "../../include/simulation.hpp"

int main()
{
    // Define system and simulation parameters
    const double half_l = 0.5;
    const double L0 = 2 * half_l;
    const double R = 0.8;
    const double Rcell = 0.35;
    const double E0 = 3900.0;
    const double Ecell = 390000.0;
    const double eta_ambient = 0.072;
    const double eta_surface = 720.0;
    const double sigma0 = 100.0;
    const double Ldiv = 2 * (R + L0);
    const double neighbor_threshold = 2 * (2 * R + L0);
    const int max_iter = 10000;
    const double max_stepsize = 1e-6;
    const bool write = false;
    const std::string outprefix = "";
    const int iter_write = 10 * max_iter;
    const int iter_update_neighbors = 10 * max_iter;
    const int iter_update_stepsize = 5;
    const double max_error_allowed = 1e-10;
    const double min_error = 1e-30;
    const double max_tries_update_stepsize = 3;
    const double nz_threshold = 1e-10;
    const int rng_seed = 1234567890;
    const double growth_mean = 0.0;
    const double growth_std = 0.0;
    const double daughter_length_std = 0.0; 
    const double daughter_angle_xy_bound = 0.0;
    const double daughter_angle_z_bound = 0.0;

    // Define a horizontal cell in which the cell is just touching the surface
    Array<double, 3, 1> r, n;
    r << 0.0, 0.0, 0.95 * R;
    n << 1.0, 0.0, 0.0;
    Array<double, Dynamic, Dynamic> cells(1, 13);
    cells << r(0), r(1), r(2), n(0), n(1), n(2),
             L0, half_l, 0.0, growth_mean, eta_ambient, eta_surface, sigma0;
    std::cout << cells << std::endl;

    // Run simulation
    cells = runSimulation<double>(
        cells, max_iter, -1, R, Rcell, L0, Ldiv, E0, Ecell, max_stepsize,
        write, outprefix, iter_write, iter_update_neighbors,
        iter_update_stepsize, max_error_allowed, min_error,
        max_tries_update_stepsize, neighbor_threshold, nz_threshold, rng_seed,
        growth_mean, growth_std, daughter_length_std, daughter_angle_xy_bound,
        daughter_angle_z_bound
    ); 

    std::cout << cells << std::endl;
    std::cout << "Final z-coordinate of cell center: " << cells(0, 2) << std::endl;
    double adhesion = sigma0 / (R * E0);
    double theoretical = R * (1 - std::pow(0.25 * adhesion, 2./3.));
    std::cout << "Theoretical value: " << theoretical << std::endl;
}
