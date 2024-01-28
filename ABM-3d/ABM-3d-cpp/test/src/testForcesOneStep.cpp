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
#include "../../include/growth.hpp"
#include "../../include/mechanics.hpp"
#include "../../include/utils.hpp"

using namespace Eigen;

void computeForces(const Ref<const Array<double, 3, 1> >& r,
                   const Ref<const Array<double, 3, 1> >& n)
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
    const double max_error_allowed = 1e-8;
    const double max_tries = 3;
    const double max_stepsize = 1e-6;
    const double noise_strength = 1e-6;
    const double nz_threshold = 1e-10;
    double dt = 1e-6;

    // Initialize array containing a single cell
    Array<double, Dynamic, Dynamic> cells(1, 12);
    cells << r(0), r(1), r(2), n(0), n(1), n(2),
             2 * half_l, half_l, 0.0, growth_rate, eta_ambient, eta_surface,
    std::cout << cells << std::endl;

    // Prefactors for cell-cell interaction forces
    const double sqrtR = std::sqrt(R); 
    const double powRdiff = std::pow(R - Rcell, 1.5);
    Array<double, 4, 1> cell_cell_prefactors; 
    cell_cell_prefactors << 2.5 * sqrtR,
                            2.5 * E0 * sqrtR,
                            E0 * powRdiff,
                            Ecell;

    // Compute forces
    Array<double, Dynamic, 1> ss = (R - cells.col(2)) / cells.col(5);
    Array<double, Dynamic, 2> cell_surface_repulsion_forces = cellSurfaceRepulsionForces<double>(
        cells, ss, R, E0, nz_threshold
    );
    std::cout << cell_surface_repulsion_forces << "\n--\n";
    Array<double, Dynamic, 2> cell_surface_adhesion_forces = cellSurfaceAdhesionForces<double>(
        cells, ss, R, sigma0, nz_threshold
    );
    std::cout << cell_surface_adhesion_forces << "\n--\n";

    // Compute viscosity force matrix
    Array<double, 6, 6> viscosity_forces = compositeViscosityForceMatrix<double>(
        cells(0, 2), cells(0, 5), cells(0, 6), cells(0, 7), ss(0), eta_ambient,
        eta_surface, R, nz_threshold
    );
    std::cout << viscosity_forces << "\n--\n";
}

int main()
{
    // Compute forces for a cell that is just touching the surface at various
    // angles with the xy-plane
    const double R = 0.8;
    const double half_l = 0.5;
    std::vector<double> c{5.0, 10.0, 90.0};
    for (const double x : c)
    {
        Array<double, 3, 1> r, n;
        const double theta = (x / 180.0) * boost::math::constants::pi<double>();
        r << 0.0, 0.0, 0.95 * (R + half_l * std::sin(theta));
        n << std::cos(theta), 0.0, -std::sin(theta);
        computeForces(r, n);
    }
}
