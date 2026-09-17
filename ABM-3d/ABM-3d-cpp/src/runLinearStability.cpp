/**
 * Authors:
 *     Kee-Myoung Nam
 *
 * Last updated:
 *     9/17/2026
 */

#include <iostream>
#include <fstream>
#include <iomanip>
#include <Eigen/Dense>
#include "../include/linearStability.hpp"

using namespace Eigen;

typedef double T; 

int main(int argc, char** argv)
{
    const T R = 0.8; 
    const T Rcell = 0.5;
    const T E0 = 3900.0;
    const T Ecell = 100 * E0;
    const T sigma0 = 100.0;
    const T eta0 = 0.1872;  
    const T eta1 = 720.0;

    // A gamma value of ~186.21 kg/hr^2 yields an equilibrium overlap of
    // 300 nm (= 0.3 microns)
    //
    // So vary gamma from 0 to 200  
    Matrix<T, Dynamic, 1> gamma = Matrix<T, Dynamic, 1>::LinSpaced(100, 0.0, 200.0);

    // Vary sigma0 from its wild-type value to 20-fold less than that 
    Matrix<T, Dynamic, 1> log_factors = Matrix<T, Dynamic, 1>::LinSpaced(100, 0.0, log2(20));  
    Matrix<T, Dynamic, 1> factors = Eigen::pow(2, log_factors.array()).matrix(); 

    // Vary cell length from 1 micron to 2 microns  
    Matrix<T, Dynamic, 1> lengths = Matrix<T, Dynamic, 1>::LinSpaced(5, 1.0, 2.0);

    // Allow for critical overlaps between 0.001 and 0.6 microns 
    const T delta_min = 1e-3; 
    const T delta_max = 2 * (R - Rcell);  
    const T brent_tol = 1e-8;
    const T brent_max_iter = 1000;

    // Open output file and write header 
    std::ofstream outfile(argv[1]);
    outfile << std::setprecision(10); 
    outfile << "# R = " << R << std::endl
            << "# Rcell = " << Rcell << std::endl
            << "# E0 = " << E0 << std::endl
            << "# Ecell = " << Ecell << std::endl 
            << "# eta0 = " << eta0 << std::endl
            << "# delta_min = " << delta_min << std::endl
            << "# delta_max = " << delta_max << std::endl
            << "# brent_tol = " << brent_tol << std::endl 
            << "# brent_max_iter = " << brent_max_iter << std::endl; 

    // For each cell length and cell-surface adhesion energy value ...  
    for (int k = 0; k < lengths.size(); ++k)
    {
        for (int j = 0; j < factors.size(); ++j)
        {
            // Get the eight-neighbor configuration with the maximum overlap 
            Matrix<T, Dynamic, 6> coords = getEightNeighborConfiguration<T>(
                delta_max, 0.0, lengths(k), R, Rcell, E0, sigma0 / factors(j)
            );
            Matrix<T, Dynamic, 2> neighbors_rxy = coords(Eigen::seqN(1, coords.rows() - 1), Eigen::seqN(0, 2)); 
            Matrix<T, Dynamic, 2> neighbors_nxy = coords(Eigen::seqN(1, coords.rows() - 1), Eigen::seqN(3, 2)); 

            for (int i = 0; i < gamma.size(); ++i)
            { 
                // Get the stability eigenvalues corresponding to the maximum overlap 
                Matrix<T, 2, 1> delta_max_eigenvalues = get2DStability<T>(
                    neighbors_rxy, neighbors_nxy, lengths(k), R, Rcell, E0,
                    Ecell, sigma0 / factors(j), eta0, eta1 / factors(j),
                    gamma(i)
                ); 
                
                // Is the maximum eigenvalue greater than zero?
                T delta_crit; 
                if (delta_max_eigenvalues.maxCoeff() > 0)
                {
                    // Then calculate the critical overlap 
                    delta_crit = findCriticalOverlapEightNeighbor<T>(
                        lengths(k), R, Rcell, E0, Ecell, sigma0 / factors(j),
                        eta0, eta1 / factors(j), gamma(i), delta_min, delta_max,
                        brent_tol, brent_max_iter, false
                    );
                }
                else    // Otherwise, set to NaN
                {
                    delta_crit = std::numeric_limits<T>::quiet_NaN(); 
                }

                // Write the output file 
                outfile << lengths(k) << '\t' 
                        << gamma(i) << '\t'
                        << sigma0 / factors(j) << '\t'
                        << eta1 / factors(j) << '\t'
                        << delta_crit << std::endl; 
            }
        }
    }
    outfile.close(); 

    return 0;
}
