/**
 * Test module for JKR contact force calculations. 
 *
 * Authors:
 *     Kee-Myoung Nam
 *
 * Last updated:
 *     6/7/2025
 */
#include <iostream>
#include <fstream>
#include <string>
#include <iomanip>
#include <utility>
#include <Eigen/Dense>
#include <boost/multiprecision/mpfr.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "../../include/adhesion.hpp"
#include "../../include/jkr.hpp"

using boost::multiprecision::sqrt;
using boost::multiprecision::pow; 

typedef boost::multiprecision::number<boost::multiprecision::mpfr_float_backend<100> > PreciseType; 

/**
 * Calculate the JKR force between two spheres at various overlap distances,
 * and write the resulting force vs overlap dependence to an output file. 
 */
template <typename T>
void jkrForceVsOverlap(const T R, const T E, const T gamma, const T dmin, 
                       const T dmax, const std::string& outfilename)
{
    const int n = 1000;
    const T imag_tol = 1e-8; 
    const T aberth_tol = 1e-20; 
    const double force_tol = 1e-8; 

    // Compute the contact radius for a range of overlaps 
    Matrix<T, Dynamic, 1> delta = Matrix<T, Dynamic, 1>::LinSpaced(n, dmin, dmax);
    Matrix<T, Dynamic, 2> radii(n, 2); 
    for (int i = 0; i < n; ++i)
    {
        std::pair<T, T> radii_i = jkrContactRadius<T>(
            delta(i), R, E, gamma, imag_tol, aberth_tol
        );
        radii(i, 0) = radii_i.first; 
        radii(i, 1) = radii_i.second;  
    }

    // Compute the corresponding forces 
    Matrix<T, Dynamic, 2> forces(n, 2);

    // Define a set of dummy orientations and centerline coordinates 
    Matrix<T, 3, 1> n1, n2, d12; 
    n1 << 1, 0, 0; 
    n2 << 1, 0, 0;
    T half_l = 0.5; 
    T s = half_l; 
    T t = -half_l;

    // For each overlap ... 
    for (int i = 0; i < n; ++i)
    {
        d12 << 2 * R - delta(i), 0, 0;

        // ... and each of the two possible contact radii ... 
        for (int j = 0; j < 2; ++j)
        {
            // ... calculate the Hertz and JKR forces 
            T f_hertz = (4. / 3.) * E * radii(i, j) * radii(i, j) * radii(i, j) / R;
            T f_jkr = 4 * sqrt(boost::math::constants::pi<T>() * gamma * E) * pow(radii(i, j), 1.5);

            // For the larger radius, if the overlap is positive, compare the
            // JKR force against that calculated using forcesIsotropicJKRLangrange()
            if (j == 1 && d12.norm() <= 2 * R)
            {
                Array<T, 2, 6> forces_jkr = forcesIsotropicJKRLagrange<T, 3>(
                    n1, n2, d12, R, E, gamma, s, t, false, imag_tol, aberth_tol
                );
                REQUIRE_THAT(
                    static_cast<double>(-forces_jkr(0, 0)), 
                    Catch::Matchers::WithinAbs(static_cast<double>(f_jkr), force_tol)
                );
            } 
            forces(i, j) = f_hertz - f_jkr; 
        }  
    }

    // Write the contact radii and forces to file 
    std::ofstream outfile(outfilename);
    outfile << std::setprecision(10);  
    for (int i = 0; i < n; ++i)
        outfile << delta(i) << '\t' << radii(i, 0) << '\t' << radii(i, 1) << '\t'
                << forces(i, 0) << '\t' << forces(i, 1) << std::endl;
    outfile.close();  
}

TEST_CASE(
    "Calculate JKR force vs overlap dependence",
    "[jkrContactRadius(), forcesIsostropicJKRLagrange()]"
)
{
    jkrForceVsOverlap<PreciseType>(
        0.8, 3900.0, 200.0, -0.2, 0.6, "jkr_force_vs_overlap_200.txt"
    ); 
}

