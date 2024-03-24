/**
 * Authors:
 *     Kee-Myoung Nam
 *
 * Last updated:
 *     3/18/2024
 */

#include <iostream>
#include <cmath>
#include <complex>
#include <Eigen/Dense>
#include <boost/math/constants/constants.hpp>
#include <boost/random.hpp>
#include "../../include/polynomials.hpp"
#include "../../include/homotopies.hpp"
#include "../../include/utils.hpp"

/**
 * Solve the univariate cubic polynomial -2*x^3 - 5*x^2 + 4*z + 1 by homotopy 
 * continuation from the polynomial x^3 + 1.
 */
void testSolveCubic()
{
    // Sample a random complex number
    boost::random::mt19937 rng(42);
    boost::random::uniform_01<> dist;
    std::complex<double> gamma(dist(rng), dist(rng)); 

    // Define the start polynomial system (as a bivariate polynomial system
    // with one polynomial being a dummy linear polynomial)
    Matrix<std::complex<double>, Dynamic, 1> g1_coefs(4);
    Matrix<std::complex<double>, Dynamic, 1> g2_coefs(2);
    g1_coefs << 1, 0, 0, 1;
    g2_coefs << -1, 1;
    g1_coefs *= gamma; 
    g2_coefs *= gamma;
    auto g1_coefmap = vectorToMap<std::complex<double>, 2>(g1_coefs, 0);
    auto g2_coefmap = vectorToMap<std::complex<double>, 2>(g2_coefs, 1);
    MultivariatePolynomial<double, 2> g1(g1_coefmap);
    MultivariatePolynomial<double, 2> g2(g2_coefmap);
    std::array<MultivariatePolynomial<double, 2>, 2> g { g1, g2 };

    // Define the end polynomial system (as a bivariate polynomial system
    // with one polynomial being a dummy linear polynomial)
    Matrix<std::complex<double>, Dynamic, 1> f1_coefs(4);
    Matrix<std::complex<double>, Dynamic, 1> f2_coefs(2);
    f1_coefs << 1, 4, -5, -2;
    f2_coefs << -1, 1;
    auto f1_coefmap = vectorToMap<std::complex<double>, 2>(f1_coefs, 0);
    auto f2_coefmap = vectorToMap<std::complex<double>, 2>(f2_coefs, 1);
    MultivariatePolynomial<double, 2> f1(f1_coefmap);
    MultivariatePolynomial<double, 2> f2(f2_coefmap);
    std::array<MultivariatePolynomial<double, 2>, 2> f { f1, f2 };
    
    // Get the roots of the start system
    Matrix<std::complex<double>, Dynamic, 2> g_roots(3, 2);
    g_roots.col(0) = -rootsOfUnity<double>(3);
    g_roots(0, 1) = std::complex<double>(1, 0);
    g_roots(1, 1) = std::complex<double>(1, 0);
    g_roots(2, 1) = std::complex<double>(1, 0);

    // Get the Butcher tableau for Euler's method 
    auto tableau = eulerTableau<double>(); 
    Matrix<double, 1, 1> rkA = std::get<0>(tableau); 
    Matrix<double, 1, 1> rkb = std::get<1>(tableau); 
    Matrix<double, 1, 1> rkc = std::get<2>(tableau);

    // Get the roots of the end system
    ProjectiveStraightLineHomotopy<double, 2, 1> h(g, f, g_roots, rkA, rkb, rkc);
    std::cout << h.solve(1e-8, 1e-8, 5, 1e-3, 0.01) << std::endl;
}

/**
 * Solve for the intersection of two circles by homotopy continuation from the
 * total-degree start system (x^2 - 1, y^2 - 1).
 */
void testSolveCircleIntersectEuler()
{
    // Sample a random complex number
    boost::random::mt19937 rng(42);
    boost::random::uniform_01<> dist;
    std::complex<double> gamma(dist(rng), dist(rng)); 

    // Define the start polynomial system
    MultivariatePolynomial<double, 2> g1 = polynomialOfUnity<double, 2>(0, 2);
    MultivariatePolynomial<double, 2> g2 = polynomialOfUnity<double, 2>(1, 2);
    g1 *= gamma; 
    g2 *= gamma;
    std::array<MultivariatePolynomial<double, 2>, 2> g { g1, g2 };

    // Define the end polynomial system
    double d = 1.5;
    Matrix<std::complex<double>, Dynamic, Dynamic> f1_coefs(3, 3);
    Matrix<std::complex<double>, Dynamic, Dynamic> f2_coefs(3, 3);
    f1_coefs << -1, 0, 1,
                 0, 0, 0,
                 1, 0, 0;
    f2_coefs << d*d - 1, 0, 1,
                   -2*d, 0, 0,
                      1, 0, 0;
    auto f1_coefmap = matrixToMap<std::complex<double>, 2>(f1_coefs, 0, 1);
    auto f2_coefmap = matrixToMap<std::complex<double>, 2>(f2_coefs, 0, 1);
    MultivariatePolynomial<double, 2> f1(f1_coefmap);
    MultivariatePolynomial<double, 2> f2(f2_coefmap);
    std::array<MultivariatePolynomial<double, 2>, 2> f { f1, f2 };

    // Get the roots of the start system
    Matrix<std::complex<double>, Dynamic, 2> g_roots = rootsOfUnity<double>(2, 2);

    // Get the Butcher tableau for Euler's method 
    auto tableau = eulerTableau<double>(); 
    Matrix<double, 1, 1> rkA = std::get<0>(tableau); 
    Matrix<double, 1, 1> rkb = std::get<1>(tableau); 
    Matrix<double, 1, 1> rkc = std::get<2>(tableau);

    // Get the roots of the end system
    ProjectiveStraightLineHomotopy<double, 2, 1> h(g, f, g_roots, rkA, rkb, rkc);
    std::cout << h.solve(1e-8, 1e-8, 5, 1e-3, 0.01) << std::endl;
}

/**
 * Solve for the intersection of two circles by homotopy continuation from the
 * total-degree start system (x^2 - 1, y^2 - 1).
 */
void testSolveCircleIntersectRK4()
{
    // Sample a random complex number
    boost::random::mt19937 rng(42);
    boost::random::uniform_01<> dist;
    std::complex<double> gamma(dist(rng), dist(rng)); 

    // Define the start polynomial system
    MultivariatePolynomial<double, 2> g1 = polynomialOfUnity<double, 2>(0, 2);
    MultivariatePolynomial<double, 2> g2 = polynomialOfUnity<double, 2>(1, 2);
    g1 *= gamma; 
    g2 *= gamma;
    std::array<MultivariatePolynomial<double, 2>, 2> g { g1, g2 };

    // Define the end polynomial system
    double d = 1.5;
    Matrix<std::complex<double>, Dynamic, Dynamic> f1_coefs(3, 3);
    Matrix<std::complex<double>, Dynamic, Dynamic> f2_coefs(3, 3);
    f1_coefs << -1, 0, 1,
                 0, 0, 0,
                 1, 0, 0;
    f2_coefs << d*d - 1, 0, 1,
                   -2*d, 0, 0,
                      1, 0, 0;
    auto f1_coefmap = matrixToMap<std::complex<double>, 2>(f1_coefs, 0, 1);
    auto f2_coefmap = matrixToMap<std::complex<double>, 2>(f2_coefs, 0, 1);
    MultivariatePolynomial<double, 2> f1(f1_coefmap);
    MultivariatePolynomial<double, 2> f2(f2_coefmap);
    std::array<MultivariatePolynomial<double, 2>, 2> f { f1, f2 };

    // Get the roots of the start system
    Matrix<std::complex<double>, Dynamic, 2> g_roots = rootsOfUnity<double>(2, 2);

    // Get the Butcher tableau for the 4th-order Runge-Kutta method
    auto tableau = rk4Tableau<double>(); 
    Matrix<double, 4, 4> rkA = std::get<0>(tableau); 
    Matrix<double, 4, 1> rkb = std::get<1>(tableau); 
    Matrix<double, 4, 1> rkc = std::get<2>(tableau);

    // Get the roots of the end system
    ProjectiveStraightLineHomotopy<double, 2, 4> h(g, f, g_roots, rkA, rkb, rkc);
    std::cout << h.solve(1e-8, 1e-8, 5, 1e-3, 0.1) << std::endl;
}

int main()
{
    testSolveCubic();
    testSolveCircleIntersectEuler();
    testSolveCircleIntersectRK4();
}
