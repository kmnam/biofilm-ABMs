/**
 * Authors:
 *     Kee-Myoung Nam
 *
 * Last updated:
 *     3/20/2024
 */

#include <iostream>
#include <cmath>
#include <complex>
#include <Eigen/Dense>
#include <boost/math/constants/constants.hpp>
#include "../../include/polynomials.hpp"
#include "../../include/homotopies.hpp"
#include "../../include/spherocylinders.hpp"
#include "../../include/utils.hpp"

void testContactParallel()
{
    double R = 0.8;
    Matrix<double, 3, 1> r2, n2; 
    r2 << 1.9 * R, 0, 0;
    n2 << 0, 0, 1;
    double l = 1.0;

    std::array<MultivariatePolynomial<double, 2>, 2> g;
    std::complex<double> gamma(0.44, 0.70);
    g[0] = gamma * polynomialOfUnity<double, 2>(0, 2);
    g[1] = gamma * polynomialOfUnity<double, 2>(1, 2);
    Matrix<std::complex<double>, Dynamic, 2> g_roots = rootsOfUnity<double>(2, 2);

    auto tableau = eulerTableau<double>();
    auto contact = getContactRegion<double, 1>(
        r2, n2, l, R, l, R, g, g_roots, 10,
        std::get<0>(tableau), std::get<1>(tableau), std::get<2>(tableau),
        1e-8, 1e-8, 2, 0.1, 0.001, 1e-8, 1e+5
    );
    std::cout << contact.first << std::endl; 
    std::cout << contact.second << std::endl;
}

void testContactPerpendicular()
{
    double R = 0.8;
    Matrix<double, 3, 1> r2, n2;
    r2 << 1.9 * R, 0, 0;
    n2 << 0, 1, 0;
    double l = 1.0;

    std::array<MultivariatePolynomial<double, 2>, 2> g;
    std::complex<double> gamma(0.44, 0.70);
    g[0] = gamma * polynomialOfUnity<double, 2>(0, 2);
    g[1] = gamma * polynomialOfUnity<double, 2>(1, 2);
    Matrix<std::complex<double>, Dynamic, 2> g_roots = rootsOfUnity<double>(2, 2); 

    auto tableau = eulerTableau<double>();
    auto contact = getContactRegion<double, 1>(
        r2, n2, l, R, l, R, g, g_roots, 10,
        std::get<0>(tableau), std::get<1>(tableau), std::get<2>(tableau),
        1e-8, 1e-8, 2, 0.01, 0.001, 1e-8, 1e+5
    );
    std::cout << contact.first << std::endl; 
    std::cout << contact.second << std::endl;
}

void testContactSlanted()
{
    double R = 0.8;
    Matrix<double, 3, 1> r2, n2; 
    r2 << 1.9 * R, 0, 0;
    n2 << 0,
          std::sin(boost::math::constants::pi<double>() * 10 / 180),
          std::cos(boost::math::constants::pi<double>() * 10 / 180);
    double l = 1.0;

    std::array<MultivariatePolynomial<double, 2>, 2> g;
    std::complex<double> gamma(0.44, 0.70);
    g[0] = gamma * polynomialOfUnity<double, 2>(0, 2);
    g[1] = gamma * polynomialOfUnity<double, 2>(1, 2);
    Matrix<std::complex<double>, Dynamic, 2> g_roots = rootsOfUnity<double>(2, 2);

    auto tableau = eulerTableau<double>();
    auto contact = getContactRegion<double, 1>(
        r2, n2, l, R, l, R, g, g_roots, 10,
        std::get<0>(tableau), std::get<1>(tableau), std::get<2>(tableau),
        1e-8, 1e-8, 2, 0.01, 0.001, 1e-8, 1e+5
    );
    std::cout << contact.first << std::endl; 
    std::cout << contact.second << std::endl;
}

void testContactSlantedAlternativeStart()
{
    double R = 0.8;
    Matrix<double, 3, 1> r2, n2; 
    r2 << 1.9 * R, 0, 0;
    n2 << 0,
          std::sin(boost::math::constants::pi<double>() * 10 / 180),
          std::cos(boost::math::constants::pi<double>() * 10 / 180);
    double l = 1.0;

    double theta = 10.0;
    double z0 = 0.4; 
    Matrix<double, 3, 1> r0, n0, r, n;
    r0 << 0, 0, 0;
    n0 << 0, 0, 1; 
    r << 0.1, 0.2, 0.3; 
    n << std::sin(boost::math::constants::pi<double>() * theta / 180),
         0,
         std::cos(boost::math::constants::pi<double>() * theta / 180);
    std::array<MultivariatePolynomial<double, 2>, 2> g, f;
    std::complex<double> gamma(0.44, 0.70);
    g[0] = gamma * polynomialOfUnity<double, 2>(0, 2);
    g[1] = gamma * polynomialOfUnity<double, 2>(1, 2);
    Matrix<std::complex<double>, Dynamic, 2> g_roots = rootsOfUnity<double>(2, 2); 
    f[0] = getCylinder<double>(r0, n0, 0.8).evalElim(2, z0);
    f[1] = getCylinder<double>(r, n, 0.8).evalElim(2, z0);
    std::cout << f[0].toString() << std::endl; 
    std::cout << f[1].toString() << std::endl;

    auto tableau = eulerTableau<double>();
    ProjectiveStraightLineHomotopy<double, 2, 1> h(
        g, f, g_roots, std::get<0>(tableau), std::get<1>(tableau), std::get<2>(tableau)
    );
    Matrix<std::complex<double>, Dynamic, 2> f_roots = h.solve(1e-8, 1e-8, 2, 0.001, 0.01);
    auto contact = getContactRegion<double, 1>(
        r2, n2, l, R, l, R, g, g_roots, 10,
        std::get<0>(tableau), std::get<1>(tableau), std::get<2>(tableau),
        1e-8, 1e-8, 2, 0.01, 0.001, 1e-8, 1e+5
    );
    std::cout << contact.first << std::endl; 
    std::cout << contact.second << std::endl;
}

int main()
{
    testContactParallel();
    testContactPerpendicular();
    testContactSlanted();
    testContactSlantedAlternativeStart();
}
