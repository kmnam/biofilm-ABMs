/**
 * Test module for the `Polynomial` class. 
 *
 * Authors:
 *     Kee-Myoung Nam
 *
 * Last updated:
 *     6/3/2025
 */

#include <iostream>
#include <complex>
#include <algorithm>
#include <Eigen/Dense>
#include <boost/random.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "../../include/polynomials.hpp"

using namespace Eigen;

using std::abs;
using std::real; 
using std::imag; 

/**
 * A series of tests for the various solution methods. 
 */
TEST_CASE("Tests for solution methods", "[solveCompanion(), solveDurandKerner(), solveAberth()]")
{
    const double tol = 1e-8;
    boost::random::mt19937 rng(1234567890);
    boost::random::uniform_01<> dist;
    const double minroot = -10; 
    const double maxroot = 10;  

    // Case 1: Quadratic polynomials (x - a) * (x - b) = x^2 - (a + b)*x + a*b
    Matrix<double, Dynamic, 1> coefs(3);
    for (int i = 0; i < 10; ++i)
    {
        double a = minroot + (maxroot - minroot) * dist(rng); 
        double b = minroot + (maxroot - minroot) * dist(rng);  
        coefs << a * b, -a - b, 1;
        Polynomial<double> p(coefs);

        // Solve via the companion matrix 
        Matrix<std::complex<double>, Dynamic, 1> roots = p.solveCompanion();
        REQUIRE(roots.size() == 2); 
        REQUIRE(abs(imag(roots(0))) < tol);
        REQUIRE(abs(imag(roots(1))) < tol);  
        double err1 = abs(roots(0) - a) + abs(roots(1) - b); 
        double err2 = abs(roots(0) - b) + abs(roots(1) - a);
        REQUIRE((err1 < tol || err2 < tol)); 

        // Solve using the Durand-Kerner method 
        roots = p.solveDurandKerner(1e-8);
        REQUIRE(roots.size() == 2); 
        REQUIRE(abs(imag(roots(0))) < tol);
        REQUIRE(abs(imag(roots(1))) < tol);  
        err1 = abs(roots(0) - a) + abs(roots(1) - b);
        err2 = abs(roots(0) - b) + abs(roots(1) - a);  
        REQUIRE((err1 < tol || err2 < tol));

        // Solve using the Aberth-Ehrlich method 
        roots = p.solveAberth(1e-8);
        REQUIRE(roots.size() == 2); 
        REQUIRE(abs(imag(roots(0))) < tol);
        REQUIRE(abs(imag(roots(1))) < tol);  
        err1 = abs(roots(0) - a) + abs(roots(1) - b);
        err2 = abs(roots(0) - b) + abs(roots(1) - a);  
        REQUIRE((err1 < tol || err2 < tol));
    }
    
    // Case 2: Cubic polynomials (x - a) * (x - b) * (x - c)
    //                         = x^3 - (a + b + c)*x^2 + (a*b + a*c + b*c)*x - a*b*c
    coefs.resize(4); 
    for (int i = 0; i < 10; ++i)
    {
        double a = minroot + (maxroot - minroot) * dist(rng); 
        double b = minroot + (maxroot - minroot) * dist(rng);
        double c = minroot + (maxroot - minroot) * dist(rng);
        std::vector<double> targets {a, b, c}; 
        std::sort(targets.begin(), targets.end());  
        coefs << -a * b * c, a * b + a * c + b * c, -a - b - c, 1; 
        Polynomial<double> p(coefs);

        // Solve via the companion matrix 
        Matrix<std::complex<double>, Dynamic, 1> roots = p.solveCompanion();
        REQUIRE(roots.size() == 3); 
        REQUIRE(abs(imag(roots(0))) < tol);
        REQUIRE(abs(imag(roots(1))) < tol);
        REQUIRE(abs(imag(roots(2))) < tol);
        std::vector<double> roots1 {real(roots(0)), real(roots(1)), real(roots(2))};
        std::sort(roots1.begin(), roots1.end()); 
        double err = 0.0; 
        for (int j = 0; j < 3; ++j)
            err += abs(roots1[j] - targets[j]); 
        REQUIRE(err < tol);

        // Solve using the Durand-Kerner method 
        roots = p.solveDurandKerner(1e-8);
        REQUIRE(roots.size() == 3); 
        REQUIRE(abs(imag(roots(0))) < tol);
        REQUIRE(abs(imag(roots(1))) < tol);
        REQUIRE(abs(imag(roots(2))) < tol);
        std::vector<double> roots2 {real(roots(0)), real(roots(1)), real(roots(2))};
        std::sort(roots2.begin(), roots2.end()); 
        err = 0.0; 
        for (int j = 0; j < 3; ++j)
            err += abs(roots2[j] - targets[j]); 
        REQUIRE(err < tol);

        // Solve using the Aberth-Ehrlich method 
        roots = p.solveAberth(1e-8); 
        REQUIRE(roots.size() == 3); 
        REQUIRE(abs(imag(roots(0))) < tol);
        REQUIRE(abs(imag(roots(1))) < tol);
        REQUIRE(abs(imag(roots(2))) < tol);
        std::vector<double> roots3 {real(roots(0)), real(roots(1)), real(roots(2))};
        std::sort(roots3.begin(), roots3.end()); 
        err = 0.0; 
        for (int j = 0; j < 3; ++j)
            err += abs(roots3[j] - targets[j]); 
        REQUIRE(err < tol);
    }
}

