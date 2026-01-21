/**
 * Test module for the `HighPrecisionPolynomial` class. 
 *
 * Authors:
 *     Kee-Myoung Nam
 *
 * Last updated:
 *     1/21/2026
 */

#include <iostream>
#include <algorithm>
#include <Eigen/Dense>
#include <boost/random.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "../../include/polynomials.hpp"

using namespace Eigen;

using boost::multiprecision::abs;
using boost::multiprecision::real; 
using boost::multiprecision::imag;

/**
 * A series of tests for the various solution methods. 
 */
TEST_CASE("Tests for solution methods", "[solveCompanion(), solveDurandKerner(), solveAberth()]")
{
    const RealType<50> tol = 1e-8;
    boost::random::mt19937 rng(1234567890);
    boost::random::uniform_01<> dist;
    const RealType<50> minroot = -10; 
    const RealType<50> maxroot = 10;

    // Case 1: Quadratic polynomials (x - a) * (x - b) = x^2 - (a + b)*x + a*b
    Matrix<RealType<50>, Dynamic, 1> coefs(3);
    for (int i = 0; i < 10; ++i)
    {
        RealType<50> a = minroot + (maxroot - minroot) * dist(rng); 
        RealType<50> b = minroot + (maxroot - minroot) * dist(rng);  
        coefs << a * b, -a - b, 1;
        HighPrecisionPolynomial<50> p(coefs);
        Matrix<ComplexType<50>, Dynamic, 1> roots; 
        RealType<50> err1, err2; 

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
        RealType<50> a = minroot + (maxroot - minroot) * static_cast<RealType<50> >(dist(rng)); 
        RealType<50> b = minroot + (maxroot - minroot) * static_cast<RealType<50> >(dist(rng));
        RealType<50> c = minroot + (maxroot - minroot) * static_cast<RealType<50> >(dist(rng));
        std::vector<RealType<50> > targets {a, b, c}; 
        std::sort(targets.begin(), targets.end());  
        coefs << -a * b * c, a * b + a * c + b * c, -a - b - c, 1; 
        HighPrecisionPolynomial<50> p(coefs);
        Matrix<ComplexType<50>, Dynamic, 1> roots; 
        RealType<50> err; 

        // Solve using the Durand-Kerner method 
        roots = p.solveDurandKerner(1e-8);
        REQUIRE(roots.size() == 3); 
        REQUIRE(abs(imag(roots(0))) < tol);
        REQUIRE(abs(imag(roots(1))) < tol);
        REQUIRE(abs(imag(roots(2))) < tol);
        std::vector<RealType<50> > roots2 {real(roots(0)), real(roots(1)), real(roots(2))};
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
        std::vector<RealType<50> > roots3 {real(roots(0)), real(roots(1)), real(roots(2))};
        std::sort(roots3.begin(), roots3.end()); 
        err = 0.0; 
        for (int j = 0; j < 3; ++j)
            err += abs(roots3[j] - targets[j]); 
        REQUIRE(err < tol);
    }
}

/**
 * Test interpolation. 
 */
TEST_CASE("Tests for interpolation methods", "[interpolate()]")
{
    // Example taken from: https://ubcmath.github.io/MATH307/systems/interpolation.html
    Array<RealType<50>, Dynamic, 1> x(4), y(4);
    x << 0, 1, 2, 3; 
    y << -1, -1, 1, -1;
    HighPrecisionPolynomial<50> f1 = interpolate<50>(x, y); 
    REQUIRE(f1.getDegree() == 3);
    Matrix<ComplexType<50>, Dynamic, 1> f1_coefs = f1.getCoefs(); 
    REQUIRE_THAT(
        static_cast<double>(real(f1_coefs(0))), Catch::Matchers::WithinAbs(-1, 1e-8)
    ); 
    REQUIRE_THAT(
        static_cast<double>(real(f1_coefs(1))), Catch::Matchers::WithinAbs(-3, 1e-8)
    ); 
    REQUIRE_THAT(
        static_cast<double>(real(f1_coefs(2))), Catch::Matchers::WithinAbs(4, 1e-8)
    ); 
    REQUIRE_THAT(
        static_cast<double>(real(f1_coefs(3))), Catch::Matchers::WithinAbs(-1, 1e-8)
    );

    // Cubic example 
    Array<RealType<50>, Dynamic, 1> g2_coefs(4); 
    g2_coefs << 4, 3, 2, 1; 
    HighPrecisionPolynomial<50> g2(g2_coefs); 
    REQUIRE(g2.getDegree() == 3); 
    x << -2, -1, 5, 17;
    for (int i = 0; i < 4; ++i) 
        y(i) = real(g2.eval(x(i)));
    HighPrecisionPolynomial<50> f2 = interpolate<50>(x, y);
    REQUIRE(f2.getDegree() == 3); 
    Matrix<ComplexType<50>, Dynamic, 1> f2_coefs = f2.getCoefs();
    for (int i = 0; i < 4; ++i) 
        REQUIRE_THAT( 
            static_cast<double>(real(f2_coefs(i))),
            Catch::Matchers::WithinAbs(static_cast<double>(g2_coefs(i)), 1e-8)
        ); 
}

