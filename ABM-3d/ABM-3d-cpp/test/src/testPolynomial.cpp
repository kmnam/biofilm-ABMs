/**
 * Test module for the `HighPrecisionPolynomial` class. 
 *
 * Authors:
 *     Kee-Myoung Nam
 *
 * Last updated:
 *     10/24/2025
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

typedef boost::multiprecision::number<boost::multiprecision::mpfr_float_backend<50> >  RealType; 
typedef boost::multiprecision::number<boost::multiprecision::mpc_complex_backend<50> > ComplexType;

/**
 * A series of tests for the various solution methods. 
 */
TEST_CASE("Tests for solution methods", "[solveCompanion(), solveDurandKerner(), solveAberth()]")
{
    const RealType tol = 1e-8;
    boost::random::mt19937 rng(1234567890);
    boost::random::uniform_01<> dist;
    const RealType minroot = -10; 
    const RealType maxroot = 10;

    // Case 1: Quadratic polynomials (x - a) * (x - b) = x^2 - (a + b)*x + a*b
    Matrix<RealType, Dynamic, 1> coefs(3);
    for (int i = 0; i < 10; ++i)
    {
        RealType a = minroot + (maxroot - minroot) * dist(rng); 
        RealType b = minroot + (maxroot - minroot) * dist(rng);  
        coefs << a * b, -a - b, 1;
        HighPrecisionPolynomial<50> p(coefs);
        Matrix<ComplexType, Dynamic, 1> roots; 
        RealType err1, err2; 

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
        RealType a = minroot + (maxroot - minroot) * static_cast<RealType>(dist(rng)); 
        RealType b = minroot + (maxroot - minroot) * static_cast<RealType>(dist(rng));
        RealType c = minroot + (maxroot - minroot) * static_cast<RealType>(dist(rng));
        std::vector<RealType> targets {a, b, c}; 
        std::sort(targets.begin(), targets.end());  
        coefs << -a * b * c, a * b + a * c + b * c, -a - b - c, 1; 
        HighPrecisionPolynomial<50> p(coefs);
        Matrix<ComplexType, Dynamic, 1> roots; 
        RealType err; 

        // Solve using the Durand-Kerner method 
        roots = p.solveDurandKerner(1e-8);
        REQUIRE(roots.size() == 3); 
        REQUIRE(abs(imag(roots(0))) < tol);
        REQUIRE(abs(imag(roots(1))) < tol);
        REQUIRE(abs(imag(roots(2))) < tol);
        std::vector<RealType> roots2 {real(roots(0)), real(roots(1)), real(roots(2))};
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
        std::vector<RealType> roots3 {real(roots(0)), real(roots(1)), real(roots(2))};
        std::sort(roots3.begin(), roots3.end()); 
        err = 0.0; 
        for (int j = 0; j < 3; ++j)
            err += abs(roots3[j] - targets[j]); 
        REQUIRE(err < tol);
    }
}

