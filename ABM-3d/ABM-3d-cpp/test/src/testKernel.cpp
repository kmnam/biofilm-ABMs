/**
 * Test module for the `kernel()` function.
 *
 * Authors:
 *     Kee-Myoung Nam
 *
 * Last updated:
 *     6/30/2025
 */
#include <iostream>
#include <functional>
#include <Eigen/Dense>
#include <boost/multiprecision/gmp.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "../../include/utils.hpp"

using namespace Eigen;

typedef boost::multiprecision::mpq_rational Rational; 

TEST_CASE("Tests for kernel function", "[kernel()]")
{
    // Example taken from Wikipedia: 
    // https://en.wikipedia.org/wiki/Kernel_(linear_algebra)
    Matrix<Rational, Dynamic, Dynamic> A(2, 3); 
    A <<  2, 3, 5, 
         -4, 2, 3;
    Matrix<Rational, Dynamic, Dynamic> kernel = ::kernel<Rational>(A);
    for (int i = 0; i < kernel.rows(); ++i)
        REQUIRE(((A * kernel.row(i).transpose()).array() == 0).all());

    // Another example taken from Wikipedia 
    A.resize(4, 6);
    A << 1, 0, -3, 0,  2, -8,
         0, 1,  5, 0, -1,  4,
         0, 0,  0, 1,  7, -9,
         0, 0,  0, 0,  0,  0; 
    kernel = ::kernel<Rational>(A); 
    for (int i = 0; i < kernel.rows(); ++i)
        REQUIRE(((A * kernel.row(i).transpose()).array() == 0).all());
}

