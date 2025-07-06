/**
 * Test module for the `Field` class and various linear algebra functions.
 *
 * Authors:
 *     Kee-Myoung Nam
 *
 * Last updated:
 *     7/6/2025
 */
#include <iostream>
#include <functional>
#include <Eigen/Dense>
#include <boost/multiprecision/gmp.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "../../include/utils.hpp"
#include "../../include/fields.hpp"

using namespace Eigen;

typedef boost::multiprecision::mpq_rational Rational;

TEST_CASE("Tests for Fp class with p == 2", "[Fp]")
{
    Fp<2> z(0);
    Fp<2> x(1); 
    REQUIRE(z == 0);
    REQUIRE(x == 1);

    // Test all binary arithmetic functions
    REQUIRE(z + z == 0); 
    REQUIRE(z - z == 0); 
    REQUIRE(z * z == 0); 
    REQUIRE_THROWS(z / z);  
    REQUIRE(x + z == 1);
    REQUIRE(x - z == 1);
    REQUIRE(x * z == 0); 
    REQUIRE_THROWS(x / z); 
    REQUIRE(z + x == 1); 
    REQUIRE(z - x == 1); 
    REQUIRE(z * x == 0); 
    REQUIRE(z / x == 0);
    REQUIRE(x + x == 2); 
    REQUIRE(x - x == 0);
    REQUIRE(x * x == 1); 
    REQUIRE(x / x == 1);

    // Test modular equivalence
    REQUIRE(z != 1);  
    REQUIRE(z == 2); 
    REQUIRE(z != 3); 
    REQUIRE(z == 4); 
    REQUIRE(z != 5);
    REQUIRE(z != -1); 
    REQUIRE(z == -2);
    REQUIRE(z != -3);  
    REQUIRE(z == -4); 
    REQUIRE(z != -5);
    REQUIRE(x != 0); 
    REQUIRE(x != 2); 
    REQUIRE(x == 3);
    REQUIRE(x != 4); 
    REQUIRE(x == 5);
    REQUIRE(x == -1);
    REQUIRE(x != -2); 
    REQUIRE(x == -3);
    REQUIRE(x != -4); 
    REQUIRE(x == -5);

    // Test (if macro is defined) that non-integer rationals cannot be 
    // added, subtracted, etc.
    boost::multiprecision::mpq_rational p(1, 3);
    REQUIRE_THROWS(Fp<2>(p));
    REQUIRE(x != p); 
    REQUIRE(z != p); 
    REQUIRE_THROWS(x + p); 
    REQUIRE_THROWS(x - p); 
    REQUIRE_THROWS(x * p);
    REQUIRE_THROWS(x / p); 
    #ifdef CHECK_FP_RATIONAL_NONINTEGER
        REQUIRE_THROWS(x += p); 
        REQUIRE_THROWS(x -= p); 
        REQUIRE_THROWS(x *= p);
        REQUIRE_THROWS(x /= p); 
    #endif
}

TEST_CASE("Tests for row echelon form function", "[rowEchelonForm()]")
{
    // Example taken from Wikipedia:
    // https://en.wikipedia.org/wiki/Gaussian_elimination
    Matrix<Rational, Dynamic, Dynamic> A(3, 4); 
    A << 1,  3,  1,  9, 
         1,  1, -1,  1,
         3, 11,  5, 35;
    Matrix<Rational, Dynamic, Dynamic> refA = ::rowEchelonForm<Rational>(A, true);
    REQUIRE(refA(0, 0) == 1); 
    REQUIRE(refA(0, 1) == 0); 
    REQUIRE(refA(0, 2) == -2); 
    REQUIRE(refA(0, 3) == -3);
    REQUIRE(refA(1, 0) == 0); 
    REQUIRE(refA(1, 1) == 1); 
    REQUIRE(refA(1, 2) == 1); 
    REQUIRE(refA(1, 3) == 4); 
    REQUIRE(refA(2, 0) == 0); 
    REQUIRE(refA(2, 1) == 0); 
    REQUIRE(refA(2, 2) == 0); 
    REQUIRE(refA(2, 3) == 0);

    // Another example from Wikipedia
    A.resize(4, 4); 
    A << 1, 0, 4, 2,
         1, 2, 6, 2,
         2, 0, 8, 8,
         2, 1, 9, 4; 
    refA = ::rowEchelonForm<Rational>(A, true);
    REQUIRE(refA(0, 0) == 1); 
    REQUIRE(refA(0, 1) == 0); 
    REQUIRE(refA(0, 2) == 4); 
    REQUIRE(refA(0, 3) == 0); 
    REQUIRE(refA(1, 0) == 0); 
    REQUIRE(refA(1, 1) == 1); 
    REQUIRE(refA(1, 2) == 1); 
    REQUIRE(refA(1, 3) == 0); 
    REQUIRE(refA(2, 0) == 0); 
    REQUIRE(refA(2, 1) == 0); 
    REQUIRE(refA(2, 2) == 0); 
    REQUIRE(refA(2, 3) == 1);
    REQUIRE(refA(3, 0) == 0); 
    REQUIRE(refA(3, 1) == 0); 
    REQUIRE(refA(3, 2) == 0); 
    REQUIRE(refA(3, 3) == 0);

    // Another example from Wikipedia
    A.resize(3, 4); 
    A <<  2,  1, -1,   8,
         -3, -1,  2, -11,
         -2,  1,  2,  -3; 
    refA = ::rowEchelonForm<Rational>(A, true);
    REQUIRE(refA(0, 0) == 1); 
    REQUIRE(refA(0, 1) == 0); 
    REQUIRE(refA(0, 2) == 0); 
    REQUIRE(refA(0, 3) == 2); 
    REQUIRE(refA(1, 0) == 0); 
    REQUIRE(refA(1, 1) == 1); 
    REQUIRE(refA(1, 2) == 0); 
    REQUIRE(refA(1, 3) == 3); 
    REQUIRE(refA(2, 0) == 0); 
    REQUIRE(refA(2, 1) == 0); 
    REQUIRE(refA(2, 2) == 1); 
    REQUIRE(refA(2, 3) == -1);

    // Another example from Wikipedia
    A.resize(3, 6); 
    A <<  2, -1,  0, 1, 0, 0,
         -1,  2, -1, 0, 1, 0,
          0, -1,  2, 0, 0, 1;
    refA = ::rowEchelonForm<Rational>(A, true);
    REQUIRE(refA(0, 0) == 1); 
    REQUIRE(refA(0, 1) == 0); 
    REQUIRE(refA(0, 2) == 0); 
    REQUIRE(refA(0, 3) == Rational(3, 4));
    REQUIRE(refA(0, 4) == Rational(1, 2)); 
    REQUIRE(refA(0, 5) == Rational(1, 4)); 
    REQUIRE(refA(1, 0) == 0); 
    REQUIRE(refA(1, 1) == 1); 
    REQUIRE(refA(1, 2) == 0); 
    REQUIRE(refA(1, 3) == Rational(1, 2));
    REQUIRE(refA(1, 4) == 1); 
    REQUIRE(refA(1, 5) == Rational(1, 2)); 
    REQUIRE(refA(2, 0) == 0); 
    REQUIRE(refA(2, 1) == 0); 
    REQUIRE(refA(2, 2) == 1); 
    REQUIRE(refA(2, 3) == Rational(1, 4));
    REQUIRE(refA(2, 4) == Rational(1, 2)); 
    REQUIRE(refA(2, 5) == Rational(3, 4));

    // Example taken from Wikipedia:
    // https://en.wikipedia.org/wiki/System_of_linear_equations
    A.resize(3, 4); 
    A << 1, 3, -2, 5,
         3, 5,  6, 7,
         2, 4,  3, 8;
    refA = ::rowEchelonForm<Rational>(A, true);
    REQUIRE(refA(0, 0) == 1); 
    REQUIRE(refA(0, 1) == 0); 
    REQUIRE(refA(0, 2) == 0); 
    REQUIRE(refA(0, 3) == -15); 
    REQUIRE(refA(1, 0) == 0); 
    REQUIRE(refA(1, 1) == 1); 
    REQUIRE(refA(1, 2) == 0); 
    REQUIRE(refA(1, 3) == 8); 
    REQUIRE(refA(2, 0) == 0); 
    REQUIRE(refA(2, 1) == 0); 
    REQUIRE(refA(2, 2) == 1); 
    REQUIRE(refA(2, 3) == 2); 
}

TEST_CASE("Tests for kernel function", "[kernel()]")
{
    // Example taken from Wikipedia: 
    // https://en.wikipedia.org/wiki/Kernel_(linear_algebra)
    Matrix<Rational, Dynamic, Dynamic> A(2, 3); 
    A <<  2, 3, 5, 
         -4, 2, 3;
    Matrix<Rational, Dynamic, Dynamic> kerA = ::kernel<Rational>(A);
    REQUIRE(kerA.rows() == 1);     // #rows = dim(ker A)
    REQUIRE(kerA.cols() == 3); 
    for (int i = 0; i < kerA.rows(); ++i)
        REQUIRE(((A * kerA.row(i).transpose()).array() == 0).all());

    // Another example taken from Wikipedia 
    A.resize(4, 6);
    A << 1, 0, -3, 0,  2, -8,
         0, 1,  5, 0, -1,  4,
         0, 0,  0, 1,  7, -9,
         0, 0,  0, 0,  0,  0; 
    kerA = ::kernel<Rational>(A);
    REQUIRE(kerA.rows() == 3);     // #rows = dim(ker A)
    REQUIRE(kerA.cols() == 6); 
    for (int i = 0; i < kerA.rows(); ++i)
        REQUIRE(((A * kerA.row(i).transpose()).array() == 0).all());

    // Convert the second example into Z/2Z
    Matrix<Fp<2>, Dynamic, Dynamic> B(4, 6);
    B << 1, 0, 1, 0, 0, 0,
         0, 1, 1, 0, 1, 0,
         0, 0, 0, 1, 1, 1,
         0, 0, 0, 0, 0, 0;
    Matrix<Fp<2>, Dynamic, Dynamic> kerB = ::kernel<Fp<2> >(B);
    for (int i = 0; i < kerB.rows(); ++i)
        REQUIRE(((B * kerB.row(i).transpose()).array() == 0).all());

    // Another example taken from Wikipedia:
    // https://en.wikipedia.org/wiki/Row_and_column_spaces
    A.resize(4, 5); 
    A <<  2,  4, 1, 3, 2,
         -1, -2, 1, 0, 5,
          1,  6, 2, 2, 2,
          3,  6, 2, 5, 1; 
    kerA = ::kernel<Rational>(A); 
    REQUIRE(kerA.rows() == 1);     // #rows = dim(ker A)
    REQUIRE(kerA.cols() == 5);
    REQUIRE(((A * kerA.row(0).transpose()).array() == 0).all());
    Rational c = 6 / kerA(0, 0); 
    REQUIRE(kerA(0) * c == 6); 
    REQUIRE(kerA(1) * c == -1); 
    REQUIRE(kerA(2) * c == 4); 
    REQUIRE(kerA(3) * c == -4); 
    REQUIRE(kerA(4) * c == 0); 
}

TEST_CASE("Tests for column space function", "[columnSpace()]")
{
    // Example taken from Wikipedia: 
    // https://en.wikipedia.org/wiki/Row_and_column_spaces
    Matrix<Rational, Dynamic, Dynamic> A(4, 5); 
    A <<  2,  4, 1, 3, 2,
         -1, -2, 1, 0, 5,
          1,  6, 2, 2, 2,
          3,  6, 2, 5, 1; 
    Matrix<Rational, Dynamic, Dynamic> imA = ::columnSpace<Rational>(A.transpose());
    REQUIRE(imA.rows() == 4);    // #rows = dim(im A)
    REQUIRE(imA.cols() == 5); 
    REQUIRE(imA == A); 

    // Another example taken from Wikipedia 
    A.resize(3, 2);
    A << 1, 0,
         0, 1,
         2, 0;
    imA = ::columnSpace<Rational>(A); 
    REQUIRE(imA.rows() == 2);    // #rows = dim(im A)
    REQUIRE(imA.cols() == 3); 
    REQUIRE(imA == A.transpose());

    // Another example taken from Wikipedia
    A.resize(4, 4); 
    A << 1, 3, 1, 4,
         2, 7, 3, 9,
         1, 5, 3, 1,
         1, 2, 0, 8; 
    imA = ::columnSpace<Rational>(A); 
    REQUIRE(imA.rows() == 3);    // #rows = dim(im A)
    REQUIRE(imA.cols() == 4); 
    REQUIRE(imA.row(0) == A.col(0)); 
    REQUIRE(imA.row(1) == A.col(1)); 
    REQUIRE(imA.row(2) == A.col(3));

    // Another example taken from Wikipedia
    A.resize(3, 3); 
    A << 1, 3, 2, 
         2, 7, 4, 
         1, 5, 2; 
    imA = ::columnSpace<Rational>(A.transpose());
    REQUIRE(imA.rows() == 2);     // #rows = dim(im A)
    REQUIRE(imA.cols() == 3); 
    REQUIRE(imA.row(0) == A.row(0)); 
    REQUIRE(imA.row(1) == A.row(1));  
}

TEST_CASE("Tests for solve function", "[solve()]")
{
    // Example taken from Wikipedia:
    // https://en.wikipedia.org/wiki/System_of_linear_equations
    Matrix<Rational, Dynamic, Dynamic> A(3, 3), B(3, 1); 
    A <<  1, 3, -2, 
          3, 5,  6,
          2, 4,  3;
    B << 5, 7, 8;
    Matrix<Rational, Dynamic, Dynamic> solutions = ::solve<Rational>(A, B);
    REQUIRE(solutions.rows() == 3); 
    REQUIRE(solutions.cols() == 1); 
    REQUIRE(solutions(0) == -15); 
    REQUIRE(solutions(1) == 8); 
    REQUIRE(solutions(2) == 2);  
}

