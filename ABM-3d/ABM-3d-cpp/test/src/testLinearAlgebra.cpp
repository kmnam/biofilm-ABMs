/**
 * Test module for the `Field` class and various linear algebra functions.
 *
 * Authors:
 *     Kee-Myoung Nam
 *
 * Last updated:
 *     7/9/2025
 */
#include <iostream>
#include <functional>
#include <Eigen/Dense>
#include <boost/multiprecision/gmp.hpp>
#include <boost/random.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "../../include/utils.hpp"
#include "../../include/fields.hpp"

using namespace Eigen;

typedef boost::multiprecision::mpq_rational Rational;

/**
 * Return true if there is no solution to the linear system A * x = b, by 
 * checking whether there is an inconsistency in the row echelon form. 
 *
 * @param A Input matrix. 
 * @param b Input vector. 
 * @returns True if there is no solution to the linear system A * x = b. 
 */
template <typename T>
bool containsInconsistency(const Ref<const Matrix<T, Dynamic, Dynamic> >& A,
                           const Ref<const Matrix<T, Dynamic, 1> >& b)
{
    Matrix<T, Dynamic, Dynamic> system(A.rows(), A.cols() + 1); 
    system(Eigen::all, Eigen::seq(0, A.cols() - 1)) = A; 
    system.col(A.cols()) = b; 
    system = ::rowEchelonForm<T>(system); 
    bool found_inconsistency = false; 
    for (int i = 0; i < A.rows(); ++i)
    {
        if ((system.row(i).head(A.cols()).array() == 0).all() && system(i, A.cols()) != 0)
        {
            found_inconsistency = true;
            break; 
        }
    }

    return found_inconsistency; 
}

/**
 * Generate a random rational matrix with a given shape and rank.
 *
 * @param nrows Number of rows. 
 * @param ncols Number of columns. 
 * @param rank Desired rank. 
 * @param minval Minimum value. 
 * @param maxval Maximum value.
 * @param zero_prob Probability of sampling a zero entry in one of the two
 *                  matrices being multiplied (see below). 
 * @param rng Random number generator. 
 * @returns Sampled matrix.  
 */
Matrix<Rational, Dynamic, Dynamic> sampleByRank(const int nrows, const int ncols,
                                                const int rank,
                                                const double minval, 
                                                const double maxval,
                                                const double zero_prob,
                                                boost::random::mt19937& rng)
{
    Matrix<Rational, Dynamic, Dynamic> A(nrows, rank); 
    Matrix<Rational, Dynamic, Dynamic> B(rank, ncols);
    boost::random::uniform_01<> dist;  

    // Make A a full-rank matrix
    for (int i = 0; i < nrows; ++i)
    {
        for (int j = 0; j < rank; ++j)
        {
            double value = minval + (maxval - minval) * dist(rng); 
            A(i, j) = Rational(static_cast<int>(std::round(value * 10)), 10); 
        }
    }

    // Generate another matrix B with greater rank than A, possibly with
    // some zero entries
    //
    // To ensure that the rank is at least rank(A), we restrict zero sampling
    // to a subset of the columns 
    for (int i = 0; i < rank; ++i)
    {
        for (int j = 0; j < ncols; ++j)
        {
            double r = (j >= rank ? dist(rng) : 1); 
            if (r < zero_prob)
            {
                B(i, j) = 0; 
            }
            else 
            {
                double value = minval + (maxval - minval) * dist(rng); 
                B(i, j) = Rational(static_cast<int>(std::round(value * 10)), 10); 
            }
        }
    }

    return A * B; 
}

TEST_CASE("Tests for Z2 class", "[Z2]")
{
    Z2 z(0);
    Z2 x(1);
    REQUIRE(!z);
    REQUIRE(x); 
    REQUIRE(z == false); 
    REQUIRE(x == true);  
    REQUIRE(z == 0);
    REQUIRE(x == 1);
    REQUIRE(!(z && z)); 
    REQUIRE(!(z || z)); 
    REQUIRE(!(z ^ z)); 
    REQUIRE(!(z && x)); 
    REQUIRE((z || x)); 
    REQUIRE((z ^ x));
    REQUIRE((x && x)); 
    REQUIRE((x || x));
    REQUIRE(!(x ^ x));  

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

    // Test all binary arithmetic with integers
    REQUIRE(1 + z == 1);     // Addition
    REQUIRE(z + 1 == 1); 
    REQUIRE(2 + z == 0); 
    REQUIRE(z + 2 == 0);
    REQUIRE(3 + z == 1); 
    REQUIRE(z + 3 == 1); 
    REQUIRE(4 + z == 0); 
    REQUIRE(z + 4 == 0); 
    REQUIRE(5 + z == 1); 
    REQUIRE(z + 5 == 1); 
    REQUIRE(1 + x == 0); 
    REQUIRE(x + 1 == 0); 
    REQUIRE(2 + x == 1); 
    REQUIRE(x + 2 == 1);
    REQUIRE(3 + x == 0); 
    REQUIRE(x + 3 == 0); 
    REQUIRE(4 + x == 1); 
    REQUIRE(x + 4 == 1); 
    REQUIRE(5 + x == 0); 
    REQUIRE(x + 5 == 0);
    REQUIRE(1 - z == 1);     // Subtraction
    REQUIRE(z - 1 == 1); 
    REQUIRE(2 - z == 0); 
    REQUIRE(z - 2 == 0);
    REQUIRE(3 - z == 1); 
    REQUIRE(z - 3 == 1); 
    REQUIRE(4 - z == 0); 
    REQUIRE(z - 4 == 0); 
    REQUIRE(5 - z == 1); 
    REQUIRE(z - 5 == 1); 
    REQUIRE(1 - x == 0); 
    REQUIRE(x - 1 == 0); 
    REQUIRE(2 - x == 1); 
    REQUIRE(x - 2 == 1);
    REQUIRE(3 - x == 0); 
    REQUIRE(x - 3 == 0); 
    REQUIRE(4 - x == 1); 
    REQUIRE(x - 4 == 1); 
    REQUIRE(5 - x == 0); 
    REQUIRE(x - 5 == 0);
    REQUIRE(1 * z == 0);     // Multiplication 
    REQUIRE(z * 1 == 0); 
    REQUIRE(2 * z == 0); 
    REQUIRE(z * 2 == 0);
    REQUIRE(3 * z == 0); 
    REQUIRE(z * 3 == 0); 
    REQUIRE(4 * z == 0); 
    REQUIRE(z * 4 == 0); 
    REQUIRE(5 * z == 0); 
    REQUIRE(z * 5 == 0); 
    REQUIRE(1 * x == 1); 
    REQUIRE(x * 1 == 1); 
    REQUIRE(2 * x == 0); 
    REQUIRE(x * 2 == 0);
    REQUIRE(3 * x == 1); 
    REQUIRE(x * 3 == 1); 
    REQUIRE(4 * x == 0); 
    REQUIRE(x * 4 == 0); 
    REQUIRE(5 * x == 1); 
    REQUIRE(x * 5 == 1);
    REQUIRE_THROWS(1 / z);   // Division 
    REQUIRE(z / 1 == 0); 
    REQUIRE_THROWS(2 / z); 
    REQUIRE_THROWS(z / 2); 
    REQUIRE_THROWS(3 / z); 
    REQUIRE(z / 3 == 0); 
    REQUIRE_THROWS(4 / z); 
    REQUIRE_THROWS(z / 4); 
    REQUIRE_THROWS(5 / z); 
    REQUIRE(z / 5 == 0); 
    REQUIRE(1 / x == 1); 
    REQUIRE(x / 1 == 1); 
    REQUIRE(2 / x == 0); 
    REQUIRE_THROWS(x / 2);
    REQUIRE(3 / x == 1); 
    REQUIRE(x / 3 == 1); 
    REQUIRE(4 / x == 0); 
    REQUIRE_THROWS(x / 4);
    REQUIRE(5 / x == 1); 
    REQUIRE(x / 5 == 1);

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
}

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

    // Test all binary arithmetic with integers
    REQUIRE(1 + z == 1);     // Addition
    REQUIRE(z + 1 == 1); 
    REQUIRE(2 + z == 0); 
    REQUIRE(z + 2 == 0);
    REQUIRE(3 + z == 1); 
    REQUIRE(z + 3 == 1); 
    REQUIRE(4 + z == 0); 
    REQUIRE(z + 4 == 0); 
    REQUIRE(5 + z == 1); 
    REQUIRE(z + 5 == 1); 
    REQUIRE(1 + x == 0); 
    REQUIRE(x + 1 == 0); 
    REQUIRE(2 + x == 1); 
    REQUIRE(x + 2 == 1);
    REQUIRE(3 + x == 0); 
    REQUIRE(x + 3 == 0); 
    REQUIRE(4 + x == 1); 
    REQUIRE(x + 4 == 1); 
    REQUIRE(5 + x == 0); 
    REQUIRE(x + 5 == 0);
    REQUIRE(1 - z == 1);     // Subtraction
    REQUIRE(z - 1 == 1); 
    REQUIRE(2 - z == 0); 
    REQUIRE(z - 2 == 0);
    REQUIRE(3 - z == 1); 
    REQUIRE(z - 3 == 1); 
    REQUIRE(4 - z == 0); 
    REQUIRE(z - 4 == 0); 
    REQUIRE(5 - z == 1); 
    REQUIRE(z - 5 == 1); 
    REQUIRE(1 - x == 0); 
    REQUIRE(x - 1 == 0); 
    REQUIRE(2 - x == 1); 
    REQUIRE(x - 2 == 1);
    REQUIRE(3 - x == 0); 
    REQUIRE(x - 3 == 0); 
    REQUIRE(4 - x == 1); 
    REQUIRE(x - 4 == 1); 
    REQUIRE(5 - x == 0); 
    REQUIRE(x - 5 == 0);
    REQUIRE(1 * z == 0);     // Multiplication 
    REQUIRE(z * 1 == 0); 
    REQUIRE(2 * z == 0); 
    REQUIRE(z * 2 == 0);
    REQUIRE(3 * z == 0); 
    REQUIRE(z * 3 == 0); 
    REQUIRE(4 * z == 0); 
    REQUIRE(z * 4 == 0); 
    REQUIRE(5 * z == 0); 
    REQUIRE(z * 5 == 0); 
    REQUIRE(1 * x == 1); 
    REQUIRE(x * 1 == 1); 
    REQUIRE(2 * x == 0); 
    REQUIRE(x * 2 == 0);
    REQUIRE(3 * x == 1); 
    REQUIRE(x * 3 == 1); 
    REQUIRE(4 * x == 0); 
    REQUIRE(x * 4 == 0); 
    REQUIRE(5 * x == 1); 
    REQUIRE(x * 5 == 1);
    REQUIRE_THROWS(1 / z);   // Division 
    REQUIRE(z / 1 == 0); 
    REQUIRE_THROWS(2 / z); 
    REQUIRE_THROWS(z / 2); 
    REQUIRE_THROWS(3 / z); 
    REQUIRE(z / 3 == 0); 
    REQUIRE_THROWS(4 / z); 
    REQUIRE_THROWS(z / 4); 
    REQUIRE_THROWS(5 / z); 
    REQUIRE(z / 5 == 0); 
    REQUIRE(1 / x == 1); 
    REQUIRE(x / 1 == 1); 
    REQUIRE(2 / x == 0); 
    REQUIRE_THROWS(x / 2);
    REQUIRE(3 / x == 1); 
    REQUIRE(x / 3 == 1); 
    REQUIRE(4 / x == 0); 
    REQUIRE_THROWS(x / 4);
    REQUIRE(5 / x == 1); 
    REQUIRE(x / 5 == 1);

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
    REQUIRE(kerA.cols() == 1);     // #cols = dim(ker A)
    REQUIRE(kerA.rows() == 3); 
    for (int i = 0; i < kerA.cols(); ++i)
        REQUIRE(((A * kerA.col(i)).array() == 0).all());

    // Another example taken from Wikipedia 
    A.resize(4, 6);
    A << 1, 0, -3, 0,  2, -8,
         0, 1,  5, 0, -1,  4,
         0, 0,  0, 1,  7, -9,
         0, 0,  0, 0,  0,  0; 
    kerA = ::kernel<Rational>(A);
    REQUIRE(kerA.cols() == 3);     // #cols = dim(ker A)
    REQUIRE(kerA.rows() == 6); 
    for (int i = 0; i < kerA.cols(); ++i)
        REQUIRE(((A * kerA.col(i)).array() == 0).all());

    // Convert the second example into Z/2Z
    Matrix<Fp<2>, Dynamic, Dynamic> B(4, 6);
    B << 1, 0, 1, 0, 0, 0,
         0, 1, 1, 0, 1, 0,
         0, 0, 0, 1, 1, 1,
         0, 0, 0, 0, 0, 0;
    Matrix<Fp<2>, Dynamic, Dynamic> kerB = ::kernel<Fp<2> >(B);
    for (int i = 0; i < kerB.cols(); ++i)
        REQUIRE(((B * kerB.col(i)).array() == 0).all());

    // Another example taken from Wikipedia:
    // https://en.wikipedia.org/wiki/Row_and_column_spaces
    A.resize(4, 5); 
    A <<  2,  4, 1, 3, 2,
         -1, -2, 1, 0, 5,
          1,  6, 2, 2, 2,
          3,  6, 2, 5, 1; 
    kerA = ::kernel<Rational>(A); 
    REQUIRE(kerA.cols() == 1);     // #cols = dim(ker A)
    REQUIRE(kerA.rows() == 5);
    REQUIRE(((A * kerA.col(0)).array() == 0).all());
    Rational c = 6 / kerA(0, 0); 
    REQUIRE(kerA(0) * c == 6); 
    REQUIRE(kerA(1) * c == -1); 
    REQUIRE(kerA(2) * c == 4); 
    REQUIRE(kerA(3) * c == -4); 
    REQUIRE(kerA(4) * c == 0);

    // Generate a 20x30 matrix of rank 13
    boost::random::mt19937 rng(1234567890); 
    A = sampleByRank(20, 30, 13, -5, 5, 0.5, rng);
    kerA = ::kernel<Rational>(A);

    // Check that the nullity is correctly calculated (30 - 13 = 17) 
    REQUIRE(kerA.cols() == 17);    // #cols = dim(ker A)
    REQUIRE(kerA.rows() == 30);

    // Check that each basis vector indeed lies in the kernel of A
    for (int i = 0; i < kerA.cols(); ++i)
        REQUIRE(((A * kerA.col(i)).array() == 0).all()); 
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
    A = A.transpose();  
    Matrix<Rational, Dynamic, Dynamic> imA = ::columnSpace<Rational>(A); 
    REQUIRE(imA.cols() == 4);    // #cols = dim(im A)
    REQUIRE(imA.rows() == 5); 
    REQUIRE(imA == A); 

    // Another example taken from Wikipedia 
    A.resize(3, 2);
    A << 1, 0,
         0, 1,
         2, 0;
    imA = ::columnSpace<Rational>(A); 
    REQUIRE(imA.cols() == 2);    // #cols = dim(im A)
    REQUIRE(imA.rows() == 3); 
    REQUIRE(imA == A); 

    // Another example taken from Wikipedia
    A.resize(4, 4); 
    A << 1, 3, 1, 4,
         2, 7, 3, 9,
         1, 5, 3, 1,
         1, 2, 0, 8; 
    imA = ::columnSpace<Rational>(A); 
    REQUIRE(imA.cols() == 3);    // #cols = dim(im A)
    REQUIRE(imA.rows() == 4); 
    REQUIRE(imA.col(0) == A.col(0)); 
    REQUIRE(imA.col(1) == A.col(1)); 
    REQUIRE(imA.col(2) == A.col(3));

    // Another example taken from Wikipedia
    A.resize(3, 3); 
    A << 1, 3, 2, 
         2, 7, 4, 
         1, 5, 2;
    A = A.transpose();  
    imA = ::columnSpace<Rational>(A); 
    REQUIRE(imA.cols() == 2);     // #cols = dim(im A)
    REQUIRE(imA.rows() == 3); 
    REQUIRE(imA.col(0) == A.col(0)); 
    REQUIRE(imA.col(1) == A.col(1));

    // Generate a 20x30 matrix of rank 13
    boost::random::mt19937 rng(1234567890); 
    A = sampleByRank(20, 30, 13, -5, 5, 0.5, rng);
    imA = ::columnSpace<Rational>(A);

    // Test that the rank is correct 
    REQUIRE(imA.cols() == 13);    // #cols = dim(im A)
    REQUIRE(imA.rows() == 20);

    // Test that each column in the column space is linearly independent 
    // of the others 
    for (int i = 0; i < imA.cols(); ++i)
    {
        Matrix<Rational, Dynamic, Dynamic> B(imA.rows(), imA.cols() - 1);
        B(Eigen::all, Eigen::seq(0, i - 1)) = imA(Eigen::all, Eigen::seq(0, i - 1));
        B(Eigen::all, Eigen::seq(i, imA.cols() - 2))
            = imA(Eigen::all, Eigen::seq(i + 1, imA.cols() - 1));
        REQUIRE(containsInconsistency<Rational>(B, imA.col(i)));  
    }

    // Test that each column in A that does not lie within the basis is 
    // spanned by the basis vectors
    for (int i = 0; i < A.cols(); ++i)
        REQUIRE_NOTHROW(::solve<Rational>(imA, A.col(i))); 
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

    // Example taken from:
    // https://math.libretexts.org/Bookshelves/Linear_Algebra/
    // Fundamentals_of_Matrix_Algebra_(Hartman)/02%3A_Matrix_Arithmetic/
    // 2.05%3A_Solving_Matrix_Equations_AXB
    A.resize(2, 2); 
    B.resize(2, 3); 
    A << 1, -1,
         5,  3; 
    B << -8, -13,  1,
         32, -17, 21; 
    solutions = ::solve<Rational>(A, B); 
    REQUIRE(solutions.rows() == 2); 
    REQUIRE(solutions.cols() == 3); 
    REQUIRE(solutions(0, 0) == 1); 
    REQUIRE(solutions(1, 0) == 9); 
    REQUIRE(solutions(0, 1) == -7); 
    REQUIRE(solutions(1, 1) == 6); 
    REQUIRE(solutions(0, 2) == 3); 
    REQUIRE(solutions(1, 2) == 2);

    // Another example taken from:
    // https://math.libretexts.org/Bookshelves/Linear_Algebra/
    // Fundamentals_of_Matrix_Algebra_(Hartman)/02%3A_Matrix_Arithmetic/
    // 2.05%3A_Solving_Matrix_Equations_AXB
    A.resize(3, 3); 
    B.resize(3, 2); 
    A << 1,  0,  2,
         0, -1, -2,
         2, -1,  0; 
    B << -1,  2,
          2, -6,
          2, -4;
    solutions = ::solve<Rational>(A, B);
    REQUIRE(solutions.rows() == 3); 
    REQUIRE(solutions.cols() == 2);
    REQUIRE(solutions(0, 0) == 1); 
    REQUIRE(solutions(1, 0) == 0);
    REQUIRE(solutions(2, 0) == -1); 
    REQUIRE(solutions(0, 1) == 0); 
    REQUIRE(solutions(1, 1) == 4); 
    REQUIRE(solutions(2, 1) == 1);

    // Another example taken from:
    // https://math.libretexts.org/Bookshelves/Linear_Algebra/
    // Fundamentals_of_Matrix_Algebra_(Hartman)/02%3A_Matrix_Arithmetic/
    // 2.05%3A_Solving_Matrix_Equations_AXB
    A.resize(2, 2); 
    B.resize(2, 3); 
    A << 1, 2,
         3, 4;
    B << 3, 1, 17,
         7, 1, 39;
    solutions = ::solve<Rational>(A, B);
    REQUIRE(solutions.rows() == 2); 
    REQUIRE(solutions.cols() == 3);
    REQUIRE(solutions(0, 0) == 1); 
    REQUIRE(solutions(1, 0) == 1);
    REQUIRE(solutions(0, 1) == -1); 
    REQUIRE(solutions(1, 1) == 1); 
    REQUIRE(solutions(0, 2) == 5); 
    REQUIRE(solutions(1, 2) == 6);
}

TEST_CASE("Tests for complement functions", "[complement()]")
{
    // Example taken from Wikipedia:
    // https://en.wikipedia.org/wiki/Quotient_space_(linear_algebra)
    Matrix<Rational, Dynamic, Dynamic> B(2, 1);
    B << 1, 2;  
    Matrix<Rational, Dynamic, Dynamic> complement = ::complement<Rational>(B);
    REQUIRE(complement.cols() == 1);
    REQUIRE((
        complement(0) == 0 || complement(1) == 0 ||
        complement(0) / complement(1) != Rational(1, 2) 
    )); 

    // Example taken from:
    // https://math.stackexchange.com/questions/2554408/
    // how-to-find-the-basis-of-a-quotient-space
    B.resize(4, 2); 
    B <<  2,  0,
         -1,  0,
          0,  3,
          0, -1;
    complement = ::complement<Rational>(B); 
    REQUIRE(complement.cols() == 2);
    for (int i = 0; i < complement.cols(); ++i)
    {
        Matrix<Rational, Dynamic, Dynamic> Bv(B.rows(), B.cols() + 1); 
        Bv(Eigen::all, Eigen::seq(0, B.cols() - 1)) = B; 
        Bv.col(B.cols()) = complement.col(i); 
        Matrix<Rational, Dynamic, Dynamic> Bv_reduced = ::rowEchelonForm<Rational>(Bv, true);

        // Look for an inconsistency in the row echelon form 
        bool found_inconsistency = false;
        for (int j = 0; j < B.rows(); ++j)
        {
            if ((Bv_reduced.row(j).head(B.cols()).array() == 0).all() && Bv_reduced(j, B.cols()) != 0)
            {
                found_inconsistency = true; 
                break;
            }
        }
        REQUIRE(found_inconsistency); 
    } 
}

