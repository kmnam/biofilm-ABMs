/**
 * Test module for the functions in `ellipsoid.hpp`.
 *
 * Authors:
 *     Kee-Myoung Nam
 *
 * Last updated:
 *     6/2/2025
 */
#include <iostream>
#include <cmath>
#include <functional>
#include <Eigen/Dense>
#include <boost/math/constants/constants.hpp>
#include <boost/multiprecision/mpfr.hpp>
#include <boost/random.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "../../include/ellipsoid.hpp"

using namespace Eigen; 

// Use high-precision type for testing 
typedef boost::multiprecision::number<boost::multiprecision::mpfr_float_backend<100> > T; 

using std::abs; 
using boost::multiprecision::abs; 
using std::sin; 
using boost::multiprecision::sin;
using std::cos; 
using boost::multiprecision::cos; 
using std::sqrt; 
using boost::multiprecision::sqrt;
using std::real; 
using boost::multiprecision::real; 
using std::imag; 
using boost::multiprecision::imag; 

/* ------------------------------------------------------------------- //
 *                             TEST MODULES                            //
 * ------------------------------------------------------------------- */
/**
 * A series of tests for rotationToXUnitVector(). 
 */
TEST_CASE("Tests for rotation matrix function", "[rotationToXUnitVector()]")
{
    const double tol = 1e-8; 
    Matrix<T, Dynamic, 1> x = Matrix<T, Dynamic, 1>::LinSpaced(20, 0, 1);
    for (int i = 0; i < 20; ++i)
    {
        for (int j = 0; j < 20; ++j)
        {
            T sqnorm = x(i) * x(i) + x(j) * x(j);  
            if (sqnorm < 1)
            {
                Matrix<T, 3, 1> n;
                n << x(i), x(j), sqrt(1 - x(i) * x(i) - x(j) * x(j));
                Matrix<T, 3, 3> rot = rotationToXUnitVector<T>(n); 
                Matrix<T, 3, 1> y = rot * n; 
                REQUIRE_THAT(static_cast<double>(y(0)), Catch::Matchers::WithinAbs(1.0, tol)); 
                REQUIRE_THAT(static_cast<double>(y(1)), Catch::Matchers::WithinAbs(0.0, tol)); 
                REQUIRE_THAT(static_cast<double>(y(2)), Catch::Matchers::WithinAbs(0.0, tol));
                REQUIRE_THAT(static_cast<double>(rot.determinant()), Catch::Matchers::WithinAbs(1.0, tol));  
            }
        }
    }
}

/**
 * A series of tests for getEllipsoidQuadraticForm(). 
 */
TEST_CASE("Tests for ellipsoid quadratic form function", "[getEllipsoidQuadraticForm()]")
{
    const double tol = 1e-8; 

    // Case 1: Ellipsoid with long axis parallel to the x-axis  
    Matrix<T, 3, 1> r, n; 
    r << 0, 0, 0; 
    n << 1, 0, 0; 
    const T R = 0.8; 
    const T half_l = 0.5;
    auto result = getEllipsoidQuadraticForm<T>(r, n, R, half_l); 
    Matrix<T, 3, 3> A = result.first;
    Matrix<T, 3, 1> b = result.second; 
    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            if (i != j)
                REQUIRE_THAT(static_cast<double>(A(i, j)), Catch::Matchers::WithinAbs(0.0, tol)); 
            else if (i == 0) 
                REQUIRE_THAT(
                    static_cast<double>(A(i, i)),
                    Catch::Matchers::WithinAbs(static_cast<double>(1.0 / ((R + half_l) * (R + half_l))), tol)
                );
            else
                REQUIRE_THAT(
                    static_cast<double>(A(i, i)),
                    Catch::Matchers::WithinAbs(static_cast<double>(1.0 / (R * R)), tol)
                );
        }
        REQUIRE_THAT(static_cast<double>(b(i)), Catch::Matchers::WithinAbs(0.0, tol)); 
    }

    // Case 2: Ellipsoid with long axis parallel to the y-axis  
    n << 0, 1, 0; 
    result = getEllipsoidQuadraticForm<T>(r, n, R, half_l); 
    A = result.first; 
    b = result.second; 
    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            if (i != j)
                REQUIRE_THAT(static_cast<double>(A(i, j)), Catch::Matchers::WithinAbs(0.0, tol)); 
            else if (i == 1) 
                REQUIRE_THAT(
                    static_cast<double>(A(i, i)),
                    Catch::Matchers::WithinAbs(static_cast<double>(1.0 / ((R + half_l) * (R + half_l))), tol)
                );
            else
                REQUIRE_THAT(
                    static_cast<double>(A(i, i)),
                    Catch::Matchers::WithinAbs(static_cast<double>(1.0 / (R * R)), tol)
                );
        }
        REQUIRE_THAT(static_cast<double>(b(i)), Catch::Matchers::WithinAbs(0.0, tol)); 
    }

    // Case 3: Ellipsoid with long axis parallel to the z-axis  
    n << 0, 0, 1; 
    result = getEllipsoidQuadraticForm<T>(r, n, R, half_l); 
    A = result.first;
    b = result.second;
    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            if (i != j)
                REQUIRE_THAT(static_cast<double>(A(i, j)), Catch::Matchers::WithinAbs(0.0, tol)); 
            else if (i == 2) 
                REQUIRE_THAT(
                    static_cast<double>(A(i, i)),
                    Catch::Matchers::WithinAbs(static_cast<double>(1.0 / ((R + half_l) * (R + half_l))), tol)
                );
            else
                REQUIRE_THAT(
                    static_cast<double>(A(i, i)),
                    Catch::Matchers::WithinAbs(static_cast<double>(1.0 / (R * R)), tol)
                );
        }
        REQUIRE_THAT(static_cast<double>(b(i)), Catch::Matchers::WithinAbs(0.0, tol)); 
    }

    // Case 4: Ellipsoid with long axis parallel to (cos(30), sin(30), 0)
    n << cos(boost::math::constants::sixth_pi<T>()), 
         sin(boost::math::constants::sixth_pi<T>()),
         0.0; 
    result = getEllipsoidQuadraticForm<T>(r, n, R, half_l); 
    A = result.first;
    b = result.second;

    // Test that A is symmetric 
    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            if (i != j)
                REQUIRE(abs(A(i, j) - A(j, i)) < tol); 
        }
    }

    // Test that the eigenvalues are all positive
    EigenSolver<Matrix<double, 3, 3> > eigen(A.cast<double>());  
    Matrix<std::complex<double>, 3, 1> eigvals = eigen.eigenvalues(); 
    for (int i = 0; i < 3; ++i)
    {
        REQUIRE(real(eigvals(i)) > tol);
        REQUIRE(abs(imag(eigvals(i))) < tol);
    }

    // Check the semi-major and semi-minor axis lengths 
    Matrix<double, 3, 1> axis_lengths;
    axis_lengths << 1.0 / sqrt(real(eigvals(0))),
                    1.0 / sqrt(real(eigvals(1))), 
                    1.0 / sqrt(real(eigvals(2)));
    Matrix<double, 3, 1>::Index maxidx; 
    double max_length = axis_lengths.maxCoeff(&maxidx);
    REQUIRE_THAT(max_length, Catch::Matchers::WithinAbs(static_cast<double>(R + half_l), tol));  
    if (maxidx == 0) 
    {
        REQUIRE_THAT(axis_lengths(1), Catch::Matchers::WithinAbs(static_cast<double>(R), tol)); 
        REQUIRE_THAT(axis_lengths(2), Catch::Matchers::WithinAbs(static_cast<double>(R), tol));
    }
    else if (maxidx == 1)
    {
        REQUIRE_THAT(axis_lengths(0), Catch::Matchers::WithinAbs(static_cast<double>(R), tol)); 
        REQUIRE_THAT(axis_lengths(2), Catch::Matchers::WithinAbs(static_cast<double>(R), tol));
    }
    else 
    {
        REQUIRE_THAT(axis_lengths(0), Catch::Matchers::WithinAbs(static_cast<double>(R), tol)); 
        REQUIRE_THAT(axis_lengths(1), Catch::Matchers::WithinAbs(static_cast<double>(R), tol));
    }

    // Check the semi-major axis direction 
    REQUIRE_THAT(abs(eigen.eigenvectors()(0, maxidx)), Catch::Matchers::WithinAbs(static_cast<double>(n(0)), tol)); 
    REQUIRE_THAT(abs(eigen.eigenvectors()(1, maxidx)), Catch::Matchers::WithinAbs(static_cast<double>(n(1)), tol)); 
    REQUIRE_THAT(abs(eigen.eigenvectors()(2, maxidx)), Catch::Matchers::WithinAbs(static_cast<double>(n(2)), tol));  

    // Case 5: Ellipsoid with long axis parallel to (cos(30), sin(15), sqrt(1 - cos^2(30) - sin^2(15)))
    n << cos(boost::math::constants::sixth_pi<T>()), 
         sin(boost::math::constants::sixth_pi<T>() / 2.0),
         sqrt(1.0 - pow(cos(boost::math::constants::sixth_pi<T>()), 2) - pow(sin(boost::math::constants::sixth_pi<T>() / 2.0), 2));  
    result = getEllipsoidQuadraticForm<T>(r, n, R, half_l); 
    A = result.first;
    b = result.second;

    // Test that A is symmetric 
    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            if (i != j)
                REQUIRE(abs(A(i, j) - A(j, i)) < tol); 
        }
    }

    // Test that the eigenvalues are all positive
    eigen.compute(A.cast<double>()); 
    eigvals = eigen.eigenvalues();
    for (int i = 0; i < 3; ++i)
    {
        REQUIRE(real(eigvals(i)) > tol);
        REQUIRE(abs(imag(eigvals(i))) < tol);
    }

    // Check the semi-major and semi-minor axis lengths
    axis_lengths << 1.0 / sqrt(real(eigvals(0))),
                    1.0 / sqrt(real(eigvals(1))), 
                    1.0 / sqrt(real(eigvals(2)));
    max_length = axis_lengths.maxCoeff(&maxidx);
    REQUIRE_THAT(max_length, Catch::Matchers::WithinAbs(static_cast<double>(R + half_l), tol));  
    if (maxidx == 0) 
    {
        REQUIRE_THAT(axis_lengths(1), Catch::Matchers::WithinAbs(static_cast<double>(R), tol)); 
        REQUIRE_THAT(axis_lengths(2), Catch::Matchers::WithinAbs(static_cast<double>(R), tol));
    }
    else if (maxidx == 1)
    {
        REQUIRE_THAT(axis_lengths(0), Catch::Matchers::WithinAbs(static_cast<double>(R), tol)); 
        REQUIRE_THAT(axis_lengths(2), Catch::Matchers::WithinAbs(static_cast<double>(R), tol));
    }
    else 
    {
        REQUIRE_THAT(axis_lengths(0), Catch::Matchers::WithinAbs(static_cast<double>(R), tol)); 
        REQUIRE_THAT(axis_lengths(1), Catch::Matchers::WithinAbs(static_cast<double>(R), tol));
    }

    // Check the semi-major axis direction 
    REQUIRE_THAT(abs(eigen.eigenvectors()(0, maxidx)), Catch::Matchers::WithinAbs(static_cast<double>(n(0)), tol)); 
    REQUIRE_THAT(abs(eigen.eigenvectors()(1, maxidx)), Catch::Matchers::WithinAbs(static_cast<double>(n(1)), tol)); 
    REQUIRE_THAT(abs(eigen.eigenvectors()(2, maxidx)), Catch::Matchers::WithinAbs(static_cast<double>(n(2)), tol)); 

    // Case 6: Ellipsoid with same orientation but translated to (1, 2, 3)
    //
    // In this case, simply compare with the quadratic form from Case 5
    r << 1, 2, 3; 
    auto result2 = getEllipsoidQuadraticForm<T>(r, n, R, half_l); 
    Matrix<T, 3, 3> B = result2.first;
    Matrix<T, 3, 1> c = result2.second;
    T factor = 1.0 / (1.0 - r.dot(A * r)); 
    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            REQUIRE_THAT(static_cast<double>(B(i, j)), Catch::Matchers::WithinAbs(static_cast<double>(A(i, j) * factor), tol));  
        }
    }
}

/**
 * A series of tests for projectOntoEllipsoid().
 */
TEST_CASE("Tests for projection function", "[projectOntoEllipsoid()]")
{
    const double tol = 1e-8;
    const T project_tol = 1e-20;  
    const int max_iter = 10000; 

    // Case 1: Ellipsoid with long axis parallel to the x-axis 
    Matrix<T, 3, 1> r, n, a; 
    r << 0, 0, 0; 
    n << 1, 0, 0; 
    const T R = 0.8; 
    const T half_l = 0.5;
    auto result = getEllipsoidQuadraticForm<T>(r, n, R, half_l); 
    Matrix<T, 3, 3> A = result.first;
    Matrix<T, 3, 1> b = result.second; 

    // Case 1a: Query point lies along x-axis 
    a << 2 * (R + half_l), 0, 0;
    Matrix<T, 3, 1> x = projectOntoEllipsoid<T>(a, A, b, project_tol, max_iter, true);
    REQUIRE_THAT(static_cast<double>(x(0)), Catch::Matchers::WithinAbs(static_cast<double>(R + half_l), tol)); 
    REQUIRE_THAT(static_cast<double>(x(1)), Catch::Matchers::WithinAbs(0.0, tol)); 
    REQUIRE_THAT(static_cast<double>(x(2)), Catch::Matchers::WithinAbs(0.0, tol));
    a << -2 * (R + half_l), 0, 0;
    x = projectOntoEllipsoid<T>(a, A, b, project_tol, max_iter, true);
    REQUIRE_THAT(static_cast<double>(x(0)), Catch::Matchers::WithinAbs(static_cast<double>(-R - half_l), tol)); 
    REQUIRE_THAT(static_cast<double>(x(1)), Catch::Matchers::WithinAbs(0.0, tol)); 
    REQUIRE_THAT(static_cast<double>(x(2)), Catch::Matchers::WithinAbs(0.0, tol));

    // Case 1b: Query point lies along y-axis 
    a << 0, 2 * R, 0; 
    x = projectOntoEllipsoid<T>(a, A, b, project_tol, max_iter, true);
    REQUIRE_THAT(static_cast<double>(x(0)), Catch::Matchers::WithinAbs(0.0, tol)); 
    REQUIRE_THAT(static_cast<double>(x(1)), Catch::Matchers::WithinAbs(static_cast<double>(R), tol)); 
    REQUIRE_THAT(static_cast<double>(x(2)), Catch::Matchers::WithinAbs(0.0, tol));
    a << 0, -2 * R, 0; 
    x = projectOntoEllipsoid<T>(a, A, b, project_tol, max_iter, true);
    REQUIRE_THAT(static_cast<double>(x(0)), Catch::Matchers::WithinAbs(0.0, tol)); 
    REQUIRE_THAT(static_cast<double>(x(1)), Catch::Matchers::WithinAbs(static_cast<double>(-R), tol)); 
    REQUIRE_THAT(static_cast<double>(x(2)), Catch::Matchers::WithinAbs(0.0, tol));

    // Case 1c: Query point lies along z-axis 
    a << 0, 0, 2 * R; 
    x = projectOntoEllipsoid<T>(a, A, b, project_tol, max_iter, true);
    REQUIRE_THAT(static_cast<double>(x(0)), Catch::Matchers::WithinAbs(0.0, tol)); 
    REQUIRE_THAT(static_cast<double>(x(1)), Catch::Matchers::WithinAbs(0.0, tol)); 
    REQUIRE_THAT(static_cast<double>(x(2)), Catch::Matchers::WithinAbs(static_cast<double>(R), tol));
    a << 0, 0, -2 * R;  
    x = projectOntoEllipsoid<T>(a, A, b, project_tol, max_iter, true);
    REQUIRE_THAT(static_cast<double>(x(0)), Catch::Matchers::WithinAbs(0.0, tol)); 
    REQUIRE_THAT(static_cast<double>(x(1)), Catch::Matchers::WithinAbs(0.0, tol)); 
    REQUIRE_THAT(static_cast<double>(x(2)), Catch::Matchers::WithinAbs(static_cast<double>(-R), tol));

    // Case 1d: Query point lies elsewhere 
    //
    // Here, we check optimality by randomly sampling values along the 
    // ellipsoid surface and checking their distance to the query point
    boost::random::mt19937 rng(1234567890);
    boost::random::uniform_01<> dist;  
    a << static_cast<T>(dist(rng)),
         static_cast<T>(dist(rng)), 
         static_cast<T>(dist(rng)), 
    a *= (5 * R / a.norm());    // Ensure that the point lies outside the ellipsoid   
    x = projectOntoEllipsoid<T>(a, A, b, project_tol, max_iter, true);
    for (int i = 0; i < 1000; ++i)
    {
        T theta = boost::math::constants::pi<T>() * static_cast<T>(dist(rng));
        T phi = boost::math::constants::two_pi<T>() * static_cast<T>(dist(rng));
        Matrix<T, 3, 1> y; 
        y << (R + half_l) * sin(theta) * cos(phi),
             R * sin(theta) * sin(phi),
             R * cos(theta); 
        REQUIRE((x - a).norm() < (y - a).norm());  
    }
}

