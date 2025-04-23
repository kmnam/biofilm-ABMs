/**
 * Test module for the auxiliary integral calculations in `integrals.hpp`.
 *
 * Authors:
 *     Kee-Myoung Nam
 *
 * Last updated:
 *     4/23/2025
 */
#include <iostream>
#include <cmath>
#include <functional>
#include <Eigen/Dense>
#include <boost/math/constants/constants.hpp>
#include <boost/multiprecision/mpfr.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "../../include/integrals.hpp"

using namespace Eigen; 

// Use high-precision type for testing 
typedef boost::multiprecision::number<boost::multiprecision::mpfr_float_backend<100> > T; 

using std::sin; 
using boost::multiprecision::sin; 
using std::sqrt; 
using boost::multiprecision::sqrt; 

/* ------------------------------------------------------------------- //
 *                           HELPER FUNCTIONS                          //
 * ------------------------------------------------------------------- */
/**
 * Evaluate a 1-D integral of the given function over the given interval
 * using the trapezoid rule.
 *
 * @param func Input function. 
 * @param xmin Input minimum. 
 * @param xmax Input maximum.
 * @param meshsize Mesh size. 
 * @returns Corresponding integral.  
 */
T integrate(std::function<T(const T)>& func, const T xmin, const T xmax,
            const int meshsize = 10000)
{
    // Generate a mesh over the interval 
    Array<T, Dynamic, 1> mesh = Array<T, Dynamic, 1>::LinSpaced(meshsize, xmin, xmax);

    // Evaluate the function at each point along the mesh 
    Array<T, Dynamic, 1> y = Array<T, Dynamic, 1>::Zero(meshsize); 
    for (int i = 0; i < meshsize; ++i)
        y(i) = func(mesh(i));

    // Evaluate the integral 
    T delta = (xmax - xmin) / (meshsize - 1);
    T integral = 0.0; 
    for (int i = 1; i < meshsize; ++i)
        integral += delta * (y(i - 1) + y(i)) / 2;

    return integral;  
}

/* ------------------------------------------------------------------- //
 *                             TEST MODULES                            //
 * ------------------------------------------------------------------- */
/**
 * A series of tests for integral1(), which integrates \delta_i^\gamma(s). 
 */
TEST_CASE("Tests for auxiliary integral 1", "[integral1()]")
{
    const T R = 0.8;
    const double tol = 1e-8;
    int meshsize = 10000;
    const int max_meshsize = 1e+7;
    T rz, nz; 
    Array<T, 4, 1> exponents;  
    exponents << 0.5, 1.0, 1.5, 2.0;
    Array<T, 10, 1> angles = Array<T, 10, 1>::Zero();
    for (int i = 0; i < angles.size(); ++i)
        angles(i) = boost::math::constants::half_pi<T>() * (i + 1) / 10.0; 

    auto run_tests = [](const T rz, const T nz, const T R, const T half_l,
                        const T gamma, const double tol, const int meshsize,
                        const double check_value = std::numeric_limits<double>::quiet_NaN()) -> std::pair<T, T>
    {
        T ss = (R - rz) / nz;
        std::function<T(const T)> func = [rz, nz, R, gamma](const T s) -> T
        {
            return overlapGamma<T>(rz, nz, R, s, gamma); 
        };
        T target = integrate(func, -half_l, half_l, meshsize);
        T integral = integral1<T>(rz, nz, R, half_l, gamma, ss);
        if (!std::isnan(check_value))
            return std::make_pair(abs(integral - target), abs(target - check_value));
        else 
            return std::make_pair(abs(integral - target), 0.0);
    }; 
    
    // For each exponent ... 
    for (int i = 0; i < exponents.size(); ++i)
    {
        T gamma = exponents(i);

        // For each angle ... 
        for (int j = 0; j < angles.size(); ++j)
        {
            std::cout << "Running tests for integral1(), exponent = " << gamma
                      << ", nz = " << sin(angles(j)) << std::endl; 

            // Define the z-orientation
            nz = sin(angles(j));

            // Case 1: Assume the cell has a maximum overlap of 0.2 * R
            meshsize = 10000; 
            T half_l = 0.5;
            T max_overlap = 0.2 * R;
            rz = R + half_l * nz - max_overlap;
            std::pair<T, T> res = run_tests(rz, nz, R, half_l, gamma, tol, meshsize);
            while ((res.first > tol || res.second > tol) && meshsize < max_meshsize)
            {
                meshsize *= 10;
                std::cout << "- Rerunning tests with increased meshsize = "
                          << meshsize << std::endl; 
                res = run_tests(rz, nz, R, half_l, gamma, tol, meshsize);  
            }
            REQUIRE(res.first < tol);
            REQUIRE(res.second < tol);  

            // Case 2: Assume the cell does not contact the surface
            meshsize = 10000; 
            max_overlap = -0.1 * R; 
            rz = R + half_l * nz - max_overlap;
            res = run_tests(rz, nz, R, half_l, gamma, tol, meshsize, 0.0);
            while ((res.first > tol || res.second > tol) && meshsize < max_meshsize)
            {
                meshsize *= 10;
                std::cout << "- Rerunning tests with increased meshsize = "
                          << meshsize << std::endl; 
                res = run_tests(rz, nz, R, half_l, gamma, tol, meshsize, 0.0);  
            }
            REQUIRE(res.first < tol);
            REQUIRE(res.second < tol);  
        }
    }
}

/**
 * A series of tests for integral2(), which integrates s * \delta_i^\gamma(s). 
 */
TEST_CASE("Tests for auxiliary integral 2", "[integral2()]")
{
    const T R = 0.8;
    const double tol = 1e-8;
    int meshsize = 10000; 
    const int max_meshsize = 1e+7;
    T rz, nz; 
    Array<T, 4, 1> exponents;  
    exponents << 0.5, 1.0, 1.5, 2.0;
    Array<T, 10, 1> angles = Array<T, 10, 1>::Zero();
    for (int i = 0; i < angles.size(); ++i)
        angles(i) = boost::math::constants::half_pi<T>() * (i + 1) / 10.0; 

    auto run_tests = [](const T rz, const T nz, const T R, const T half_l,
                        const T gamma, const double tol, const int meshsize,  
                        const double check_value = std::numeric_limits<double>::quiet_NaN()) -> std::pair<T, T>
    {
        T ss = (R - rz) / nz;
        std::function<T(const T)> func = [rz, nz, R, gamma](const T s) -> T
        {
            return s * overlapGamma<T>(rz, nz, R, s, gamma); 
        };
        T target = integrate(func, -half_l, half_l, meshsize);
        T integral = integral2<T>(rz, nz, R, half_l, gamma, ss); 
        if (!std::isnan(check_value))
            return std::make_pair(abs(integral - target), abs(target - check_value)); 
        else 
            return std::make_pair(abs(integral - target), 0.0); 
    }; 
    
    // For each exponent ... 
    for (int i = 0; i < exponents.size(); ++i)
    {
        T gamma = exponents(i);

        // For each angle ... 
        for (int j = 0; j < angles.size(); ++j)
        {
            std::cout << "Running tests for integral2(), exponent = " << gamma
                      << ", nz = " << sin(angles(j)) << std::endl; 

            // Define the z-orientation
            nz = sin(angles(j));

            // Case 1: Assume the cell has a maximum overlap of 0.2 * R
            meshsize = 10000; 
            T half_l = 0.5;
            T max_overlap = 0.2 * R;
            rz = R + half_l * nz - max_overlap;
            std::pair<T, T> res = run_tests(rz, nz, R, half_l, gamma, tol, meshsize);
            while ((res.first > tol || res.second > tol) && meshsize < max_meshsize)
            {
                meshsize *= 10;
                std::cout << "- Rerunning tests with increased meshsize = "
                          << meshsize << std::endl; 
                res = run_tests(rz, nz, R, half_l, gamma, tol, meshsize);  
            }
            REQUIRE(res.first < tol);
            REQUIRE(res.second < tol);  

            // Case 2: Assume the cell does not contact the surface
            meshsize = 10000; 
            max_overlap = -0.1 * R; 
            rz = R + half_l * nz - max_overlap;
            res = run_tests(rz, nz, R, half_l, gamma, tol, meshsize, 0.0);
            while ((res.first > tol || res.second > tol) && meshsize < max_meshsize)
            {
                meshsize *= 10;
                std::cout << "- Rerunning tests with increased meshsize = "
                          << meshsize << std::endl; 
                res = run_tests(rz, nz, R, half_l, gamma, tol, meshsize, 0.0);  
            }
            REQUIRE(res.first < tol);
            REQUIRE(res.second < tol);  
        }
    }
}

/**
 * A series of tests for integral3(), which integrates s^2 * \delta_i^\gamma(s). 
 */
TEST_CASE("Tests for auxiliary integral 3", "[integral3()]")
{
    const T R = 0.8;
    const double tol = 1e-8;
    int meshsize = 10000;
    const int max_meshsize = 1e+7;
    T rz, nz; 
    Array<T, 4, 1> exponents;  
    exponents << 0.5, 1.0, 1.5, 2.0;
    Array<T, 10, 1> angles = Array<T, 10, 1>::Zero();
    for (int i = 0; i < angles.size(); ++i)
        angles(i) = boost::math::constants::half_pi<T>() * (i + 1) / 10.0; 

    auto run_tests = [](const T rz, const T nz, const T R, const T half_l,
                        const T gamma, const double tol, const int meshsize, 
                        const double check_value = std::numeric_limits<double>::quiet_NaN()) -> std::pair<T, T>
    {
        T ss = (R - rz) / nz;
        std::function<T(const T)> func = [rz, nz, R, gamma](const T s) -> T
        {
            return s * s * overlapGamma<T>(rz, nz, R, s, gamma); 
        };
        T target = integrate(func, -half_l, half_l, meshsize);
        T integral = integral3<T>(rz, nz, R, half_l, gamma, ss);
        if (!std::isnan(check_value))
            return std::make_pair(abs(integral - target), abs(target - check_value));
        else 
            return std::make_pair(abs(integral - target), 0.0); 
    }; 
    
    // For each exponent ... 
    for (int i = 0; i < exponents.size(); ++i)
    {
        T gamma = exponents(i);

        // For each angle ... 
        for (int j = 0; j < angles.size(); ++j)
        {
            std::cout << "Running tests for integral3(), exponent = " << gamma
                      << ", nz = " << sin(angles(j)) << std::endl; 

            // Define the z-orientation
            nz = sin(angles(j));

            // Case 1: Assume the cell has a maximum overlap of 0.2 * R
            meshsize = 10000; 
            T half_l = 0.5;
            T max_overlap = 0.2 * R;
            rz = R + half_l * nz - max_overlap;
            std::pair<T, T> res = run_tests(rz, nz, R, half_l, gamma, tol, meshsize);
            while ((res.first > tol || res.second > tol) && meshsize < max_meshsize)
            {
                meshsize *= 10;
                std::cout << "- Rerunning tests with increased meshsize = "
                          << meshsize << std::endl; 
                res = run_tests(rz, nz, R, half_l, gamma, tol, meshsize);  
            }
            REQUIRE(res.first < tol);
            REQUIRE(res.second < tol);  

            // Case 2: Assume the cell does not contact the surface
            meshsize = 10000; 
            max_overlap = -0.1 * R; 
            rz = R + half_l * nz - max_overlap;
            res = run_tests(rz, nz, R, half_l, gamma, tol, meshsize, 0.0);
            while ((res.first > tol || res.second > tol) && meshsize < max_meshsize)
            {
                meshsize *= 10;
                std::cout << "- Rerunning tests with increased meshsize = "
                          << meshsize << std::endl; 
                res = run_tests(rz, nz, R, half_l, gamma, tol, meshsize, 0.0);  
            }
            REQUIRE(res.first < tol);
            REQUIRE(res.second < tol);  
        }
    }
}

/**
 * A series of tests for integral4(), which integrates \Theta(\delta_i(s)). 
 */
TEST_CASE("Tests for auxiliary integral 4", "[integral4()]")
{
    const T R = 0.8;
    const double tol = 1e-7;
    int meshsize = 1e+7; 
    const int max_meshsize = 1e+9;
    T rz, nz; 
    Array<T, 4, 1> exponents;  
    exponents << 0.5, 1.0, 1.5, 2.0;
    Array<T, 10, 1> angles = Array<T, 10, 1>::Zero();
    for (int i = 0; i < angles.size(); ++i)
        angles(i) = boost::math::constants::half_pi<T>() * (i + 1) / 10.0; 

    auto run_tests = [](const T rz, const T nz, const T R, const T half_l,
                        const double tol, const int meshsize,
                        const double check_value = std::numeric_limits<double>::quiet_NaN()) -> std::pair<T, T>
    {
        T ss = (R - rz) / nz;
        std::function<T(const T)> func = [rz, nz, R](const T s) -> T
        {
            T value = overlap<T>(rz, nz, R, s);
            return (value > 0 ? 1.0 : 0.0); 
        };
        T target = integrate(func, -half_l, half_l, meshsize); 
        T integral = integral4<T>(nz, half_l, ss);
        if (!std::isnan(check_value))
            return std::make_pair(abs(integral - target), abs(target - check_value)); 
        else 
            return std::make_pair(abs(integral - target), 0.0); 
    }; 
    
    // For each angle ... 
    for (int j = 0; j < angles.size(); ++j)
    {
        std::cout << "Running tests for integral4(), nz = " << sin(angles(j)) << std::endl; 

        // Define the z-orientation
        nz = sin(angles(j));

        // Case 1: Assume the cell has a maximum overlap of 0.2 * R
        meshsize = 1e+7;
        T half_l = 0.5;
        T max_overlap = 0.2 * R;
        rz = R + half_l * nz - max_overlap;
        std::pair<T, T> res = run_tests(rz, nz, R, half_l, tol, meshsize);
        while ((res.first > tol || res.second > tol) && meshsize < max_meshsize)
        {
            meshsize *= 10;
            std::cout << "- Rerunning tests with increased meshsize = "
                      << meshsize << std::endl; 
            res = run_tests(rz, nz, R, half_l, tol, meshsize);  
        }
        REQUIRE(res.first < tol);
        REQUIRE(res.second < tol);  

        // Case 2: Assume the cell does not contact the surface
        meshsize = 1e+7;
        max_overlap = -0.1 * R; 
        rz = R + half_l * nz - max_overlap;
        res = run_tests(rz, nz, R, half_l, tol, meshsize, 0.0);
        while ((res.first > tol || res.second > tol) && meshsize < max_meshsize)
        {
            meshsize *= 10;
            std::cout << "- Rerunning tests with increased meshsize = "
                      << meshsize << std::endl; 
            res = run_tests(rz, nz, R, half_l, tol, meshsize, 0.0);  
        }
        REQUIRE(res.first < tol);
        REQUIRE(res.second < tol);  
    }
}

/**
 * A series of tests for integral5(), which integrates s * \Theta(\delta_i(s)). 
 */
TEST_CASE("Tests for auxiliary integral 5", "[integral5()]")
{
    const T R = 0.8;
    const double tol = 1e-7;
    int meshsize = 1e+7; 
    const int max_meshsize = 1e+9;
    T rz, nz; 
    Array<T, 4, 1> exponents;  
    exponents << 0.5, 1.0, 1.5, 2.0;
    Array<T, 10, 1> angles = Array<T, 10, 1>::Zero();
    for (int i = 0; i < angles.size(); ++i)
        angles(i) = boost::math::constants::half_pi<T>() * (i + 1) / 10.0; 

    auto run_tests = [](const T rz, const T nz, const T R, const T half_l,
                        const double tol, const int meshsize,
                        const double check_value = std::numeric_limits<double>::quiet_NaN()) -> std::pair<T, T>
    {
        T ss = (R - rz) / nz;
        std::function<T(const T)> func = [rz, nz, R](const T s) -> T
        {
            T value = overlap<T>(rz, nz, R, s);
            return (value > 0 ? s : 0.0); 
        };
        T target = integrate(func, -half_l, half_l, meshsize);
        T integral = integral5<T>(nz, half_l, ss);
        if (!std::isnan(check_value))
            return std::make_pair(abs(integral - target), abs(target - check_value)); 
        else 
            return std::make_pair(abs(integral - target), 0.0); 
    }; 
    
    // For each angle ... 
    for (int j = 0; j < angles.size(); ++j)
    {
        std::cout << "Running tests for integral5(), nz = " << sin(angles(j)) << std::endl; 

        // Define the z-orientation
        nz = sin(angles(j));

        // Case 1: Assume the cell has a maximum overlap of 0.2 * R
        meshsize = 1e+7;
        T half_l = 0.5;
        T max_overlap = 0.2 * R;
        rz = R + half_l * nz - max_overlap;
        std::pair<T, T> res = run_tests(rz, nz, R, half_l, tol, meshsize);
        while ((res.first > tol || res.second > tol) && meshsize < max_meshsize)
        {
            meshsize *= 10;
            std::cout << "- Rerunning tests with increased meshsize = "
                      << meshsize << std::endl; 
            res = run_tests(rz, nz, R, half_l, tol, meshsize);  
        }
        REQUIRE(res.first < tol);
        REQUIRE(res.second < tol);  

        // Case 2: Assume the cell does not contact the surface
        meshsize = 1e+7;
        max_overlap = -0.1 * R; 
        rz = R + half_l * nz - max_overlap;
        res = run_tests(rz, nz, R, half_l, tol, meshsize, 0.0);
        while ((res.first > tol || res.second > tol) && meshsize < max_meshsize)
        {
            meshsize *= 10;
            std::cout << "- Rerunning tests with increased meshsize = "
                      << meshsize << std::endl; 
            res = run_tests(rz, nz, R, half_l, tol, meshsize, 0.0);  
        }
        REQUIRE(res.first < tol);
        REQUIRE(res.second < tol);  
    }
}

/**
 * A series of tests for integral6(), which integrates s^2 * \Theta(\delta_i(s)). 
 */
TEST_CASE("Tests for auxiliary integral 6", "[integral6()]")
{
    const T R = 0.8;
    const double tol = 1e-7;
    int meshsize = 1e+7; 
    const int max_meshsize = 1e+9;
    T rz, nz; 
    Array<T, 4, 1> exponents;  
    exponents << 0.5, 1.0, 1.5, 2.0;
    Array<T, 10, 1> angles = Array<T, 10, 1>::Zero();
    for (int i = 0; i < angles.size(); ++i)
        angles(i) = boost::math::constants::half_pi<T>() * (i + 1) / 10.0; 

    auto run_tests = [](const T rz, const T nz, const T R, const T half_l,
                        const double tol, const int meshsize, 
                        const double check_value = std::numeric_limits<double>::quiet_NaN()) -> std::pair<T, T>
    {
        T ss = (R - rz) / nz;
        std::function<T(const T)> func = [rz, nz, R](const T s) -> T
        {
            T value = overlap<T>(rz, nz, R, s);
            if (value > 0)
                return s * s; 
            else 
                return 0.0;
        };
        T target = integrate(func, -half_l, half_l, meshsize);
        T integral = integral6<T>(nz, half_l, ss);
        if (!std::isnan(check_value))
            return std::make_pair(abs(integral - target), abs(target - check_value));
        else 
            return std::make_pair(abs(integral - target), 0.0); 
    }; 
    
    // For each angle ... 
    for (int j = 0; j < angles.size(); ++j)
    {
        std::cout << "Running tests for integral6(), nz = " << sin(angles(j)) << std::endl; 

        // Define the z-orientation
        nz = sin(angles(j));

        // Case 1: Assume the cell has a maximum overlap of 0.2 * R
        meshsize = 1e+7;
        T half_l = 0.5;
        T max_overlap = 0.2 * R;
        rz = R + half_l * nz - max_overlap;
        std::pair<T, T> res = run_tests(rz, nz, R, half_l, tol, meshsize);
        while ((res.first > tol || res.second > tol) && meshsize < max_meshsize)
        {
            meshsize *= 10;
            std::cout << "- Rerunning tests with increased meshsize = "
                      << meshsize << std::endl; 
            res = run_tests(rz, nz, R, half_l, tol, meshsize);  
        }
        REQUIRE(res.first < tol);
        REQUIRE(res.second < tol); 

        // Case 2: Assume the cell does not contact the surface
        meshsize = 1e+7;
        max_overlap = -0.1 * R; 
        rz = R + half_l * nz - max_overlap;
        res = run_tests(rz, nz, R, half_l, tol, meshsize, 0.0);
        while ((res.first > tol || res.second > tol) && meshsize < max_meshsize)
        {
            meshsize *= 10;
            std::cout << "- Rerunning tests with increased meshsize = "
                      << meshsize << std::endl; 
            res = run_tests(rz, nz, R, half_l, tol, meshsize, 0.0);  
        }
        REQUIRE(res.first < tol);
        REQUIRE(res.second < tol); 
    }
}

/**
 * A series of tests for areaIntegral1(), which integrates the cell-surface 
 * contact area density. 
 */
TEST_CASE("Tests for area integral 1", "[areaIntegral1()]")
{
    const T R = 0.8;
    const double tol = 1e-7;
    int meshsize = 1e+7; 
    const int max_meshsize = 1e+9;
    T rz, nz; 
    Array<T, 4, 1> exponents;  
    exponents << 0.5, 1.0, 1.5, 2.0;
    Array<T, 10, 1> angles = Array<T, 10, 1>::Zero();
    for (int i = 0; i < angles.size(); ++i)
        angles(i) = boost::math::constants::half_pi<T>() * (i + 1) / 10.0; 

    auto run_tests = [](const T rz, const T nz, const T R, const T half_l,
                        const double tol, const int meshsize, 
                        const double check_value = std::numeric_limits<double>::quiet_NaN()) -> std::pair<T, T>
    {
        T ss = (R - rz) / nz;
        std::function<T(const T)> func = [rz, nz, R](const T s) -> T
        {
            T d = overlap<T>(rz, nz, R, s); 
            T sqrt_d = overlapGamma<T>(rz, nz, R, s, 0.5);
            T step = (d > 0 ? 1.0 : 0.0); 
            return sqrt(R) * (1 - nz * nz) * sqrt_d + boost::math::constants::pi<T>() * R * nz * nz * step;  
        };
        T target = integrate(func, -half_l, half_l, meshsize);
        T integral = areaIntegral1<T>(rz, nz, R, half_l, ss); 
        if (!std::isnan(check_value))
            return std::make_pair(abs(integral - target), abs(target - check_value)); 
        else 
            return std::make_pair(abs(integral - target), 0.0); 
    }; 
    
    // For each angle ... 
    for (int j = 0; j < angles.size(); ++j)
    {
        std::cout << "Running tests for areaIntegral1(), nz = " << sin(angles(j)) << std::endl; 

        // Define the z-orientation
        nz = sin(angles(j));

        // Case 1: Assume the cell has a maximum overlap of 0.2 * R
        meshsize = 1e+7;
        T half_l = 0.5;
        T max_overlap = 0.2 * R;
        rz = R + half_l * nz - max_overlap;
        std::pair<T, T> res = run_tests(rz, nz, R, half_l, tol, meshsize);
        while ((res.first > tol || res.second > tol) && meshsize < max_meshsize)
        {
            meshsize *= 10;
            std::cout << "- Rerunning tests with increased meshsize = "
                      << meshsize << std::endl; 
            res = run_tests(rz, nz, R, half_l, tol, meshsize);  
        }
        REQUIRE(res.first < tol);
        REQUIRE(res.second < tol);  

        // Case 2: Assume the cell does not contact the surface
        meshsize = 1e+7;
        max_overlap = -0.1 * R; 
        rz = R + half_l * nz - max_overlap;
        res = run_tests(rz, nz, R, half_l, tol, meshsize, 0.0);
        while ((res.first > tol || res.second > tol) && meshsize < max_meshsize)
        {
            meshsize *= 10;
            std::cout << "- Rerunning tests with increased meshsize = "
                      << meshsize << std::endl; 
            res = run_tests(rz, nz, R, half_l, tol, meshsize, 0.0);  
        }
        REQUIRE(res.first < tol);
        REQUIRE(res.second < tol);  
    }
}

/**
 * A series of tests for areaIntegral2(), which integrates s times the 
 * cell-surface contact area density. 
 */
TEST_CASE("Tests for area integral 2", "[areaIntegral2()]")
{
    const T R = 0.8;
    const double tol = 1e-7;
    int meshsize = 1e+7; 
    const int max_meshsize = 1e+9;
    T rz, nz; 
    Array<T, 4, 1> exponents;  
    exponents << 0.5, 1.0, 1.5, 2.0;
    Array<T, 10, 1> angles = Array<T, 10, 1>::Zero();
    for (int i = 0; i < angles.size(); ++i)
        angles(i) = boost::math::constants::half_pi<T>() * (i + 1) / 10.0; 

    auto run_tests = [](const T rz, const T nz, const T R, const T half_l,
                        const double tol, const int meshsize,
                        const double check_value = std::numeric_limits<double>::quiet_NaN()) -> std::pair<T, T>
    {
        T ss = (R - rz) / nz;
        std::function<T(const T)> func = [rz, nz, R](const T s) -> T
        {
            T d = overlap<T>(rz, nz, R, s); 
            T sqrt_d = overlapGamma<T>(rz, nz, R, s, 0.5);
            T step = (d > 0 ? 1.0 : 0.0); 
            return s * (sqrt(R) * (1 - nz * nz) * sqrt_d + boost::math::constants::pi<T>() * R * nz * nz * step);  
        };
        T target = integrate(func, -half_l, half_l, meshsize);
        T integral = areaIntegral2<T>(rz, nz, R, half_l, ss); 
        if (!std::isnan(check_value))
            return std::make_pair(abs(integral - target), abs(target - check_value)); 
        else 
            return std::make_pair(abs(integral - target), 0.0); 
    }; 
    
    // For each angle ... 
    for (int j = 0; j < angles.size(); ++j)
    {
        std::cout << "Running tests for areaIntegral2(), nz = " << sin(angles(j)) << std::endl; 

        // Define the z-orientation
        nz = sin(angles(j));

        // Case 1: Assume the cell has a maximum overlap of 0.2 * R
        meshsize = 1e+7;
        T half_l = 0.5;
        T max_overlap = 0.2 * R;
        rz = R + half_l * nz - max_overlap;
        std::pair<T, T> res = run_tests(rz, nz, R, half_l, tol, meshsize);
        while ((res.first > tol || res.second > tol) && meshsize < max_meshsize)
        {
            meshsize *= 10;
            std::cout << "- Rerunning tests with increased meshsize = "
                      << meshsize << std::endl; 
            res = run_tests(rz, nz, R, half_l, tol, meshsize);  
        }
        REQUIRE(res.first < tol);
        REQUIRE(res.second < tol);  

        // Case 2: Assume the cell does not contact the surface
        meshsize = 1e+7;
        max_overlap = -0.1 * R; 
        rz = R + half_l * nz - max_overlap;
        res = run_tests(rz, nz, R, half_l, tol, meshsize, 0.0);
        while ((res.first > tol || res.second > tol) && meshsize < max_meshsize)
        {
            meshsize *= 10;
            std::cout << "- Rerunning tests with increased meshsize = "
                      << meshsize << std::endl; 
            res = run_tests(rz, nz, R, half_l, tol, meshsize, 0.0);  
        }
        REQUIRE(res.first < tol);
        REQUIRE(res.second < tol);  
    }
}

/**
 * A series of tests for areaIntegral3(), which integrates s^2 times the 
 * cell-surface contact area density. 
 */
TEST_CASE("Tests for area integral 3", "[areaIntegral3()]")
{
    const T R = 0.8;
    const double tol = 1e-7;
    int meshsize = 1e+7; 
    const int max_meshsize = 1e+9;
    T rz, nz; 
    Array<T, 4, 1> exponents;  
    exponents << 0.5, 1.0, 1.5, 2.0;
    Array<T, 10, 1> angles = Array<T, 10, 1>::Zero();
    for (int i = 0; i < angles.size(); ++i)
        angles(i) = boost::math::constants::half_pi<T>() * (i + 1) / 10.0; 

    auto run_tests = [](const T rz, const T nz, const T R, const T half_l,
                        const double tol, const int meshsize,
                        const double check_value = std::numeric_limits<double>::quiet_NaN()) -> std::pair<T, T>
    {
        T ss = (R - rz) / nz;
        std::function<T(const T)> func = [rz, nz, R](const T s) -> T
        {
            T d = overlap<T>(rz, nz, R, s); 
            T sqrt_d = overlapGamma<T>(rz, nz, R, s, 0.5);
            T step = (d > 0 ? 1.0 : 0.0); 
            return s * s * (sqrt(R) * (1 - nz * nz) * sqrt_d + boost::math::constants::pi<T>() * R * nz * nz * step);  
        };
        T target = integrate(func, -half_l, half_l, meshsize);
        T integral = areaIntegral3<T>(rz, nz, R, half_l, ss); 
        if (!std::isnan(check_value))
            return std::make_pair(abs(integral - target), abs(target - check_value)); 
        else 
            return std::make_pair(abs(integral - target), 0.0); 
    }; 
    
    // For each angle ... 
    for (int j = 0; j < angles.size(); ++j)
    {
        std::cout << "Running tests for areaIntegral3(), nz = " << sin(angles(j)) << std::endl; 

        // Define the z-orientation
        nz = sin(angles(j));

        // Case 1: Assume the cell has a maximum overlap of 0.2 * R
        meshsize = 1e+7;
        T half_l = 0.5;
        T max_overlap = 0.2 * R;
        rz = R + half_l * nz - max_overlap;
        std::pair<T, T> res = run_tests(rz, nz, R, half_l, tol, meshsize);
        while ((res.first > tol || res.second > tol) && meshsize < max_meshsize)
        {
            meshsize *= 10;
            std::cout << "- Rerunning tests with increased meshsize = "
                      << meshsize << std::endl; 
            res = run_tests(rz, nz, R, half_l, tol, meshsize);  
        }
        REQUIRE(res.first < tol);
        REQUIRE(res.second < tol); 

        // Case 2: Assume the cell does not contact the surface
        meshsize = 1e+7;
        max_overlap = -0.1 * R; 
        rz = R + half_l * nz - max_overlap;
        res = run_tests(rz, nz, R, half_l, tol, meshsize, 0.0);
        while ((res.first > tol || res.second > tol) && meshsize < max_meshsize)
        {
            meshsize *= 10;
            std::cout << "- Rerunning tests with increased meshsize = "
                      << meshsize << std::endl; 
            res = run_tests(rz, nz, R, half_l, tol, meshsize, 0.0);  
        }
        REQUIRE(res.first < tol);
        REQUIRE(res.second < tol); 
    }
}

/**
 * A series of tests for areaIntegrals(), which computes all three area 
 * integrals in one fell swoop.
 */
TEST_CASE("Tests for all three area integrals", "[areaIntegrals()]")
{
    const T R = 0.8;
    const double tol = 1e-7;
    int meshsize = 1e+7; 
    const int max_meshsize = 1e+9;
    T rz, nz; 
    Array<T, 4, 1> exponents;  
    exponents << 0.5, 1.0, 1.5, 2.0;
    Array<T, 10, 1> angles = Array<T, 10, 1>::Zero();
    for (int i = 0; i < angles.size(); ++i)
        angles(i) = boost::math::constants::half_pi<T>() * (i + 1) / 10.0; 

    // For each angle ... 
    for (int j = 0; j < angles.size(); ++j)
    {
        std::cout << "Running tests for areaIntegrals(), nz = " << sin(angles(j)) << std::endl; 

        // Define the z-orientation
        nz = sin(angles(j));

        // Case 1: Assume the cell has a maximum overlap of 0.2 * R
        meshsize = 1e+7;
        T half_l = 0.5;
        T max_overlap = 0.2 * R;
        rz = R + half_l * nz - max_overlap;
        T ss = (R - rz) / nz; 
        T area1 = areaIntegral1<T>(rz, nz, R, half_l, ss);
        T area2 = areaIntegral2<T>(rz, nz, R, half_l, ss); 
        T area3 = areaIntegral3<T>(rz, nz, R, half_l, ss); 
        auto result = areaIntegrals<T>(rz, nz, R, half_l, ss); 
        REQUIRE_THAT(
            static_cast<double>(area1 - std::get<0>(result)),
            Catch::Matchers::WithinAbs(0, tol)
        );
        REQUIRE_THAT(
            static_cast<double>(area2 - std::get<1>(result)), 
            Catch::Matchers::WithinAbs(0, tol)
        );
        REQUIRE_THAT(
            static_cast<double>(area3 - std::get<2>(result)), 
            Catch::Matchers::WithinAbs(0, tol)
        );  
        
        // Case 2: Assume the cell does not contact the surface
        meshsize = 1e+7;
        max_overlap = -0.1 * R; 
        rz = R + half_l * nz - max_overlap;
        ss = (R - rz) / nz; 
        area1 = areaIntegral1<T>(rz, nz, R, half_l, ss);
        area2 = areaIntegral2<T>(rz, nz, R, half_l, ss); 
        area3 = areaIntegral3<T>(rz, nz, R, half_l, ss); 
        result = areaIntegrals<T>(rz, nz, R, half_l, ss); 
        REQUIRE_THAT(
            static_cast<double>(area1 - std::get<0>(result)),
            Catch::Matchers::WithinAbs(0, tol)
        );
        REQUIRE_THAT(
            static_cast<double>(area2 - std::get<1>(result)), 
            Catch::Matchers::WithinAbs(0, tol)
        );
        REQUIRE_THAT(
            static_cast<double>(area3 - std::get<2>(result)), 
            Catch::Matchers::WithinAbs(0, tol)
        );  
    }
}

