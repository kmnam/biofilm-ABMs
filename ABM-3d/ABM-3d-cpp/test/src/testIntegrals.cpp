/**
 * Test module for the auxiliary integral calculations in `integrals.hpp`.
 *
 * Authors:
 *     Kee-Myoung Nam
 *
 * Last updated:
 *     2/26/2025
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

typedef double T; 

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
    const T delta = 1e-6;
    T rz, nz; 
    Array<T, 4, 1> exponents;  
    exponents << 0.5, 1.0, 1.5, 2.0;
    Array<T, 10, 1> angles = Array<T, 10, 1>::Zero();
    for (int i = 0; i < angles.size(); ++i)
        angles(i) = boost::math::constants::half_pi<T>() * (i + 1) / 10.0; 

    auto run_tests = [](const T rz, const T nz, const T R, const T half_l,
                        const T gamma, const T delta,
                        const T check_value = std::numeric_limits<T>::quiet_NaN())
    {
        T ss = (R - rz) / nz;
        std::function<T(const T)> func = [rz, nz, R, gamma](const T s)
        {
            return overlapGamma<T>(rz, nz, R, s, gamma); 
        };
        T target = integrate(func, -half_l, half_l, 10000);
        REQUIRE_THAT(
            integral1<T>(rz, nz, R, half_l, gamma, ss),
            Catch::Matchers::WithinAbs(target, delta)
        );
        if (!std::isnan(check_value))
            REQUIRE_THAT(target, Catch::Matchers::WithinAbs(check_value, delta)); 
    }; 
    
    // For each exponent ... 
    for (int i = 0; i < exponents.size(); ++i)
    {
        T gamma = exponents(i);

        // For each angle ... 
        for (int j = 0; j < angles.size(); ++j)
        {
            // Define the z-orientation
            nz = sin(angles(j));

            // Case 1: Assume the cell has a maximum overlap of 0.2 * R
            T half_l = 0.5;
            T max_overlap = 0.2 * R;
            rz = R + half_l * nz - max_overlap;
            run_tests(rz, nz, R, half_l, gamma, delta);

            // Case 2: Assume the cell does not contact the surface
            max_overlap = -0.1 * R; 
            rz = R + half_l * nz - max_overlap;
            run_tests(rz, nz, R, half_l, gamma, delta, 0.0);
        }
    }
}

/**
 * A series of tests for integral2(), which integrates s * \delta_i^\gamma(s). 
 */
TEST_CASE("Tests for auxiliary integral 2", "[integral2()]")
{
    const T R = 0.8;
    const T delta = 1e-6;
    T rz, nz; 
    Array<T, 4, 1> exponents;  
    exponents << 0.5, 1.0, 1.5, 2.0;
    Array<T, 10, 1> angles = Array<T, 10, 1>::Zero();
    for (int i = 0; i < angles.size(); ++i)
        angles(i) = boost::math::constants::half_pi<T>() * (i + 1) / 10.0; 

    auto run_tests = [](const T rz, const T nz, const T R, const T half_l,
                        const T gamma, const T delta,
                        const T check_value = std::numeric_limits<T>::quiet_NaN())
    {
        T ss = (R - rz) / nz;
        std::function<T(const T)> func = [rz, nz, R, gamma](const T s)
        {
            return s * overlapGamma<T>(rz, nz, R, s, gamma); 
        };
        T target = integrate(func, -half_l, half_l, 10000);
        REQUIRE_THAT(
            integral2<T>(rz, nz, R, half_l, gamma, ss),
            Catch::Matchers::WithinAbs(target, delta)
        );
        if (!std::isnan(check_value))
            REQUIRE_THAT(target, Catch::Matchers::WithinAbs(check_value, delta)); 
    }; 
    
    // For each exponent ... 
    for (int i = 0; i < exponents.size(); ++i)
    {
        T gamma = exponents(i);

        // For each angle ... 
        for (int j = 0; j < angles.size(); ++j)
        {
            // Define the z-orientation
            nz = sin(angles(j));

            // Case 1: Assume the cell has a maximum overlap of 0.2 * R
            T half_l = 0.5;
            T max_overlap = 0.2 * R;
            rz = R + half_l * nz - max_overlap;
            run_tests(rz, nz, R, half_l, gamma, delta);

            // Case 2: Assume the cell does not contact the surface
            max_overlap = -0.1 * R; 
            rz = R + half_l * nz - max_overlap;
            run_tests(rz, nz, R, half_l, gamma, delta, 0.0);
        }
    }
}

/**
 * A series of tests for integral3(), which integrates s^2 * \delta_i^\gamma(s). 
 */
TEST_CASE("Tests for auxiliary integral 3", "[integral3()]")
{
    const T R = 0.8;
    const T delta = 1e-6;
    T rz, nz; 
    Array<T, 4, 1> exponents;  
    exponents << 0.5, 1.0, 1.5, 2.0;
    Array<T, 10, 1> angles = Array<T, 10, 1>::Zero();
    for (int i = 0; i < angles.size(); ++i)
        angles(i) = boost::math::constants::half_pi<T>() * (i + 1) / 10.0; 

    auto run_tests = [](const T rz, const T nz, const T R, const T half_l,
                        const T gamma, const T delta,
                        const T check_value = std::numeric_limits<T>::quiet_NaN())
    {
        T ss = (R - rz) / nz;
        std::function<T(const T)> func = [rz, nz, R, gamma](const T s)
        {
            return s * s * overlapGamma<T>(rz, nz, R, s, gamma); 
        };
        T target = integrate(func, -half_l, half_l, 10000);
        REQUIRE_THAT(
            integral3<T>(rz, nz, R, half_l, gamma, ss),
            Catch::Matchers::WithinAbs(target, delta)
        );
        if (!std::isnan(check_value))
            REQUIRE_THAT(target, Catch::Matchers::WithinAbs(check_value, delta)); 
    }; 
    
    // For each exponent ... 
    for (int i = 0; i < exponents.size(); ++i)
    {
        T gamma = exponents(i);

        // For each angle ... 
        for (int j = 0; j < angles.size(); ++j)
        {
            // Define the z-orientation
            nz = sin(angles(j));

            // Case 1: Assume the cell has a maximum overlap of 0.2 * R
            T half_l = 0.5;
            T max_overlap = 0.2 * R;
            rz = R + half_l * nz - max_overlap;
            run_tests(rz, nz, R, half_l, gamma, delta);

            // Case 2: Assume the cell does not contact the surface
            max_overlap = -0.1 * R; 
            rz = R + half_l * nz - max_overlap;
            run_tests(rz, nz, R, half_l, gamma, delta, 0.0);
        }
    }
}

/**
 * A series of tests for integral4(), which integrates \Theta(\delta_i(s)). 
 */
TEST_CASE("Tests for auxiliary integral 4", "[integral4()]")
{
    const T R = 0.8;
    const T delta = 1e-6;
    T rz, nz; 
    Array<T, 4, 1> exponents;  
    exponents << 0.5, 1.0, 1.5, 2.0;
    Array<T, 10, 1> angles = Array<T, 10, 1>::Zero();
    for (int i = 0; i < angles.size(); ++i)
        angles(i) = boost::math::constants::half_pi<T>() * (i + 1) / 10.0; 

    auto run_tests = [](const T rz, const T nz, const T R, const T half_l,
                        const T delta,
                        const T check_value = std::numeric_limits<T>::quiet_NaN())
    {
        T ss = (R - rz) / nz;
        std::function<T(const T)> func = [rz, nz, R](const T s)
        {
            T value = overlap<T>(rz, nz, R, s);
            return (value > 0 ? 1.0 : 0.0); 
        };
        T target = integrate(func, -half_l, half_l, 1000000);
        REQUIRE_THAT(
            integral4<T>(half_l, ss),
            Catch::Matchers::WithinAbs(target, delta)
        );
        if (!std::isnan(check_value))
            REQUIRE_THAT(target, Catch::Matchers::WithinAbs(check_value, delta)); 
    }; 
    
    // For each angle ... 
    for (int j = 0; j < angles.size(); ++j)
    {
        // Define the z-orientation
        nz = sin(angles(j));

        // Case 1: Assume the cell has a maximum overlap of 0.2 * R
        T half_l = 0.5;
        T max_overlap = 0.2 * R;
        rz = R + half_l * nz - max_overlap;
        run_tests(rz, nz, R, half_l, delta);

        // Case 2: Assume the cell does not contact the surface
        max_overlap = -0.1 * R; 
        rz = R + half_l * nz - max_overlap;
        run_tests(rz, nz, R, half_l, delta, 0.0);
    }
}

/**
 * A series of tests for integral5(), which integrates s * \Theta(\delta_i(s)). 
 */
TEST_CASE("Tests for auxiliary integral 5", "[integral5()]")
{
    const T R = 0.8;
    const T delta = 1e-6;
    T rz, nz; 
    Array<T, 4, 1> exponents;  
    exponents << 0.5, 1.0, 1.5, 2.0;
    Array<T, 10, 1> angles = Array<T, 10, 1>::Zero();
    for (int i = 0; i < angles.size(); ++i)
        angles(i) = boost::math::constants::half_pi<T>() * (i + 1) / 10.0; 

    auto run_tests = [](const T rz, const T nz, const T R, const T half_l,
                        const T delta,
                        const T check_value = std::numeric_limits<T>::quiet_NaN())
    {
        T ss = (R - rz) / nz;
        std::function<T(const T)> func = [rz, nz, R](const T s)
        {
            T value = overlap<T>(rz, nz, R, s);
            return (value > 0 ? s : 0.0); 
        };
        T target = integrate(func, -half_l, half_l, 1000000);
        REQUIRE_THAT(
            integral5<T>(rz, nz, R, half_l, ss),
            Catch::Matchers::WithinAbs(target, delta)
        );
        if (!std::isnan(check_value))
            REQUIRE_THAT(target, Catch::Matchers::WithinAbs(check_value, delta)); 
    }; 
    
    // For each angle ... 
    for (int j = 0; j < angles.size(); ++j)
    {
        // Define the z-orientation
        nz = sin(angles(j));

        // Case 1: Assume the cell has a maximum overlap of 0.2 * R
        T half_l = 0.5;
        T max_overlap = 0.2 * R;
        rz = R + half_l * nz - max_overlap;
        run_tests(rz, nz, R, half_l, delta);

        // Case 2: Assume the cell does not contact the surface
        max_overlap = -0.1 * R; 
        rz = R + half_l * nz - max_overlap;
        run_tests(rz, nz, R, half_l, delta, 0.0);
    }
}

/**
 * A series of tests for integral6(), which integrates s^2 * \Theta(\delta_i(s)). 
 */
TEST_CASE("Tests for auxiliary integral 6", "[integral6()]")
{
    const T R = 0.8;
    const T delta = 1e-6;
    T rz, nz; 
    Array<T, 4, 1> exponents;  
    exponents << 0.5, 1.0, 1.5, 2.0;
    Array<T, 10, 1> angles = Array<T, 10, 1>::Zero();
    for (int i = 0; i < angles.size(); ++i)
        angles(i) = boost::math::constants::half_pi<T>() * (i + 1) / 10.0; 

    auto run_tests = [](const T rz, const T nz, const T R, const T half_l,
                        const T delta,
                        const T check_value = std::numeric_limits<T>::quiet_NaN())
    {
        T ss = (R - rz) / nz;
        std::function<T(const T)> func = [rz, nz, R](const T s)
        {
            T value = overlap<T>(rz, nz, R, s);
            return (value > 0 ? s * s : 0.0); 
        };
        T target = integrate(func, -half_l, half_l, 1000000);
        REQUIRE_THAT(
            integral6<T>(rz, nz, R, half_l, ss),
            Catch::Matchers::WithinAbs(target, delta)
        );
        if (!std::isnan(check_value))
            REQUIRE_THAT(target, Catch::Matchers::WithinAbs(check_value, delta)); 
    }; 
    
    // For each angle ... 
    for (int j = 0; j < angles.size(); ++j)
    {
        // Define the z-orientation
        nz = sin(angles(j));

        // Case 1: Assume the cell has a maximum overlap of 0.2 * R
        T half_l = 0.5;
        T max_overlap = 0.2 * R;
        rz = R + half_l * nz - max_overlap;
        run_tests(rz, nz, R, half_l, delta);

        // Case 2: Assume the cell does not contact the surface
        max_overlap = -0.1 * R; 
        rz = R + half_l * nz - max_overlap;
        run_tests(rz, nz, R, half_l, delta, 0.0);
    }
}

/**
 * A series of tests for areaIntegral1(), which integrates the cell-surface 
 * contact area density. 
 */
TEST_CASE("Tests for area integral 1", "[areaIntegral1()]")
{
    const T R = 0.8;
    const T delta = 1e-6;
    T rz, nz; 
    Array<T, 4, 1> exponents;  
    exponents << 0.5, 1.0, 1.5, 2.0;
    Array<T, 10, 1> angles = Array<T, 10, 1>::Zero();
    for (int i = 0; i < angles.size(); ++i)
        angles(i) = boost::math::constants::half_pi<T>() * (i + 1) / 10.0; 

    auto run_tests = [](const T rz, const T nz, const T R, const T half_l,
                        const T delta,
                        const T check_value = std::numeric_limits<T>::quiet_NaN())
    {
        T ss = (R - rz) / nz;
        std::function<T(const T)> func = [rz, nz, R](const T s)
        {
            T d = overlap<T>(rz, nz, R, s); 
            T sqrt_d = overlapGamma<T>(rz, nz, R, s, 0.5);
            T step = (d > 0 ? 1.0 : 0.0); 
            return sqrt(R) * (1 - nz * nz) * sqrt_d + boost::math::constants::pi<T>() * R * nz * nz * step;  
        };
        T target = integrate(func, -half_l, half_l, 1000000);
        REQUIRE_THAT(
            areaIntegral1<T>(rz, nz, R, half_l, ss),
            Catch::Matchers::WithinAbs(target, delta)
        );
        if (!std::isnan(check_value))
            REQUIRE_THAT(target, Catch::Matchers::WithinAbs(check_value, delta)); 
    }; 
    
    // For each angle ... 
    for (int j = 0; j < angles.size(); ++j)
    {
        // Define the z-orientation
        nz = sin(angles(j));

        // Case 1: Assume the cell has a maximum overlap of 0.2 * R
        T half_l = 0.5;
        T max_overlap = 0.2 * R;
        rz = R + half_l * nz - max_overlap;
        run_tests(rz, nz, R, half_l, delta);

        // Case 2: Assume the cell does not contact the surface
        max_overlap = -0.1 * R; 
        rz = R + half_l * nz - max_overlap;
        run_tests(rz, nz, R, half_l, delta, 0.0);
    }
}

/**
 * A series of tests for areaIntegral2(), which integrates s times the 
 * cell-surface contact area density. 
 */
TEST_CASE("Tests for area integral 2", "[areaIntegral2()]")
{
    const T R = 0.8;
    const T delta = 1e-6;
    T rz, nz; 
    Array<T, 4, 1> exponents;  
    exponents << 0.5, 1.0, 1.5, 2.0;
    Array<T, 10, 1> angles = Array<T, 10, 1>::Zero();
    for (int i = 0; i < angles.size(); ++i)
        angles(i) = boost::math::constants::half_pi<T>() * (i + 1) / 10.0; 

    auto run_tests = [](const T rz, const T nz, const T R, const T half_l,
                        const T delta,
                        const T check_value = std::numeric_limits<T>::quiet_NaN())
    {
        T ss = (R - rz) / nz;
        std::function<T(const T)> func = [rz, nz, R](const T s)
        {
            T d = overlap<T>(rz, nz, R, s); 
            T sqrt_d = overlapGamma<T>(rz, nz, R, s, 0.5);
            T step = (d > 0 ? 1.0 : 0.0); 
            return s * (sqrt(R) * (1 - nz * nz) * sqrt_d + boost::math::constants::pi<T>() * R * nz * nz * step);  
        };
        T target = integrate(func, -half_l, half_l, 1000000);
        REQUIRE_THAT(
            areaIntegral2<T>(rz, nz, R, half_l, ss),
            Catch::Matchers::WithinAbs(target, delta)
        );
        if (!std::isnan(check_value))
            REQUIRE_THAT(target, Catch::Matchers::WithinAbs(check_value, delta)); 
    }; 
    
    // For each angle ... 
    for (int j = 0; j < angles.size(); ++j)
    {
        // Define the z-orientation
        nz = sin(angles(j));

        // Case 1: Assume the cell has a maximum overlap of 0.2 * R
        T half_l = 0.5;
        T max_overlap = 0.2 * R;
        rz = R + half_l * nz - max_overlap;
        run_tests(rz, nz, R, half_l, delta);

        // Case 2: Assume the cell does not contact the surface
        max_overlap = -0.1 * R; 
        rz = R + half_l * nz - max_overlap;
        run_tests(rz, nz, R, half_l, delta, 0.0);
    }
}

/**
 * A series of tests for areaIntegral3(), which integrates s^2 times the 
 * cell-surface contact area density. 
 */
TEST_CASE("Tests for area integral 3", "[areaIntegral3()]")
{
    const T R = 0.8;
    const T delta = 1e-6;
    T rz, nz; 
    Array<T, 4, 1> exponents;  
    exponents << 0.5, 1.0, 1.5, 2.0;
    Array<T, 10, 1> angles = Array<T, 10, 1>::Zero();
    for (int i = 0; i < angles.size(); ++i)
        angles(i) = boost::math::constants::half_pi<T>() * (i + 1) / 10.0; 

    auto run_tests = [](const T rz, const T nz, const T R, const T half_l,
                        const T delta,
                        const T check_value = std::numeric_limits<T>::quiet_NaN())
    {
        T ss = (R - rz) / nz;
        std::function<T(const T)> func = [rz, nz, R](const T s)
        {
            T d = overlap<T>(rz, nz, R, s); 
            T sqrt_d = overlapGamma<T>(rz, nz, R, s, 0.5);
            T step = (d > 0 ? 1.0 : 0.0); 
            return s * s * (sqrt(R) * (1 - nz * nz) * sqrt_d + boost::math::constants::pi<T>() * R * nz * nz * step);  
        };
        T target = integrate(func, -half_l, half_l, 1000000);
        REQUIRE_THAT(
            areaIntegral3<T>(rz, nz, R, half_l, ss),
            Catch::Matchers::WithinAbs(target, delta)
        );
        if (!std::isnan(check_value))
            REQUIRE_THAT(target, Catch::Matchers::WithinAbs(check_value, delta)); 
    }; 
    
    // For each angle ... 
    for (int j = 0; j < angles.size(); ++j)
    {
        // Define the z-orientation
        nz = sin(angles(j));

        // Case 1: Assume the cell has a maximum overlap of 0.2 * R
        T half_l = 0.5;
        T max_overlap = 0.2 * R;
        rz = R + half_l * nz - max_overlap;
        run_tests(rz, nz, R, half_l, delta);

        // Case 2: Assume the cell does not contact the surface
        max_overlap = -0.1 * R; 
        rz = R + half_l * nz - max_overlap;
        run_tests(rz, nz, R, half_l, delta, 0.0);
    }
}

