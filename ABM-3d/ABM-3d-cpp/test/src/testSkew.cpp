/**
 * Test simulations with one non-growing cell that forms different angles
 * with the surface. 
 * 
 * Authors:
 *     Kee-Myoung Nam
 *
 * Last updated:
 *     2/27/2025
 */
#include <iostream>
#include <cmath>
#include <functional>
#include <Eigen/Dense>
#include <boost/math/constants/constants.hpp>
#include <boost/multiprecision/mpfr.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "../../include/indices.hpp"
#include "../../include/simulation.hpp"
#include "../../include/switch.hpp"

using namespace Eigen;

typedef double T; 

using std::sin; 
using boost::multiprecision::sin; 
using std::cos; 
using boost::multiprecision::cos; 
using std::pow; 
using boost::multiprecision::pow;

/* ------------------------------------------------------------------- //
 *                           HELPER FUNCTIONS                          //
 * ------------------------------------------------------------------- */
/**
 *
 * @param rz
 * @param nz
 * @param half_l
 * @param R
 * @param E0
 * @param sigma0
 * @param nz_threshold
 * @returns 
 */
bool moreAdhesiveThanRepulsive(const T rz, const T nz, const T half_l, const T R,
                               const T E0, const T sigma0, const T nz_threshold)
{
    if (nz > nz_threshold)
    {
        // Compute the cell-surface repulsion energy
        T ss = (R - rz) / nz; 
        T int1 = integral1<T>(rz, nz, R, half_l, 2.0, ss);
        T int2 = integral1<T>(rz, nz, R, half_l, 1.5, ss);
        T overlap = pow(R, -0.5) * (1 - nz * nz) * int1 + (4.0 / 3.0) * nz * nz * int2;  
        T Erep = E0 * sqrt(R) * overlap;

        // Compute the cell-surface adhesion energy 
        T int3 = areaIntegral1<T>(rz, nz, R, half_l, ss);
        T Eadh = -sigma0 * int3; 

        return (Eadh >= Erep);  
    }
    else 
    {
        // Compare against the equilibrium z-coordinate 
        T adhesion = sigma0 / (R * E0);
        T target = R * (1 - pow(0.25 * adhesion, 2./3.));
        return (rz >= target); 
    } 
} 

/* ------------------------------------------------------------------- //
 *                             TEST MODULES                            //
 * ------------------------------------------------------------------- */
TEST_CASE("Test simulations with skew cells", "[runSimulationAdaptiveLagrangian()]")
{
    // Define common simulation parameters
    const T R = 0.8;
    const T half_l = 0.5;
    const T L0 = 2 * half_l;
    const T Rcell = 0.5;
    const T E0 = 3900.0;
    const T Ecell = 3900000.0;
    const T eta_ambient = 0.072;
    const T eta_surface = 720.0;
    const T sigma0 = 100.0;
    const T Ldiv = 2 * (R + L0);
    const T neighbor_threshold = 2 * (2 * R + L0);
    const int max_iter = 1000000;
    const T max_stepsize = 1e-6;
    const T min_stepsize = 1e-8;
    const bool write = false;
    const std::string outprefix = "";
    const T dt_write = 0.01;
    const int iter_update_neighbors = 100;
    const int iter_update_stepsize = 1;
    const T max_error_allowed = 1e-10;
    const T min_error = 1e-30;
    const T max_tries_update_stepsize = 3;
    const T nz_threshold = 1e-20;
    const int rng_seed = 42;
    const T growth_mean = 0.0;
    const T growth_std = 0.0;
    const int n_groups = 1; 
    std::vector<int> group_attributes; 
    Array<T, Dynamic, 1> growth_means(n_groups), growth_stds(n_groups); 
    growth_means << growth_mean; 
    growth_stds << growth_std;
    Array<T, Dynamic, Dynamic> attribute_means(n_groups, 0), attribute_stds(n_groups, 0);
    Array<T, Dynamic, Dynamic> switch_rates = Array<T, Dynamic, Dynamic>::Zero(1, 1); 
    const T daughter_length_std = 0.0; 
    const T daughter_angle_xy_bound = 0.0;
    const T daughter_angle_z_bound = 0.0;
    const bool truncate_surface_friction = false; 
    const T surface_coulomb_coeff = 0.0;
    const T max_noise = 0.0;
    const bool basal_only = false; 
    const T basal_min_overlap = 0.0; 
    std::unordered_set<std::pair<int, int>, boost::hash<std::pair<int, int> > > adhesion_map; 
    std::unordered_map<std::string, T> adhesion_params;  

    // For a selection of angles ... 
    Array<T, 10, 1> angles = Array<T, 10, 1>::Zero();
    for (int i = 0; i < angles.size(); ++i)
        angles(i) = boost::math::constants::half_pi<T>() * (i + 1) / 10.0;
    for (int i = 0; i < angles.size(); ++i)
    { 
        // Case 1: A cell making the given angle with the surface, positioned 
        // such that the maximum overlap (at s = -half_l) is 0.001 * R
        Array<T, 3, 1> r, n;
        T max_overlap = 0.001 * R;
        T rz = R + half_l * sin(angles(i)) - max_overlap; 
        r << 0.0, 0.0, rz;
        n << cos(angles(i)), 0.0, sin(angles(i));
        Array<T, Dynamic, Dynamic> cells(1, __ncols_required);
        cells << 0, r(0), r(1), r(2), n(0), n(1), n(2), 0, 0, 0, 0, 0, 0,
                 2 * half_l, half_l, 0.0, growth_mean, eta_ambient, eta_surface,
                 eta_surface, sigma0, 1;
        std::vector<int> parents; 

        // Run simulation
        auto result = runSimulationAdaptiveLagrangian<T>(
            cells, parents, max_iter, -1, R, Rcell, L0, Ldiv, E0, Ecell,
            max_stepsize, min_stepsize, write, outprefix, dt_write,
            iter_update_neighbors, iter_update_stepsize, max_error_allowed,
            min_error, max_tries_update_stepsize, neighbor_threshold, nz_threshold,
            rng_seed, n_groups, group_attributes, growth_means, growth_stds,
            attribute_means, attribute_stds, SwitchMode::NONE, switch_rates,
            daughter_length_std, daughter_angle_xy_bound, daughter_angle_z_bound,
            truncate_surface_friction, surface_coulomb_coeff, max_noise,
            basal_only, basal_min_overlap, AdhesionMode::NONE, adhesion_map, 
            adhesion_params
        );
        cells = result.first;

        // Test that the cell has not grown and every attribute other than
        // its x-, y-, z-coordinates and its x- and z-orientations remains as
        // it was 
        const T delta = 1e-8;  
        REQUIRE(static_cast<int>(cells(0, __colidx_id)) == 0); 
        REQUIRE_THAT(cells(0, __colidx_ny), Catch::Matchers::WithinAbs(0, delta)); 
        REQUIRE_THAT(cells(0, __colidx_l), Catch::Matchers::WithinAbs(2 * half_l, delta)); 
        REQUIRE_THAT(cells(0, __colidx_half_l), Catch::Matchers::WithinAbs(half_l, delta)); 
        REQUIRE_THAT(cells(0, __colidx_t0), Catch::Matchers::WithinAbs(0, delta));
        REQUIRE_THAT(cells(0, __colidx_growth), Catch::Matchers::WithinAbs(0, delta)); 
        REQUIRE_THAT(cells(0, __colidx_eta0), Catch::Matchers::WithinAbs(eta_ambient, delta)); 
        REQUIRE_THAT(cells(0, __colidx_eta1), Catch::Matchers::WithinAbs(eta_surface, delta));
        REQUIRE_THAT(cells(0, __colidx_maxeta1), Catch::Matchers::WithinAbs(eta_surface, delta));
        REQUIRE_THAT(cells(0, __colidx_sigma0), Catch::Matchers::WithinAbs(sigma0, delta));
        REQUIRE(static_cast<int>(cells(0, __colidx_group)) == 1);

        // Test that the cell has reached a steady-state position
        REQUIRE_THAT(cells(0, __colidx_drx), Catch::Matchers::WithinAbs(0, delta)); 
        REQUIRE_THAT(cells(0, __colidx_dry), Catch::Matchers::WithinAbs(0, delta));
        REQUIRE_THAT(cells(0, __colidx_drz), Catch::Matchers::WithinAbs(0, delta)); 
        REQUIRE_THAT(cells(0, __colidx_dnx), Catch::Matchers::WithinAbs(0, delta));
        REQUIRE_THAT(cells(0, __colidx_dny), Catch::Matchers::WithinAbs(0, delta)); 
        REQUIRE_THAT(cells(0, __colidx_dnz), Catch::Matchers::WithinAbs(0, delta));

        // Test that the cell is vertical or horizontal, depending on the 
        // energetics of the initial configuration
        if (moreAdhesiveThanRepulsive(rz, n(2), half_l, R, E0, sigma0, nz_threshold))
        {
            REQUIRE_THAT(cells(0, __colidx_nx), Catch::Matchers::WithinAbs(1.0, delta)); 
            REQUIRE_THAT(cells(0, __colidx_ny), Catch::Matchers::WithinAbs(0.0, delta)); 
            REQUIRE_THAT(cells(0, __colidx_nz), Catch::Matchers::WithinAbs(0.0, delta));
            
            // Test that the cell's z-coordinate matches the theoretical value 
            T adhesion = sigma0 / (R * E0);
            T target = R * (1 - pow(0.25 * adhesion, 2./3.));
            REQUIRE_THAT(cells(0, __colidx_rz), Catch::Matchers::WithinAbs(target, delta));
        } 
        else 
        {
            REQUIRE_THAT(cells(0, __colidx_nx), Catch::Matchers::WithinAbs(0.0, delta)); 
            REQUIRE_THAT(cells(0, __colidx_ny), Catch::Matchers::WithinAbs(0.0, delta)); 
            REQUIRE_THAT(cells(0, __colidx_nz), Catch::Matchers::WithinAbs(1.0, delta));

            // Test that the cell's z-coordinate matches the theoretical value 
            T adhesion = sigma0 / (R * E0);
            T target = half_l + R * (1 - pow(0.75 * boost::math::constants::pi<T>() * adhesion, 2./3.));
            REQUIRE_THAT(cells(0, __colidx_rz), Catch::Matchers::WithinAbs(target, delta));
        }

        // Case 2: A cell making the given angle with the surface, positioned 
        // such that the maximum overlap (at s = -half_l) is larger, at 0.2 * R
        max_overlap = 0.2 * R;
        rz = R + half_l * sin(angles(i)) - max_overlap; 
        r(2) = rz;
        cells << 0, r(0), r(1), r(2), n(0), n(1), n(2), 0, 0, 0, 0, 0, 0,
                 2 * half_l, half_l, 0.0, growth_mean, eta_ambient, eta_surface,
                 eta_surface, sigma0, 1;

        // Run simulation
        result = runSimulationAdaptiveLagrangian<T>(
            cells, parents, max_iter, -1, R, Rcell, L0, Ldiv, E0, Ecell,
            max_stepsize, min_stepsize, write, outprefix, dt_write,
            iter_update_neighbors, iter_update_stepsize, max_error_allowed,
            min_error, max_tries_update_stepsize, neighbor_threshold, nz_threshold,
            rng_seed, n_groups, group_attributes, growth_means, growth_stds,
            attribute_means, attribute_stds, SwitchMode::NONE, switch_rates,
            daughter_length_std, daughter_angle_xy_bound, daughter_angle_z_bound,
            truncate_surface_friction, surface_coulomb_coeff, max_noise,
            basal_only, basal_min_overlap, AdhesionMode::NONE, adhesion_map, 
            adhesion_params
        );
        cells = result.first;  

        // Test that the cell has not grown and every attribute other than its 
        // z-coordinate has remained as it was 
        REQUIRE(static_cast<int>(cells(0, __colidx_id)) == 0); 
        REQUIRE_THAT(cells(0, __colidx_ny), Catch::Matchers::WithinAbs(0, delta)); 
        REQUIRE_THAT(cells(0, __colidx_l), Catch::Matchers::WithinAbs(2 * half_l, delta)); 
        REQUIRE_THAT(cells(0, __colidx_half_l), Catch::Matchers::WithinAbs(half_l, delta)); 
        REQUIRE_THAT(cells(0, __colidx_t0), Catch::Matchers::WithinAbs(0, delta));
        REQUIRE_THAT(cells(0, __colidx_growth), Catch::Matchers::WithinAbs(0, delta)); 
        REQUIRE_THAT(cells(0, __colidx_eta0), Catch::Matchers::WithinAbs(eta_ambient, delta)); 
        REQUIRE_THAT(cells(0, __colidx_eta1), Catch::Matchers::WithinAbs(eta_surface, delta));
        REQUIRE_THAT(cells(0, __colidx_maxeta1), Catch::Matchers::WithinAbs(eta_surface, delta));
        REQUIRE_THAT(cells(0, __colidx_sigma0), Catch::Matchers::WithinAbs(sigma0, delta));
        REQUIRE(static_cast<int>(cells(0, __colidx_group)) == 1);

        // Test that the cell has reached a steady-state position 
        REQUIRE_THAT(cells(0, __colidx_drx), Catch::Matchers::WithinAbs(0, delta)); 
        REQUIRE_THAT(cells(0, __colidx_dry), Catch::Matchers::WithinAbs(0, delta));
        REQUIRE_THAT(cells(0, __colidx_drz), Catch::Matchers::WithinAbs(0, delta)); 
        REQUIRE_THAT(cells(0, __colidx_dnx), Catch::Matchers::WithinAbs(0, delta));
        REQUIRE_THAT(cells(0, __colidx_dny), Catch::Matchers::WithinAbs(0, delta)); 
        REQUIRE_THAT(cells(0, __colidx_dnz), Catch::Matchers::WithinAbs(0, delta));

        // Test that the cell is vertical or horizontal, depending on the 
        // energetics of the initial configuration
        if (moreAdhesiveThanRepulsive(rz, n(2), half_l, R, E0, sigma0, nz_threshold))
        {
            REQUIRE_THAT(cells(0, __colidx_nx), Catch::Matchers::WithinAbs(1.0, delta)); 
            REQUIRE_THAT(cells(0, __colidx_ny), Catch::Matchers::WithinAbs(0.0, delta)); 
            REQUIRE_THAT(cells(0, __colidx_nz), Catch::Matchers::WithinAbs(0.0, delta));
            
            // Test that the cell's z-coordinate matches the theoretical value 
            T adhesion = sigma0 / (R * E0);
            T target = R * (1 - pow(0.25 * adhesion, 2./3.));
            REQUIRE_THAT(cells(0, __colidx_rz), Catch::Matchers::WithinAbs(target, delta));
        } 
        else 
        {
            REQUIRE_THAT(cells(0, __colidx_nx), Catch::Matchers::WithinAbs(0.0, delta)); 
            REQUIRE_THAT(cells(0, __colidx_ny), Catch::Matchers::WithinAbs(0.0, delta)); 
            REQUIRE_THAT(cells(0, __colidx_nz), Catch::Matchers::WithinAbs(1.0, delta));

            // Test that the cell's z-coordinate matches the theoretical value 
            T adhesion = sigma0 / (R * E0);
            T target = half_l + R * (1 - pow(0.75 * boost::math::constants::pi<T>() * adhesion, 2./3.));
            REQUIRE_THAT(cells(0, __colidx_rz), Catch::Matchers::WithinAbs(target, delta));
        }

        // Case 3: A cell making the given angle with the surface that does 
        // not contact the surface 
        max_overlap = -0.01 * R;
        rz = R + half_l * sin(angles(i)) - max_overlap; 
        r(2) = rz;
        cells << 0, r(0), r(1), r(2), n(0), n(1), n(2), 0, 0, 0, 0, 0, 0,
                 2 * half_l, half_l, 0.0, growth_mean, eta_ambient, eta_surface,
                 eta_surface, sigma0, 1;

        // Run simulation
        result = runSimulationAdaptiveLagrangian<T>(
            cells, parents, max_iter, -1, R, Rcell, L0, Ldiv, E0, Ecell,
            max_stepsize, min_stepsize, write, outprefix, dt_write,
            iter_update_neighbors, iter_update_stepsize, max_error_allowed,
            min_error, max_tries_update_stepsize, neighbor_threshold, nz_threshold,
            rng_seed, n_groups, group_attributes, growth_means, growth_stds,
            attribute_means, attribute_stds, SwitchMode::NONE, switch_rates,
            daughter_length_std, daughter_angle_xy_bound, daughter_angle_z_bound,
            truncate_surface_friction, surface_coulomb_coeff, max_noise,
            basal_only, basal_min_overlap, AdhesionMode::NONE, adhesion_map, 
            adhesion_params
        );
        cells = result.first;  

        // Test that the cell has not grown and every attribute other than its 
        // z-coordinate has remained as it was 
        REQUIRE(static_cast<int>(cells(0, __colidx_id)) == 0); 
        REQUIRE_THAT(cells(0, __colidx_rx), Catch::Matchers::WithinAbs(0, delta));
        REQUIRE_THAT(cells(0, __colidx_ry), Catch::Matchers::WithinAbs(0, delta)); 
        REQUIRE_THAT(cells(0, __colidx_nx), Catch::Matchers::WithinAbs(n(0), delta));
        REQUIRE_THAT(cells(0, __colidx_ny), Catch::Matchers::WithinAbs(n(1), delta)); 
        REQUIRE_THAT(cells(0, __colidx_nz), Catch::Matchers::WithinAbs(n(2), delta));
        REQUIRE_THAT(cells(0, __colidx_l), Catch::Matchers::WithinAbs(2 * half_l, delta)); 
        REQUIRE_THAT(cells(0, __colidx_half_l), Catch::Matchers::WithinAbs(half_l, delta)); 
        REQUIRE_THAT(cells(0, __colidx_t0), Catch::Matchers::WithinAbs(0, delta));
        REQUIRE_THAT(cells(0, __colidx_growth), Catch::Matchers::WithinAbs(0, delta)); 
        REQUIRE_THAT(cells(0, __colidx_eta0), Catch::Matchers::WithinAbs(eta_ambient, delta)); 
        REQUIRE_THAT(cells(0, __colidx_eta1), Catch::Matchers::WithinAbs(eta_surface, delta));
        REQUIRE_THAT(cells(0, __colidx_maxeta1), Catch::Matchers::WithinAbs(eta_surface, delta));
        REQUIRE_THAT(cells(0, __colidx_sigma0), Catch::Matchers::WithinAbs(sigma0, delta));
        REQUIRE(static_cast<int>(cells(0, __colidx_group)) == 1);

        // Test that the cell has reached a steady-state position 
        REQUIRE_THAT(cells(0, __colidx_drx), Catch::Matchers::WithinAbs(0, delta)); 
        REQUIRE_THAT(cells(0, __colidx_dry), Catch::Matchers::WithinAbs(0, delta));
        REQUIRE_THAT(cells(0, __colidx_drz), Catch::Matchers::WithinAbs(0, delta)); 
        REQUIRE_THAT(cells(0, __colidx_dnx), Catch::Matchers::WithinAbs(0, delta));
        REQUIRE_THAT(cells(0, __colidx_dny), Catch::Matchers::WithinAbs(0, delta)); 
        REQUIRE_THAT(cells(0, __colidx_dnz), Catch::Matchers::WithinAbs(0, delta));

        // Test that the cell's z-coordinate matches the theoretical value, which 
        // should match the original value 
        REQUIRE_THAT(cells(0, __colidx_rz), Catch::Matchers::WithinAbs(rz, delta));
    } 
}
