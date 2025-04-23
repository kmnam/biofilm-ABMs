/**
 * Test simulations with two contacting or non-contacting cells in various
 * configurations. 
 * 
 * Authors:
 *     Kee-Myoung Nam
 *
 * Last updated:
 *     4/22/2025
 */
#include <iostream>
#include <cmath>
#include <functional>
#include <Eigen/Dense>
#include <boost/math/constants/constants.hpp>
#include <boost/multiprecision/mpfr.hpp>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Segment_3.h>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "../../include/indices.hpp"
#include "../../include/distances.hpp"
#include "../../include/simulation.hpp"
#include "../../include/switch.hpp"

using namespace Eigen;

typedef CGAL::Exact_predicates_inexact_constructions_kernel K; 
typedef K::Point_3 Point_3;
typedef K::Segment_3 Segment_3;
typedef double T; 

using std::pow; 
using boost::multiprecision::pow;

void run_two_cell_simulation(const Ref<const Array<T, 3, 1> >& r1, 
                             const Ref<const Array<T, 3, 1> >& n1, 
                             const Ref<const Array<T, 3, 1> >& r2, 
                             const Ref<const Array<T, 3, 1> >& n2, 
                             const T eqdist_expected)
{
    // Define simulation parameters
    const T R = 0.8;
    const T half_l = 0.5;
    const T L0 = 2 * half_l;
    const T Rcell = 0.5;
    const T E0 = 3900.0;
    const T Ecell = 3900000.0;
    const T eta_ambient = 1e-3;
    const T eta_surface = 720.0;
    const T sigma0 = 100.0;
    const T Ldiv = 2 * (R + L0);
    const T neighbor_threshold = 2 * (2 * R + L0);
    const int max_iter = 10000;
    const T max_stepsize = 1e-7;
    const T min_stepsize = 1e-9;
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
    const bool no_surface = true;  
    K kernel;

    // Define the second cell's coordinates, assuming cell lengths of 1
    Array<T, Dynamic, Dynamic> cells(2, __ncols_required);
    cells << 0, r1(0), r1(1), r1(2), n1(0), n1(1), n1(2), 0, 0, 0, 0, 0, 0,
             2 * half_l, half_l, 0.0, growth_mean, eta_ambient, eta_surface,
             eta_surface, sigma0, 1,
             1, r2(0), r2(1), r2(2), n2(0), n2(1), n2(2), 0, 0, 0, 0, 0, 0,
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
        truncate_surface_friction, surface_coulomb_coeff, max_noise, max_noise,
        max_noise, max_noise, basal_only, basal_min_overlap, AdhesionMode::NONE, 
        adhesion_map, adhesion_params, no_surface
    );
    cells = result.first;  

    // Test that the cell-cell distance at equilibrium is 2 * R
    Matrix<T, 3, 1> r1eq, n1eq, r2eq, n2eq; 
    r1eq << cells(0, __colidx_rx), cells(0, __colidx_ry), cells(0, __colidx_rz);
    n1eq << cells(0, __colidx_nx), cells(0, __colidx_ny), cells(0, __colidx_nz); 
    r2eq << cells(1, __colidx_rx), cells(1, __colidx_ry), cells(1, __colidx_rz);
    n2eq << cells(1, __colidx_nx), cells(1, __colidx_ny), cells(1, __colidx_nz); 
    Segment_3 seg1 = generateSegment<T>(r1eq, n1eq, half_l); 
    Segment_3 seg2 = generateSegment<T>(r2eq, n2eq, half_l);  
    auto distance = distBetweenCells<T>(
        seg1, seg2, 0, r1eq, n1eq, half_l, 1, r2eq, n2eq, half_l, kernel
    );
    T eqdist = std::get<0>(distance).norm();
    std::cout << "=> Equilibrium configuration: " 
              << "r1 = (" << r1eq(0) << ", " << r1eq(1) << ", " << r1eq(2) << "); "
              << "n1 = (" << n1eq(0) << ", " << n1eq(1) << ", " << n1eq(2) << ");\n"
              << "                              "
              << "r2 = (" << r2eq(0) << ", " << r2eq(1) << ", " << r2eq(2) << "); "
              << "n2 = (" << n2eq(0) << ", " << n2eq(1) << ", " << n2eq(2) << ")\n"; 
    std::cout << "=> Equilibrium distance = " << eqdist 
              << "; expected = " << eqdist_expected << std::endl; 
}

/* ------------------------------------------------------------------- //
 *                             TEST MODULES                            //
 * ------------------------------------------------------------------- */
TEST_CASE("Test simulations with parallel cells", "[runSimulationAdaptiveLagrangian()]")
{
    const T R = 0.8;
    const T half_l = 0.5;

    // Cases 1-3: Two parallel cells contacting in a head-to-head fashion
    // at various distances 
    //
    // Ensure that the two cells do not touch the surface
    Array<T, 3, 1> r1, n1, r2, n2;
    r1 << 0.0, 0.0, 1.1 * R;
    n1 << 1.0, 0.0, 0.0;
    Array<T, 3, 1> dists; 
    dists << 1.0, 1.4, 1.8; 
    for (int i = 0; i < 3; ++i)
    {
        std::cout << "Head-to-head configuration, initial distance = "
                  << dists(i) << std::endl; 
        r2 << 2 * half_l + dists(i), 0.0, 1.1 * R;
        n2 << 1.0, 0.0, 0.0;
        run_two_cell_simulation(r1, n1, r2, n2, (i == 0 || i == 1 ? 2 * R : dists(i)));
        std::cout << "-----------------------------------------------\n";  
    } 

    // Cases 4-6: Two parallel cells contacting in a side-by-side fashion 
    // at various distances 
    //
    // Ensure that the two cells do not touch the surface 
    for (int i = 0; i < 3; ++i)
    {
        std::cout << "Side-by-side configuration, initial distance = "
                  << dists(i) << std::endl; 
        r2 << 0.0, dists(i), 1.1 * R; 
        n2 << 1.0, 0.0, 0.0; 
        run_two_cell_simulation(r1, n1, r2, n2, (i == 0 || i == 1 ? 2 * R : dists(i)));
        std::cout << "-----------------------------------------------\n";  
    } 

    // Cases 7-9: Two parallel cells contacting in a side-by-side fashion 
    // at various distances, with an additional offset in the x-direction
    //
    // Ensure that the two cells do not touch the surface 
    for (int i = 0; i < 3; ++i)
    {
        std::cout << "Shifted side-by-side configuration, initial distance = "
                  << dists(i) << std::endl; 
        r2 << 0.2, dists(i), 1.1 * R; 
        n2 << 1.0, 0.0, 0.0; 
        run_two_cell_simulation(r1, n1, r2, n2, (i == 0 || i == 1 ? 2 * R : dists(i)));
        std::cout << "-----------------------------------------------\n";  
    }
}

TEST_CASE("Test simulations with perpendicular cells", "[runSimulationAdaptiveLagrangian()]")
{
    const T R = 0.8;
    const T half_l = 0.5;

    // Cases 1-3: Two perpendicular cells contacting in a "T" configuration
    // at various distances 
    //
    // Ensure that the two cells do not touch the surface
    Array<T, 3, 1> r1, n1, r2, n2;
    r1 << 0.0, 0.0, 1.1 * R;
    n1 << 1.0, 0.0, 0.0;
    Array<T, 3, 1> dists; 
    dists << 1.0, 1.4, 1.8; 
    for (int i = 0; i < 3; ++i)
    {
        std::cout << "Perpendicular T configuration, initial distance = "
                  << dists(i) << std::endl; 
        r2 << half_l + dists(i), 0.0, 1.1 * R; 
        n2 << 0.0, 1.0, 0.0;
        run_two_cell_simulation(r1, n1, r2, n2, (i == 0 || i == 1 ? 2 * R : dists(i)));
        std::cout << "-----------------------------------------------\n";  
    } 

    // Cases 4-6: Two perpendicular cells contacting in a slightly shifted
    // "T" configuration at various distances 
    //
    // Ensure that the two cells do not touch the surface 
    for (int i = 0; i < 3; ++i)
    {
        std::cout << "Shifted T configuration, initial distance = "
                  << dists(i) << std::endl; 
        r2 << half_l + dists(i), 0.2, 1.1 * R; 
        n2 << 0.0, 1.0, 0.0;
        run_two_cell_simulation(r1, n1, r2, n2, (i == 0 || i == 1 ? 2 * R : dists(i)));
        std::cout << "-----------------------------------------------\n";  
    } 

    // Cases 7-9: Two perpendicular cells in a cross configuration at 
    // various distances 
    //
    // Ensure that the two cells do not touch the surface 
    for (int i = 0; i < 3; ++i)
    {
        std::cout << "Cross configuration, initial distance = "
                  << dists(i) << std::endl; 
        r2 << 0.0, 0.0, 1.1 * R + dists(i); 
        n2 << 0.0, 1.0, 0.0; 
        run_two_cell_simulation(r1, n1, r2, n2, (i == 0 || i == 1 ? 2 * R : dists(i)));
        std::cout << "-----------------------------------------------\n";  
    }
}

TEST_CASE("Test simulations with skew cells", "[runSimulationAdaptiveLagrangian()]")
{
    const T R = 0.8;
    const T half_l = 0.5;

    // Cases 1-3: Two skew cells at a 30-degree angle within the xy-plane
    // in an end-to-end configuration, at various distances 
    //
    // Ensure that the two cells do not touch the surface
    Array<T, 3, 1> r1, n1, r2, n2;
    r1 << 0.0, 0.0, 1.1 * R;
    n1 << 1.0, 0.0, 0.0;
    Array<T, 3, 1> dists; 
    dists << 1.0, 1.4, 1.8; 
    for (int i = 0; i < 3; ++i)
    {
        std::cout << "Planar end-to-end 30-degree skew configuration, initial distance = "
                  << dists(i) << std::endl; 
        r2 << half_l + dists(i) + half_l * cos(boost::math::constants::sixth_pi<T>()),
              half_l * sin(boost::math::constants::sixth_pi<T>()),
              1.1 * R; 
        n2 << cos(boost::math::constants::sixth_pi<T>()),
              sin(boost::math::constants::sixth_pi<T>()),
              0.0;
        run_two_cell_simulation(r1, n1, r2, n2, (i == 0 || i == 1 ? 2 * R : dists(i)));
        std::cout << "-----------------------------------------------\n";  
    } 

    // Cases 4-6: Two skew cells at a 30-degree angle within the xy-plane 
    // in a side-by-side configuration, at various distances
    //
    // Ensure that the two cells do not touch the surface 
    for (int i = 0; i < 3; ++i)
    {
        std::cout << "Planar side-by-side 30-degree skew configuration, initial distance = "
                  << dists(i) << std::endl; 
        r2 << half_l * cos(boost::math::constants::sixth_pi<T>()), 
              dists(i) + half_l * sin(boost::math::constants::sixth_pi<T>()), 
              1.1 * R; 
        n2 << cos(boost::math::constants::sixth_pi<T>()),
              sin(boost::math::constants::sixth_pi<T>()),
              0.0;
        run_two_cell_simulation(r1, n1, r2, n2, (i == 0 || i == 1 ? 2 * R : dists(i)));
        std::cout << "-----------------------------------------------\n";  
    } 

    // Cases 7-9: Two skew cells at a 30-degree angle in a cross configuration,
    // at various distances 
    //
    // Ensure that the two cells do not touch the surface 
    for (int i = 0; i < 3; ++i)
    {
        std::cout << "30-degree skew cross configuration, initial distance = "
                  << dists(i) << std::endl; 
        r2 << 0.2, 0.0, 1.1 * R + dists(i); 
        n2 << cos(boost::math::constants::sixth_pi<T>()),
              sin(boost::math::constants::sixth_pi<T>()),
              0.0;
        run_two_cell_simulation(r1, n1, r2, n2, (i == 0 || i == 1 ? 2 * R : dists(i)));
        std::cout << "-----------------------------------------------\n";  
    }
}


