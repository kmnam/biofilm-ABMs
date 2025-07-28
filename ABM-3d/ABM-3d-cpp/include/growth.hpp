/**
 * Functions for modeling cell growth and division. 
 * 
 * In what follows, a population of N cells is represented as a 2-D array
 * with N rows, whose columns are as specified in `indices.hpp`.
 * 
 * Additional features may be included in the array but these are not relevant 
 * for the computations implemented here.
 *
 * Authors:
 *     Kee-Myoung Nam
 *
 * Last updated:
 *     7/28/2025
 */

#ifndef BIOFILM_CELL_GROWTH_3D_HPP
#define BIOFILM_CELL_GROWTH_3D_HPP

#include <cmath>
#include <limits>
#include <tuple>
#include <Eigen/Dense>
#include <boost/multiprecision/mpfr.hpp>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Segment_3.h>
#include "indices.hpp"
#include "distances.hpp"
#include "utils.hpp"

using namespace Eigen;

using std::pow; 
using boost::multiprecision::pow; 
using std::round; 
using boost::multiprecision::round; 

typedef CGAL::Exact_predicates_inexact_constructions_kernel K; 
typedef K::Segment_3 Segment_3;

/**
 * Grow the cells in the given population according to the exponential 
 * volume growth law.
 *
 * The given array of cell data is updated in place.  
 *
 * @param cells Existing population of cells. 
 * @param dt Timestep. 
 * @param R Cell radius.
 */
template <typename T>
void growCells(Ref<Array<T, Dynamic, Dynamic> > cells, const T dt, const T R)
{
    // Each cell grows in length according to an exponential growth law 
    cells.col(__colidx_l) += (cells.col(__colidx_growth) * (4 * R / 3 + cells.col(__colidx_l)) * dt).eval();
    cells.col(__colidx_half_l) = cells.col(__colidx_l) / 2; 
}

/**
 * Identify the cells that exceed the given division length. 
 *
 * @param cells Existing population of cells. 
 * @param Ldiv Cell division length.
 * @returns Boolean index indicating which cells are to divide. 
 */
template <typename T>
Array<int, Dynamic, 1> divideMaxLength(const Ref<const Array<T, Dynamic, Dynamic> >& cells,
                                       const T Ldiv)
{
    return (cells.col(__colidx_l) > Ldiv).template cast<int>(); 
}

/**
 * Get the minimum distance from the indicated cell to the others in 
 * the given population.
 *
 * @param cells Existing population of cells.
 * @param segments Vector of Segment_3 instances for the cells.
 * @param i Index of cell to which to calculate distances.
 * @returns Minimum distance of cell i to the other cells. 
 */
template <typename T>
T minDistToCell(const Ref<const Array<T, Dynamic, Dynamic> >& cells, 
                std::vector<Segment_3>& segments, const int i)
{
    // Instantiate kernel to be passed into distBetweenCells()
    K kernel;

    // Initializing the minimum distance to infinity, run through the 
    // cells in the population
    int n = cells.rows(); 
    T mindist = std::numeric_limits<T>::infinity();
    for (int j = 0; j < n; ++j)
    {
        if (i != j)
        {
            auto result = distBetweenCells<T>(
                segments[i], segments[j],
                static_cast<int>(cells(i, __colidx_id)), 
                cells(i, __colseq_r).matrix(), 
                cells(i, __colseq_n).matrix(),
                cells(i, __colidx_half_l),
                static_cast<int>(cells(j, __colidx_id)), 
                cells(j, __colseq_r).matrix(),
                cells(j, __colseq_n).matrix(),
                cells(j, __colidx_half_l),
                kernel
            );
            Matrix<T, 3, 1> dij = std::get<0>(result);
            T dist = dij.norm();
            if (dist < mindist)
                mindist = dist;
        }
    }

    return mindist;
}

/**
 * Divide the indicated cells at the given time.
 *
 * If i is the index of a dividing cell, row i is updated to store one 
 * daughter cell, and a new row is appended to store the other daughter
 * cell.
 *
 * A newly allocated array of cell data is returned. 
 *
 * The growth rate of each daughter cell is chosen using the given
 * distribution function, which must take a random number generator
 * instance as its single input. 
 *
 * The function daughter_length_dist() controls the degree of asymmetry
 * in cell division: the daughter cells are determined to have lengths
 * M * (L - 2 * R) and (1 - M) * (L - 2 * R), where L is the length of
 * the dividing cell, R is the cell radius, and M is sampled using
 * daughter_length_dist(). 
 *
 * The functions daughter_angle_xy_dist() and daughter_angle_z_dist()
 * control the orientations of the daughter cells: each daughter cell's
 * orientation is obtained by rotating the dividing cell's orientation
 * by two angles, one in the xy-plane and one out of the xy-plane, which
 * are sampled using daughter_angle_xy_dist() and daughter_angle_z_dist().
 *
 * @param cells Existing population of cells.
 * @param parents Vector of parent cell IDs for each cell generated throughout
 *                the simulation. 
 * @param t Current time.
 * @param R Cell radius (including the EPS).
 * @param Rcell Cell radius (excluding the EPS). 
 * @param to_divide Boolean index indicating which cells are to divide.
 * @param growth_dist Function instance specifying the growth rate
 *                    distribution. Must take boost::random::mt19937&
 *                    as its single argument.
 * @param rng Random number generator. 
 * @param daughter_length_dist Function instance specifying the daughter 
 *                             cell length ratio distribution. 
 * @param daughter_angle_xy_dist Function instance specifying the daughter 
 *                               cell re-orientation distribution in the
 *                               xy-plane (about z-axis).
 * @param daughter_angle_z_dist  Function instance specifying the daughter 
 *                               cell re-orientation distribution out of 
 *                               the xy-plane (in the z-direction).
 * @returns Updated population of cells, as well as a vector of pairs of 
 *          daughter cell indices.  
 */
template <typename T>
std::pair<Array<T, Dynamic, Dynamic>, std::vector<std::pair<int, int> > >
    divideCells(const Ref<const Array<T, Dynamic, Dynamic> >& cells,
                std::vector<int>& parents, const T t, const T R, const T Rcell,
                const Ref<const Array<int, Dynamic, 1> >& to_divide,
                std::function<T(boost::random::mt19937&)>& growth_dist,
                boost::random::mt19937& rng,
                std::function<T(boost::random::mt19937&)>& daughter_length_dist,
                std::function<T(boost::random::mt19937&)>& daughter_angle_xy_dist,
                std::function<T(boost::random::mt19937&)>& daughter_angle_z_dist)
{
    // If there are cells to be divided ...
    const int n_divide = to_divide.sum();
    std::vector<std::pair<int, int> > daughter_pairs; 
    if (n_divide > 0)
    {
        // Get indices of cells to be divided
        const int n = cells.rows(); 
        std::vector<int> idx_divide;
        for (int i = 0; i < n; ++i)
        {
            if (to_divide(i))
                idx_divide.push_back(i);
        }

        // Generate Segment_3 instances for the cells 
        std::vector<Segment_3> segments = generateSegments<T>(cells);

        // Get an extended copy of the existing population
        Array<T, Dynamic, Dynamic> cells_total(n + n_divide, cells.cols());

        // Get the minimum distance from each dividing cell among all other
        // cells in the population
        //
        // If the minimum distance is less than the default value, then the
        // daughter cells do not undergo random re-orientation 
        //
        // Otherwise, the daughter cells are randomly re-oriented and a
        // minimum distance criterion is checked for both daughter cells  
        Array<int, Dynamic, 1> check_distance = Array<int, Dynamic, 1>::Ones(n_divide); 
        int m = 0;
        const T mindist_default = 2 * Rcell;    // Default value
        for (const int i : idx_divide)
        {
            // Get the minimum distance to cell i
            T mindist = minDistToCell<T>(cells, segments, i);

            // If the minimum distance is less than the default value
            // (there is no room to rotate), then don't re-orient daughter
            // cells and forgo checking their minimum distances
            if (mindist < mindist_default)
                check_distance(m) = 0;
            m++;
        }

        // Initialize array of indicators for whether each dividing cell
        // satisfies the corresponding minimum distance criterion
        Array<int, Dynamic, 1> satisfies_distance = Array<int, Dynamic, 1>::Zero(n_divide);
        for (int i = 0; i < n_divide; ++i)
        {
            if (check_distance(i) == 0)
                satisfies_distance(i) = 1;
        }

        // Try to satisfy the distance criterion for all cells up to a fixed 
        // number of iterations
        //
        // The last iteration is attempted with no re-orientation of the 
        // daughter cells, in which case the division should give rise to no
        // change in the minimum distance from the dividing cell to the
        // daughter cells; therefore, the distance criterion should be
        // satisfied on the last iteration
        //
        // If not (due to rare errors in distance calculation), then an 
        // exception is raised
        int ntries = 0;
        const int ntries_total = 10; 
        while (ntries == 0 || (ntries < ntries_total && satisfies_distance.sum() < n_divide))
        {
            // Get a copy of the corresponding sub-array
            cells_total(Eigen::seqN(0, n), Eigen::all) = cells;
            Array<T, Dynamic, Dynamic> new_cells(cells(idx_divide, Eigen::all)); 

            // Update cell orientations ...
            Array<T, Dynamic, 3> dividing_orientations(new_cells(Eigen::all, __colseq_n)); 
            Array<T, Dynamic, 1> theta_xy1 = Array<T, Dynamic, 1>::Zero(n_divide);
            Array<T, Dynamic, 1> theta_z1 = Array<T, Dynamic, 1>::Zero(n_divide);
            Array<T, Dynamic, 1> theta_xy2 = Array<T, Dynamic, 1>::Zero(n_divide); 
            Array<T, Dynamic, 1> theta_z2 = Array<T, Dynamic, 1>::Zero(n_divide);
            for (int i = 0; i < n_divide; ++i)
            {
                // Sample angles by which to rotate the daughter cells
                //
                // If there is no room to rotate the daughter cells or the
                // maximum number of tries have been attempted, then set 
                // both angles to zero
                if (ntries < ntries_total - 1 && check_distance(i) == 1)
                {
                    theta_xy1(i) = daughter_angle_xy_dist(rng);
                    theta_z1(i) = daughter_angle_z_dist(rng); 
                    theta_xy2(i) = daughter_angle_xy_dist(rng); 
                    theta_z2(i) = daughter_angle_z_dist(rng); 
                }
            }
            for (int i = 0; i < n_divide; ++i)
            {
                Array<T, 3, 1> u = dividing_orientations.row(i).transpose();
                Array<T, 3, 1> v1 = rotateOutOfXY<T>(rotateXY<T>(u, theta_xy1(i)), theta_z1(i));
                cells_total(idx_divide[i], __colseq_n) = v1;
                Array<T, 3, 1> v2 = rotateOutOfXY<T>(rotateXY<T>(u, theta_xy2(i)), theta_z2(i));
                new_cells(i, __colseq_n) = v2;
            }

            // Update cell lengths and positions ... 
            //
            // If cell division is symmetric, then:
            // - the daughter cells both have length L / 2 - R, where L is the
            //   length of the dividing cell; 
            // - the point of division occurs at the center of the dividing cell;
            // - their x-, y-, and z-coordinates are perturbed by L / 4 + R / 2 
            //   along the daughter cells' orientation vectors 
            //
            // If not, then:
            // - the daughter cells have lengths L1 = M * (L - 2 * R) and
            //   L2 = (1 - M) * (L - 2 * R), where M is a random variable
            //   between 0 and 1;
            // - the point of division occurs at cell body coordinate
            //   L1 + R - Lh with Lh = L / 2, which has coordinates given by:
            //   - rx + (L1 + R - Lh) * nx
            //   - ry + (L1 + R - Lh) * ny,
            //   - rz + (L1 + R - Lh) * nz,
            //   where (rx, ry, rz) and (nx, ny, nz) are the position and
            //   orientation of the dividing cell;
            // - the corresponding perturbations of the cell centers from the
            //   point of division are given by R + L1 / 2 and R + L2 / 2
            //
            // Sample a normally distributed daughter cell length for each cell
            // to be divided 
            Array<T, Dynamic, 1> M(n_divide); 
            for (int i = 0; i < n_divide; ++i)
                M(i) = daughter_length_dist(rng);
            // Locate point of division along dividing cell centerline
            Array<T, Dynamic, 1> Ld = cells_total(idx_divide, __colidx_l) - 2 * R; 
            Array<T, Dynamic, 1> L1 = M * Ld;
            Array<T, Dynamic, 1> L2 = (1 - M) * Ld; 
            Array<T, Dynamic, 1> div = -cells_total(idx_divide, __colidx_half_l) + L1 + R;
            // Get perturbations from point of division along cell centerline
            // for the daughter cell centers
            Array<T, Dynamic, 1> delta1 = L1 / 2 + R; 
            Array<T, Dynamic, 1> delta2 = L2 / 2 + R;
            // Define daughter cell lengths
            cells_total(idx_divide, __colidx_l) = L1; 
            cells_total(idx_divide, __colidx_half_l) = L1 / 2;
            new_cells.col(__colidx_l) = L2;
            new_cells.col(__colidx_half_l) = L2 / 2;
            // Locate daughter cell centers
            cells_total(idx_divide, __colidx_rx) = (
                cells_total(idx_divide, __colidx_rx) + (div - delta1) * cells_total(idx_divide, __colidx_nx)
            );
            new_cells.col(__colidx_rx) = (
                new_cells.col(__colidx_rx) + (div + delta2) * new_cells.col(__colidx_nx)
            );
            cells_total(idx_divide, __colidx_ry) = (
                cells_total(idx_divide, __colidx_ry) + (div - delta1) * cells_total(idx_divide, __colidx_ny)
            );
            new_cells.col(__colidx_ry) = (
                new_cells.col(__colidx_ry) + (div + delta2) * new_cells.col(__colidx_ny)
            );
            cells_total(idx_divide, __colidx_rz) = (
                cells_total(idx_divide, __colidx_rz) + (div - delta1) * cells_total(idx_divide, __colidx_nz)
            );
            new_cells.col(__colidx_rz) = (
                new_cells.col(__colidx_rz) + (div + delta2) * new_cells.col(__colidx_nz)
            );

            // Update cell birth times
            cells_total(idx_divide, __colidx_t0) = t;
            new_cells.col(__colidx_t0) = t;

            // Update cell growth rates (sample from specified distribution)
            for (auto it = idx_divide.begin(); it != idx_divide.end(); ++it)
                cells_total(*it, __colidx_growth) = growth_dist(rng); 
            for (int i = 0; i < n_divide; ++i)
                new_cells(i, __colidx_growth) = growth_dist(rng);

            // Copy over daughter cell data
            //
            // Note that each daughter cell inherits its mother cell's viscosity,
            // friction coefficient, and surface adhesion energy density
            cells_total(Eigen::seqN(n, n_divide), Eigen::all) = new_cells;
            for (int i = 0; i < n_divide; ++i)
                daughter_pairs.emplace_back(std::make_pair(idx_divide[i], n + i));  

            // Generate Segment_3 instances for the new population of cells
            std::vector<Segment_3> segments_total = generateSegments<T>(cells_total);

            // Check the distance criterion for each daughter cell ...
            //
            // First compute the minimum distance to each daughter cell
            // in the new population
            Array<T, Dynamic, 1> daughter_mindists1(n_divide);
            Array<T, Dynamic, 1> daughter_mindists2(n_divide); 
            m = 0;
            for (const int i : idx_divide)
            {
                if (check_distance(m) == 1)
                {
                    T mindist = minDistToCell<T>(cells_total, segments_total, i);
                    daughter_mindists1(m) = mindist;
                }
                m++;
            }
            m = 0;
            for (int i = n; i < n + n_divide; ++i)
            {
                if (check_distance(m) == 1)
                {
                    T mindist = minDistToCell<T>(cells_total, segments_total, i);
                    daughter_mindists2(m) = mindist;
                }
                m++;
            }
            // Then determine whether the minimum distance to each pair
            // of daughter cells is within the desired tolerance 
            for (int i = 0; i < n_divide; ++i)
            {
                if (check_distance(i) == 1 &&
                    daughter_mindists1(i) >= mindist_default &&
                    daughter_mindists2(i) >= mindist_default)
                {
                    satisfies_distance(i) = 1;
                }
            }
            ntries++; 
        }

        // If the minimum distance criterion has not been satisfied for every
        // cell, then print a warning
        if (satisfies_distance.sum() < n_divide)
            std::cout << "[WARN] Cell division cannot satisfy minimum distance criterion";

        // Update cell ID and lineages 
        int max_id = parents.size() - 1; 
        for (int i = 0; i < n_divide; ++i)
        {
            int daughter_id1 = max_id + (2 * i + 1); 
            int daughter_id2 = max_id + (2 * i + 2);
            int parent_id = cells_total(idx_divide[i], __colidx_id); 
            cells_total(idx_divide[i], __colidx_id) = daughter_id1; 
            cells_total(n + i, __colidx_id) = daughter_id2; 
            parents.push_back(parent_id); 
            parents.push_back(parent_id); 
        }

        return std::make_pair(cells_total, daughter_pairs);  
    }
    else 
    {
        return std::make_pair(cells, daughter_pairs); 
    }
} 

/**
 * Divide the indicated cells at the given time.
 *
 * This is an extended version of `divideCells()` that assigns group-specific
 * growth rates. The groups are assumed to be numbered 1, 2, 3, ...
 *
 * @param cells Existing population of cells.
 * @param parents Vector of parent cell IDs for each cell generated throughout
 *                the simulation. 
 * @param t Current time.
 * @param R Cell radius (including the EPS).
 * @param Rcell Cell radius (excluding the EPS). 
 * @param to_divide Boolean index indicating which cells are to divide.
 * @param growth_dists Vector of function instances specifying the growth
 *                     rate distribution for each group. Each function must
 *                     take boost::random::mt19937& as its single argument.
 * @param rng Random number generator.
 * @param daughter_length_dist Function instance specifying the daughter 
 *                             cell length ratio distribution. 
 * @param daughter_angle_xy_dist Function instance specifying the daughter 
 *                               cell re-orientation distribution in the
 *                               xy-plane (about z-axis).
 * @param daughter_angle_z_dist  Function instance specifying the daughter 
 *                               cell re-orientation distribution out of 
 *                               the xy-plane (in the z-direction).
 * @returns Updated population of cells, as well as a vector of pairs of 
 *          daughter cell indices. 
 */
template <typename T>
std::pair<Array<T, Dynamic, Dynamic>, std::vector<std::pair<int, int> > >
    divideCells(const Ref<const Array<T, Dynamic, Dynamic> >& cells,
                std::vector<int>& parents, const T t, const T R, const T Rcell,
                const Ref<const Array<int, Dynamic, 1> >& to_divide,
                std::vector<std::function<T(boost::random::mt19937&)> >& growth_dists,
                boost::random::mt19937& rng,
                std::function<T(boost::random::mt19937&)>& daughter_length_dist,
                std::function<T(boost::random::mt19937&)>& daughter_angle_xy_dist,
                std::function<T(boost::random::mt19937&)>& daughter_angle_z_dist)
{
    // If there are cells to be divided ...
    const int n_divide = to_divide.sum();
    std::vector<std::pair<int, int> > daughter_pairs; 
    if (n_divide > 0)
    {
        // Get indices of cells to be divided
        const int n = cells.rows(); 
        std::vector<int> idx_divide;
        for (int i = 0; i < n; ++i)
        {
            if (to_divide(i))
                idx_divide.push_back(i);
        }

        // Generate Segment_3 instances for the cells 
        std::vector<Segment_3> segments = generateSegments<T>(cells);

        // Get an extended copy of the existing population
        Array<T, Dynamic, Dynamic> cells_total(n + n_divide, cells.cols());

        // Get the minimum distance from each dividing cell among all other
        // cells in the population
        //
        // If the minimum distance is less than the default value, then the
        // daughter cells do not undergo random re-orientation 
        //
        // Otherwise, the daughter cells are randomly re-oriented and a
        // minimum distance criterion is checked for both daughter cells  
        Array<int, Dynamic, 1> check_distance = Array<int, Dynamic, 1>::Ones(n_divide); 
        int m = 0;
        const T mindist_default = 2 * Rcell;    // Default value
        for (const int i : idx_divide)
        {
            // Get the minimum distance to cell i
            T mindist = minDistToCell<T>(cells, segments, i);

            // If the minimum distance is less than the default value
            // (there is no room to rotate), then don't re-orient daughter
            // cells and forgo checking their minimum distances
            if (mindist < mindist_default)
                check_distance(m) = 0;
            m++;
        }

        // Initialize array of indicators for whether each dividing cell satisfies
        // the corresponding minimum distance criterion
        Array<int, Dynamic, 1> satisfies_distance = Array<int, Dynamic, 1>::Zero(n_divide);
        for (int i = 0; i < n_divide; ++i)
        {
            if (check_distance(i) == 0)
                satisfies_distance(i) = 1;
        }

        // Try to satisfy the distance criterion for all cells up to a fixed 
        // number of iterations
        //
        // The last iteration is attempted with no re-orientation of the 
        // daughter cells, in which case the division should give rise to no
        // change in the minimum distance from the dividing cell to the
        // daughter cells; therefore, the distance criterion should be
        // satisfied on the last iteration
        //
        // If not (due to rare errors in distance calculation), then an 
        // exception is raised
        int ntries = 0;
        const int ntries_total = 10; 
        while (ntries == 0 || (ntries < ntries_total && satisfies_distance.sum() < n_divide))
        {
            // Get a copy of the corresponding sub-array
            cells_total(Eigen::seqN(0, n), Eigen::all) = cells; 
            Array<T, Dynamic, Dynamic> new_cells(cells(idx_divide, Eigen::all)); 

            // Update cell orientations ...
            Array<T, Dynamic, 3> dividing_orientations(new_cells(Eigen::all, __colseq_n)); 
            Array<T, Dynamic, 1> theta_xy1 = Array<T, Dynamic, 1>::Zero(n_divide);
            Array<T, Dynamic, 1> theta_z1 = Array<T, Dynamic, 1>::Zero(n_divide);
            Array<T, Dynamic, 1> theta_xy2 = Array<T, Dynamic, 1>::Zero(n_divide); 
            Array<T, Dynamic, 1> theta_z2 = Array<T, Dynamic, 1>::Zero(n_divide);
            for (int i = 0; i < n_divide; ++i)
            {
                // Sample angles by which to rotate the daughter cells
                //
                // If there is no room to rotate the daughter cells or the
                // maximum number of tries have been attempted, then set 
                // both angles to zero
                if (ntries < ntries_total - 1 && check_distance(i) == 1)
                {
                    theta_xy1(i) = daughter_angle_xy_dist(rng);
                    theta_z1(i) = daughter_angle_z_dist(rng); 
                    theta_xy2(i) = daughter_angle_xy_dist(rng); 
                    theta_z2(i) = daughter_angle_z_dist(rng); 
                }
            }
            for (int i = 0; i < n_divide; ++i)
            {
                Array<T, 3, 1> u = dividing_orientations.row(i).transpose();
                Array<T, 3, 1> v1 = rotateOutOfXY<T>(rotateXY<T>(u, theta_xy1(i)), theta_z1(i));
                cells_total(idx_divide[i], __colseq_n) = v1;
                Array<T, 3, 1> v2 = rotateOutOfXY<T>(rotateXY<T>(u, theta_xy2(i)), theta_z2(i));
                new_cells(i, __colseq_n) = v2;
            }

            // Update cell lengths and positions ... 
            //
            // If cell division is symmetric, then:
            // - the daughter cells both have length L / 2 - R, where L is the
            //   length of the dividing cell; 
            // - the point of division occurs at the center of the dividing cell;
            // - their x-, y-, and z-coordinates are perturbed by L / 4 + R / 2 
            //   along the daughter cells' orientation vectors 
            //
            // If not, then:
            // - the daughter cells have lengths L1 = M * (L - 2 * R) and
            //   L2 = (1 - M) * (L - 2 * R), where M is a random variable
            //   between 0 and 1;
            // - the point of division occurs at cell body coordinate
            //   L1 + R - Lh with Lh = L / 2, which has coordinates given by:
            //   - rx + (L1 + R - Lh) * nx
            //   - ry + (L1 + R - Lh) * ny,
            //   - rz + (L1 + R - Lh) * nz,
            //   where (rx, ry, rz) and (nx, ny, nz) are the position and
            //   orientation of the dividing cell;
            // - the corresponding perturbations of the cell centers from the
            //   point of division are given by R + L1 / 2 and R + L2 / 2
            //
            // Sample a normally distributed daughter cell length for each cell
            // to be divided 
            Array<T, Dynamic, 1> M(n_divide); 
            for (int i = 0; i < n_divide; ++i)
                M(i) = daughter_length_dist(rng);
            // Locate point of division along dividing cell centerline
            Array<T, Dynamic, 1> Ld = cells_total(idx_divide, __colidx_l) - 2 * R; 
            Array<T, Dynamic, 1> L1 = M * Ld;
            Array<T, Dynamic, 1> L2 = (1 - M) * Ld; 
            Array<T, Dynamic, 1> div = -cells_total(idx_divide, __colidx_half_l) + L1 + R;
            // Get perturbations from point of division along cell centerline
            // for the daughter cell centers
            Array<T, Dynamic, 1> delta1 = L1 / 2 + R; 
            Array<T, Dynamic, 1> delta2 = L2 / 2 + R;
            // Define daughter cell lengths
            cells_total(idx_divide, __colidx_l) = L1; 
            cells_total(idx_divide, __colidx_half_l) = L1 / 2;
            new_cells.col(__colidx_l) = L2;
            new_cells.col(__colidx_half_l) = L2 / 2;
            // Locate daughter cell centers
            cells_total(idx_divide, __colidx_rx) = (
                cells_total(idx_divide, __colidx_rx) + (div - delta1) * cells_total(idx_divide, __colidx_nx)
            );
            new_cells.col(__colidx_rx) = (
                new_cells.col(__colidx_rx) + (div + delta2) * new_cells.col(__colidx_nx)
            );
            cells_total(idx_divide, __colidx_ry) = (
                cells_total(idx_divide, __colidx_ry) + (div - delta1) * cells_total(idx_divide, __colidx_ny)
            );
            new_cells.col(__colidx_ry) = (
                new_cells.col(__colidx_ry) + (div + delta2) * new_cells.col(__colidx_ny)
            );
            cells_total(idx_divide, __colidx_rz) = (
                cells_total(idx_divide, __colidx_rz) + (div - delta1) * cells_total(idx_divide, __colidx_nz)
            );
            new_cells.col(__colidx_rz) = (
                new_cells.col(__colidx_rz) + (div + delta2) * new_cells.col(__colidx_nz)
            );

            // Update cell birth times
            cells_total(idx_divide, __colidx_t0) = t;
            new_cells.col(__colidx_t0) = t;

            // Update cell growth rates (sample from specified distributions)
            for (auto it = idx_divide.begin(); it != idx_divide.end(); ++it)
            {
                int group = static_cast<int>(cells_total(*it, __colidx_group)); 
                cells_total(*it, __colidx_growth) = growth_dists[group - 1](rng);
            } 
            for (int i = 0; i < n_divide; ++i)
            {
                int group = static_cast<int>(new_cells(i, __colidx_group));
                new_cells(i, __colidx_growth) = growth_dists[group - 1](rng);
            }

            // Copy over daughter cell data
            //
            // Note that each daughter cell inherits its mother cell's viscosity,
            // friction coefficient, and surface adhesion energy density
            cells_total(Eigen::seqN(n, n_divide), Eigen::all) = new_cells;
            for (int i = 0; i < n_divide; ++i)
                daughter_pairs.emplace_back(std::make_pair(idx_divide[i], n + i));  

            // Generate Segment_3 instances for the new population of cells
            std::vector<Segment_3> segments_total = generateSegments<T>(cells_total);

            // Check the distance criterion for each daughter cell ...
            //
            // First compute the minimum distance to each daughter cell
            // in the new population
            Array<T, Dynamic, 1> daughter_mindists1(n_divide);
            Array<T, Dynamic, 1> daughter_mindists2(n_divide); 
            m = 0;
            for (const int i : idx_divide)
            {
                if (check_distance(m) == 1)
                {
                    T mindist = minDistToCell<T>(cells_total, segments_total, i);
                    daughter_mindists1(m) = mindist;
                }
                m++;
            }
            m = 0;
            for (int i = n; i < n + n_divide; ++i)
            {
                if (check_distance(m) == 1)
                {
                    T mindist = minDistToCell<T>(cells_total, segments_total, i);
                    daughter_mindists2(m) = mindist;
                }
                m++;
            }
            // Then determine whether the minimum distance to each pair
            // of daughter cells is within the desired tolerance 
            for (int i = 0; i < n_divide; ++i)
            {
                if (check_distance(i) == 1 &&
                    daughter_mindists1(i) >= mindist_default &&
                    daughter_mindists2(i) >= mindist_default)
                {
                    satisfies_distance(i) = 1;
                }
            }
            ntries++; 
        }

        // If the minimum distance criterion has not been satisfied for every
        // cell, then print a warning
        if (satisfies_distance.sum() < n_divide)
            std::cout << "[WARN] Cell division cannot satisfy minimum distance criterion";

        // Update cell ID and lineages 
        int max_id = parents.size() - 1; 
        for (int i = 0; i < n_divide; ++i)
        {
            int daughter_id1 = max_id + (2 * i + 1); 
            int daughter_id2 = max_id + (2 * i + 2);
            int parent_id = cells_total(idx_divide[i], __colidx_id); 
            cells_total(idx_divide[i], __colidx_id) = daughter_id1; 
            cells_total(n + i, __colidx_id) = daughter_id2; 
            parents.push_back(parent_id); 
            parents.push_back(parent_id); 
        }

        return std::make_pair(cells_total, daughter_pairs); 
    }
    else 
    {
        return std::make_pair(cells, daughter_pairs); 
    }
}

/**
 * Divide the indicated cells at the given time.
 *
 * This is an extended version of `divideCells()` that:
 *
 * (1) assigns group-specific growth rates and
 * (2) keeps track of the old and new pole of each cell, as well as the 
 *     birth time of each pole. 
 *
 * The groups are assumed to be numbered 1, 2, 3, ..., and the pole birth 
 * times are stored in two additional columns.  
 *
 * @param cells Existing population of cells.
 * @param parents Vector of parent cell IDs for each cell generated throughout
 *                the simulation. 
 * @param t Current time.
 * @param R Cell radius (including the EPS).
 * @param Rcell Cell radius (excluding the EPS). 
 * @param to_divide Boolean index indicating which cells are to divide.
 * @param growth_dists Vector of function instances specifying the growth
 *                     rate distribution for each group. Each function must
 *                     take boost::random::mt19937& as its single argument.
 * @param rng Random number generator. 
 * @param daughter_length_dist Function instance specifying the daughter 
 *                             cell length ratio distribution. 
 * @param daughter_angle_xy_dist Function instance specifying the daughter 
 *                               cell re-orientation distribution in the
 *                               xy-plane (about z-axis).
 * @param daughter_angle_z_dist  Function instance specifying the daughter 
 *                               cell re-orientation distribution out of 
 *                               the xy-plane (in the z-direction).
 * @returns Updated population of cells, as well as a vector of pairs of 
 *          daughter cell indices. 
 */
template <typename T>
std::pair<Array<T, Dynamic, Dynamic>, std::vector<std::pair<int, int> > >
    divideCellsWithPoles(const Ref<const Array<T, Dynamic, Dynamic> >& cells,
                         std::vector<int>& parents, const T t, const T R,
                         const T Rcell, const Ref<const Array<int, Dynamic, 1> >& to_divide,
                         std::vector<std::function<T(boost::random::mt19937&)> >& growth_dists,
                         boost::random::mt19937& rng,
                         std::function<T(boost::random::mt19937&)>& daughter_length_dist,
                         std::function<T(boost::random::mt19937&)>& daughter_angle_xy_dist,
                         std::function<T(boost::random::mt19937&)>& daughter_angle_z_dist,
                         const int colidx_negpole_t0, const int colidx_pospole_t0)
{
    // If there are cells to be divided ...
    const int n_divide = to_divide.sum();
    std::vector<std::pair<int, int> > daughter_pairs; 
    if (n_divide > 0)
    {
        // Get indices of cells to be divided
        const int n = cells.rows(); 
        std::vector<int> idx_divide;
        for (int i = 0; i < n; ++i)
        {
            if (to_divide(i))
                idx_divide.push_back(i);
        }

        // Generate Segment_3 instances for the cells 
        std::vector<Segment_3> segments = generateSegments<T>(cells);
        
        // Get an extended copy of the existing population
        Array<T, Dynamic, Dynamic> cells_total(n + n_divide, cells.cols());

        // Get the minimum distance from each dividing cell among all other
        // cells in the population
        //
        // If the minimum distance is less than the default value, then the
        // daughter cells do not undergo random re-orientation 
        //
        // Otherwise, the daughter cells are randomly re-oriented and a
        // minimum distance criterion is checked for both daughter cells  
        Array<int, Dynamic, 1> check_distance = Array<int, Dynamic, 1>::Ones(n_divide); 
        int m = 0;
        const T mindist_default = 2 * Rcell;    // Default value
        for (const int i : idx_divide)
        {
            // Get the minimum distance to cell i
            T mindist = minDistToCell<T>(cells, segments, i);

            // If the minimum distance is less than the default value
            // (there is no room to rotate), then don't re-orient daughter
            // cells and forgo checking their minimum distances
            if (mindist < mindist_default)
                check_distance(m) = 0;
            m++;
        }

        // Initialize array of indicators for whether each dividing cell satisfies
        // the corresponding minimum distance criterion
        Array<int, Dynamic, 1> satisfies_distance = Array<int, Dynamic, 1>::Zero(n_divide);
        for (int i = 0; i < n_divide; ++i)
        {
            if (check_distance(i) == 0)
                satisfies_distance(i) = 1;
        }

        // Try to satisfy the distance criterion for all cells up to a fixed 
        // number of iterations
        //
        // The last iteration is attempted with no re-orientation of the 
        // daughter cells, in which case the division should give rise to no
        // change in the minimum distance from the dividing cell to the
        // daughter cells; therefore, the distance criterion should be
        // satisfied on the last iteration
        //
        // If not (due to rare errors in distance calculation), then an 
        // exception is raised
        int ntries = 0;
        const int ntries_total = 10; 
        while (ntries == 0 || (ntries < ntries_total && satisfies_distance.sum() < n_divide))
        {
            // Get a copy of the corresponding sub-array
            cells_total(Eigen::seqN(0, n), Eigen::all) = cells; 
            Array<T, Dynamic, Dynamic> new_cells(cells(idx_divide, Eigen::all)); 

            // Update cell orientations ...
            Array<T, Dynamic, 3> dividing_orientations(new_cells(Eigen::all, __colseq_n));
            Array<T, Dynamic, 1> theta_xy1 = Array<T, Dynamic, 1>::Zero(n_divide);
            Array<T, Dynamic, 1> theta_z1 = Array<T, Dynamic, 1>::Zero(n_divide);
            Array<T, Dynamic, 1> theta_xy2 = Array<T, Dynamic, 1>::Zero(n_divide); 
            Array<T, Dynamic, 1> theta_z2 = Array<T, Dynamic, 1>::Zero(n_divide);
            for (int i = 0; i < n_divide; ++i)
            {
                // Sample angles by which to rotate the daughter cells
                //
                // If there is no room to rotate the daughter cells or the
                // maximum number of tries have been attempted, then set 
                // both angles to zero
                if (ntries < ntries_total - 1 && check_distance(i) == 1)
                {
                    theta_xy1(i) = daughter_angle_xy_dist(rng);
                    theta_z1(i) = daughter_angle_z_dist(rng); 
                    theta_xy2(i) = daughter_angle_xy_dist(rng); 
                    theta_z2(i) = daughter_angle_z_dist(rng); 
                }
            }
            for (int i = 0; i < n_divide; ++i)
            {
                Array<T, 3, 1> u = dividing_orientations.row(i).transpose();
                Array<T, 3, 1> v1 = rotateOutOfXY<T>(rotateXY<T>(u, theta_xy1(i)), theta_z1(i));
                cells_total(idx_divide[i], __colseq_n) = v1;
                Array<T, 3, 1> v2 = rotateOutOfXY<T>(rotateXY<T>(u, theta_xy2(i)), theta_z2(i));
                new_cells(i, __colseq_n) = v2;
            }

            // Update cell lengths and positions ... 
            //
            // If cell division is symmetric, then:
            // - the daughter cells both have length L / 2 - R, where L is the
            //   length of the dividing cell; 
            // - the point of division occurs at the center of the dividing cell;
            // - their x-, y-, and z-coordinates are perturbed by L / 4 + R / 2 
            //   along the daughter cells' orientation vectors 
            //
            // If not, then:
            // - the daughter cells have lengths L1 = M * (L - 2 * R) and
            //   L2 = (1 - M) * (L - 2 * R), where M is a random variable
            //   between 0 and 1;
            // - the point of division occurs at cell body coordinate
            //   L1 + R - Lh with Lh = L / 2, which has coordinates given by:
            //   - rx + (L1 + R - Lh) * nx
            //   - ry + (L1 + R - Lh) * ny,
            //   - rz + (L1 + R - Lh) * nz,
            //   where (rx, ry, rz) and (nx, ny, nz) are the position and
            //   orientation of the dividing cell;
            // - the corresponding perturbations of the cell centers from the
            //   point of division are given by R + L1 / 2 and R + L2 / 2
            //
            // Sample a normally distributed daughter cell length for each cell
            // to be divided 
            Array<T, Dynamic, 1> M(n_divide); 
            for (int i = 0; i < n_divide; ++i)
                M(i) = daughter_length_dist(rng);
            // Locate point of division along dividing cell centerline
            Array<T, Dynamic, 1> Ld = cells_total(idx_divide, __colidx_l) - 2 * R; 
            Array<T, Dynamic, 1> L1 = M * Ld;
            Array<T, Dynamic, 1> L2 = (1 - M) * Ld; 
            Array<T, Dynamic, 1> div = -cells_total(idx_divide, __colidx_half_l) + L1 + R;
            // Get perturbations from point of division along cell centerline
            // for the daughter cell centers
            Array<T, Dynamic, 1> delta1 = L1 / 2 + R; 
            Array<T, Dynamic, 1> delta2 = L2 / 2 + R;
            // Define daughter cell lengths 
            cells_total(idx_divide, __colidx_l) = L1; 
            cells_total(idx_divide, __colidx_half_l) = L1 / 2;
            new_cells.col(__colidx_l) = L2;
            new_cells.col(__colidx_half_l) = L2 / 2;
            // Locate daughter cell centers
            cells_total(idx_divide, __colidx_rx) = (
                cells_total(idx_divide, __colidx_rx) + (div - delta1) * cells_total(idx_divide, __colidx_nx)
            );
            new_cells.col(__colidx_rx) = (
                new_cells.col(__colidx_rx) + (div + delta2) * new_cells.col(__colidx_nx)
            );
            cells_total(idx_divide, __colidx_ry) = (
                cells_total(idx_divide, __colidx_ry) + (div - delta1) * cells_total(idx_divide, __colidx_ny)
            );
            new_cells.col(__colidx_ry) = (
                new_cells.col(__colidx_ry) + (div + delta2) * new_cells.col(__colidx_ny)
            );
            cells_total(idx_divide, __colidx_rz) = (
                cells_total(idx_divide, __colidx_rz) + (div - delta1) * cells_total(idx_divide, __colidx_nz)
            );
            new_cells.col(__colidx_rz) = (
                new_cells.col(__colidx_rz) + (div + delta2) * new_cells.col(__colidx_nz)
            );

            // The first daughter cell (of length L1) inherits the *negative*
            // pole (cell body coordinate s = -L / 2) in the mother cell, and
            // gets a new *positive* pole at cell body coordinate s = L1 / 2
            //
            // The second daughter cell of length L2 inherits the *positive*
            // pole (cell body coordinate s = L / 2) in the mother cell, and
            // gets a new *negative* pole at cell body coordinate s = -L2 / 2
            std::vector<int> colseq_poles_t0 {colidx_negpole_t0, colidx_pospole_t0}; 
            Array<T, Dynamic, 2> poles_t0(cells_total(idx_divide, colseq_poles_t0)); 
            for (int i = 0; i < n_divide; ++i)
            {
                // First daughter cell inherits negative pole and gets new
                // positive pole 
                cells_total(idx_divide[i], colidx_negpole_t0) = poles_t0(i, 0);
                cells_total(idx_divide[i], colidx_pospole_t0) = t;

                // Second daughter cell inherits positive pole and gets new
                // negative pole 
                new_cells(i, colidx_negpole_t0) = t; 
                new_cells(i, colidx_pospole_t0) = poles_t0(i, 1);
            }

            // Update cell birth times
            cells_total(idx_divide, __colidx_t0) = t;
            new_cells.col(__colidx_t0) = t;

            // Update cell growth rates (sample from specified distributions)
            for (auto it = idx_divide.begin(); it != idx_divide.end(); ++it)
            {
                int group = static_cast<int>(cells_total(*it, __colidx_group)); 
                cells_total(*it, __colidx_growth) = growth_dists[group - 1](rng);
            } 
            for (int i = 0; i < n_divide; ++i)
            {
                int group = static_cast<int>(new_cells(i, __colidx_group));
                new_cells(i, __colidx_growth) = growth_dists[group - 1](rng);
            }

            // Copy over daughter cell data
            //
            // Note that each daughter cell inherits its mother cell's viscosity
            // and friction coefficient
            cells_total(Eigen::seqN(n, n_divide), Eigen::all) = new_cells;
            for (int i = 0; i < n_divide; ++i)
                daughter_pairs.emplace_back(std::make_pair(idx_divide[i], n + i));  

            // Generate Segment_3 instances for the new population of cells
            std::vector<Segment_3> segments_total = generateSegments<T>(cells_total);

            // Check the distance criterion for each daughter cell
            //
            // First compute the minimum distance to each daughter cell
            // in the new population
            Array<T, Dynamic, 1> daughter_mindists1(n_divide);
            Array<T, Dynamic, 1> daughter_mindists2(n_divide); 
            m = 0;
            for (const int i : idx_divide)
            {
                if (check_distance(m) == 1)
                {
                    T mindist = minDistToCell<T>(cells_total, segments_total, i);
                    daughter_mindists1(m) = mindist;
                }
                m++;
            }
            m = 0;
            for (int i = n; i < n + n_divide; ++i)
            {
                if (check_distance(m) == 1)
                {
                    T mindist = minDistToCell<T>(cells_total, segments_total, i);
                    daughter_mindists2(m) = mindist;
                }
                m++;
            }
            // Then determine whether the minimum distance to each pair
            // of daughter cells is within the desired tolerance 
            for (int i = 0; i < n_divide; ++i)
            {
               if (check_distance(i) == 1 &&
                   daughter_mindists1(i) >= mindist_default &&
                   daughter_mindists2(i) >= mindist_default)
               {
                   satisfies_distance(i) = 1;
               }
            }
            ntries++; 
        }

        // If the minimum distance criterion has not been satisfied for every
        // cell, then print a warning
        if (satisfies_distance.sum() < n_divide)
            std::cout << "[WARN] Cell division cannot satisfy minimum distance criterion";

        // Update cell ID and lineages 
        int max_id = parents.size() - 1; 
        for (int i = 0; i < n_divide; ++i)
        {
            int daughter_id1 = max_id + (2 * i + 1); 
            int daughter_id2 = max_id + (2 * i + 2);
            int parent_id = cells_total(idx_divide[i], __colidx_id); 
            cells_total(idx_divide[i], __colidx_id) = daughter_id1; 
            cells_total(n + i, __colidx_id) = daughter_id2; 
            parents.push_back(parent_id); 
            parents.push_back(parent_id); 
        }

        return std::make_pair(cells_total, daughter_pairs);  
    }
    else 
    {
        return std::make_pair(cells, daughter_pairs); 
    }
}

#endif
