/**
 * Functions for modeling cell growth and division. 
 * 
 * In what follows, a population of N cells is represented as a 2-D array of 
 * size (N, 11+), where each row represents a cell and stores the following data:
 * 
 * 0) cell ID
 * 1) x-coordinate of cell center
 * 2) y-coordinate of cell center
 * 3) x-coordinate of cell orientation vector
 * 4) y-coordinate of cell orientation vector
 * 5) cell length (excluding caps)
 * 6) half of cell length (excluding caps)
 * 7) timepoint at which cell was formed
 * 8) cell growth rate
 * 9) cell's ambient viscosity with respect to surrounding fluid
 * 10) cell-surface friction coefficient
 * 11) cell group identifier (integer, optional)
 * 12) plasmid copy-number (integer, optional)
 *
 * Additional features may be included in the array but these are not
 * relevant for the computations implemented here.
 *
 * Authors:
 *     Kee-Myoung Nam
 *
 * Last updated:
 *     8/1/2024
 */

#ifndef BIOFILM_CELL_GROWTH_HPP
#define BIOFILM_CELL_GROWTH_HPP

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
                cells(i, __colseq_r).matrix(), 
                cells(i, __colseq_n).matrix(),
                cells(i, __colidx_half_l),
                cells(j, __colseq_r).matrix(),
                cells(j, __colseq_n).matrix(),
                cells(j, __colidx_half_l),
                kernel
            );
            Matrix<T, 2, 1> dij = std::get<0>(result);
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
 * The function daughter_angle_dist() controls the orientations of the 
 * daughter cells: each daughter cell's orientation is obtained by
 * rotating the dividing cell's orientation by an angle sampled using
 * daughter_angle_dist(). 
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
 * @param daughter_angle_dist Function instance specifying the daughter 
 *                            cell re-orientation distribution.
 * @returns Updated population of cells. 
 */
template <typename T>
Array<T, Dynamic, Dynamic> divideCells(const Ref<const Array<T, Dynamic, Dynamic> >& cells,
                                       std::vector<int>& parents, 
                                       const T t, const T R, const T Rcell,
                                       const Ref<const Array<int, Dynamic, 1> >& to_divide,
                                       std::function<T(boost::random::mt19937&)>& growth_dist,
                                       boost::random::mt19937& rng,
                                       std::function<T(boost::random::mt19937&)>& daughter_length_dist,
                                       std::function<T(boost::random::mt19937&)>& daughter_angle_dist)
{
    // If there are cells to be divided ...
    const int n_divide = to_divide.sum();
    if (n_divide > 0)
    {
        // Get indices of cells to be divided
        int n = cells.rows(); 
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
            Array<T, Dynamic, 2> dividing_orientations(new_cells(Eigen::all, __colseq_n)); 
            Array<T, Dynamic, 1> theta1 = Array<T, Dynamic, 1>::Zero(n_divide);
            Array<T, Dynamic, 1> theta2 = Array<T, Dynamic, 1>::Zero(n_divide); 
            for (int i = 0; i < n_divide; ++i)
            {
                // Sample angles by which to rotate the daughter cells
                //
                // If there is no room to rotate the daughter cells or the
                // maximum number of tries have been attempted, then set 
                // both angles to zero
                if (ntries < ntries_total - 1 && check_distance(i) == 1)
                {
                    theta1(i) = daughter_angle_dist(rng); 
                    theta2(i) = daughter_angle_dist(rng);
                }
            }
            for (int i = 0; i < n_divide; ++i)
            {
                Array<T, 2, 1> u = dividing_orientations.row(i).transpose();
                Array<T, 2, 1> v1 = rotate<T>(u, theta1(i));
                cells_total(idx_divide[i], __colseq_n) = v1;
                Array<T, 2, 1> v2 = rotate<T>(u, theta2(i));
                new_cells(i, __colseq_n) = v2;
            }

            // Update cell lengths and positions ... 
            //
            // If cell division is symmetric, then the daughter cells both have
            // length L / 2 - R, where L is the length of the dividing cell, 
            // the point of division occurs at the center of the dividing cell,
            // and their x- and y-coordinates are perturbed by L / 4 + R / 2
            // along the daughter cells' orientation vectors 
            //
            // If not, then the daughter cells have lengths L1 = M * (L - 2 * R)
            // and L2 = (1 - M) * (L - 2 * R), where M is a random variable between
            // 0 and 1. The point of division occurs at cell body coordinate
            // L1 + R - Lh with Lh = L / 2, which has x- and y-coordinates given
            // by rx + (L1 + R - Lh) * nx and ry + (L1 + R - Lh) * ny, where
            // (rx, ry) and (nx, ny) are the position and orientation of the
            // dividing cell. The corresponding perturbations of the cell centers
            // from the point of division are given by R + L1 / 2 and R + L2 / 2
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

            // Update cell ID and lineages
            int max_id = parents.size() - 1; 
            for (int i = 0; i < n_divide; ++i)
            {
                int daughter_id1 = max_id + (2 * i + 1); 
                int daughter_id2 = max_id + (2 * i + 2);
                int parent_id = cells_total(idx_divide[i], __colidx_id); 
                cells_total(idx_divide[i], __colidx_id) = daughter_id1; 
                new_cells(i, __colidx_id) = daughter_id2; 
                parents.push_back(parent_id); 
                parents.push_back(parent_id); 
            }

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
            // Note that each daughter cell inherits its mother cell's viscosity
            // and friction coefficient
            cells_total(Eigen::seqN(n, n_divide), Eigen::all) = new_cells;

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

        return cells_total; 
    }
    else 
    {
        return cells; 
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
 * @param daughter_angle_dist Function instance specifying the daughter 
 *                            cell orientation distribution.
 * @returns Updated population of cells. 
 */
template <typename T>
Array<T, Dynamic, Dynamic> divideCells(const Ref<const Array<T, Dynamic, Dynamic> >& cells,
                                       std::vector<int>& parents, 
                                       const T t, const T R, const T Rcell,
                                       const Ref<const Array<int, Dynamic, 1> >& to_divide,
                                       std::vector<std::function<T(boost::random::mt19937&)> >& growth_dists,
                                       boost::random::mt19937& rng,
                                       std::function<T(boost::random::mt19937&)>& daughter_length_dist,
                                       std::function<T(boost::random::mt19937&)>& daughter_angle_dist)
{
    // If there are cells to be divided ...
    const int n_divide = to_divide.sum();

    if (n_divide > 0)
    {
        // Get indices of cells to be divided
        int n = cells.rows(); 
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
            Array<T, Dynamic, 2> dividing_orientations(new_cells(Eigen::all, __colseq_n));
            Array<T, Dynamic, 1> theta1 = Array<T, Dynamic, 1>::Zero(n_divide);
            Array<T, Dynamic, 1> theta2 = Array<T, Dynamic, 1>::Zero(n_divide); 
            for (int i = 0; i < n_divide; ++i)
            {
                // Sample angles by which to rotate the daughter cells
                //
                // If there is no room to rotate the daughter cells or the
                // maximum number of tries have been attempted, then set 
                // both angles to zero
                if (ntries < ntries_total - 1 && check_distance(i) == 1)
                {
                    theta1(i) = daughter_angle_dist(rng); 
                    theta2(i) = daughter_angle_dist(rng);
                }
            }
            for (int i = 0; i < n_divide; ++i)
            {
                Array<T, 2, 1> u = dividing_orientations.row(i).transpose();
                Array<T, 2, 1> v1 = rotate<T>(u, theta1(i));
                cells_total(idx_divide[i], __colseq_n) = v1;
                Array<T, 2, 1> v2 = rotate<T>(u, theta2(i));
                new_cells(i, __colseq_n) = v2;
            }

            // Update cell lengths and positions ... 
            //
            // If cell division is symmetric, then the daughter cells both have
            // length L / 2 - R, where L is the length of the dividing cell, 
            // the point of division occurs at the center of the dividing cell,
            // and their x- and y-coordinates are perturbed by L / 4 + R / 2
            // along the daughter cells' orientation vectors 
            //
            // If not, then the daughter cells have lengths L1 = M * (L - 2 * R)
            // and L2 = (1 - M) * (L - 2 * R), where M is a random variable between
            // 0 and 1. The point of division occurs at cell body coordinate
            // L1 + R - Lh with Lh = L / 2, which has x- and y-coordinates given
            // by rx + (L1 + R - Lh) * nx and ry + (L1 + R - Lh) * ny, where
            // (rx, ry) and (nx, ny) are the position and orientation of the
            // dividing cell. The corresponding perturbations of the cell centers
            // from the point of division are given by R + L1 / 2 and R + L2 / 2
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

            // Update cell ID and lineages
            int max_id = parents.size() - 1; 
            for (int i = 0; i < n_divide; ++i)
            {
                int daughter_id1 = max_id + (2 * i + 1); 
                int daughter_id2 = max_id + (2 * i + 2);
                int parent_id = cells_total(idx_divide[i], __colidx_id); 
                cells_total(idx_divide[i], __colidx_id) = daughter_id1; 
                new_cells(i, __colidx_id) = daughter_id2; 
                parents.push_back(parent_id); 
                parents.push_back(parent_id); 
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

        return cells_total; 
    }
    else 
    {
        return cells; 
    }
}

/**
 * Divide the indicated cells at the given time.
 *
 * This is an extended version of `divideCells()` that (1) assigns group-
 * specific growth rates and (2) partitions copies of a plasmid between 
 * daughter cells. The groups are assumed to be numbered 1, 2, 3, ..., and
 * the plasmid copy-numbers are stored in an additional (12th) column. 
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
 * @param daughter_angle_dist Function instance specifying the daughter 
 *                            cell orientation distribution.
 * @param partition_logratio_dist Function instance specifying the log-ratio
 *                                of plasmid copy-numbers to be partitioned 
 *                                among each pair of daughter cells.
 * @returns Updated population of cells. 
 */
template <typename T>
Array<T, Dynamic, Dynamic> divideCellsWithPlasmid(const Ref<const Array<T, Dynamic, Dynamic> >& cells,
                                                  std::vector<int>& parents, 
                                                  const T t, const T R, const T Rcell,
                                                  const Ref<const Array<int, Dynamic, 1> >& to_divide,
                                                  std::vector<std::function<T(boost::random::mt19937&)> >& growth_dists,
                                                  boost::random::mt19937& rng,
                                                  std::function<T(boost::random::mt19937&)>& daughter_length_dist,
                                                  std::function<T(boost::random::mt19937&)>& daughter_angle_dist,
                                                  std::function<T(boost::random::mt19937&)>& partition_logratio_dist)
{
    // If there are cells to be divided ...
    const int n_divide = to_divide.sum();

    if (n_divide > 0)
    {
        // Get indices of cells to be divided
        int n = cells.rows(); 
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
            Array<T, Dynamic, 2> dividing_orientations(new_cells(Eigen::all, __colseq_n)); 
            Array<T, Dynamic, 1> theta1 = Array<T, Dynamic, 1>::Zero(n_divide);
            Array<T, Dynamic, 1> theta2 = Array<T, Dynamic, 1>::Zero(n_divide); 
            for (int i = 0; i < n_divide; ++i)
            {
                // Sample angles by which to rotate the daughter cells
                //
                // If there is no room to rotate the daughter cells or the
                // maximum number of tries have been attempted, then set 
                // both angles to zero
                if (ntries < ntries_total - 1 && check_distance(i) == 1)
                {
                    theta1(i) = daughter_angle_dist(rng); 
                    theta2(i) = daughter_angle_dist(rng);
                }
            }
            for (int i = 0; i < n_divide; ++i)
            {
                Array<T, 2, 1> u = dividing_orientations.row(i).transpose();
                Array<T, 2, 1> v1 = rotate<T>(u, theta1(i));
                cells_total(idx_divide[i], __colseq_n) = v1;
                Array<T, 2, 1> v2 = rotate<T>(u, theta2(i));
                new_cells(i, __colseq_n) = v2;
            }

            // Update cell lengths and positions ... 
            //
            // If cell division is symmetric, then the daughter cells both have
            // length L / 2 - R, where L is the length of the dividing cell, 
            // the point of division occurs at the center of the dividing cell,
            // and their x- and y-coordinates are perturbed by L / 4 + R / 2
            // along the daughter cells' orientation vectors 
            //
            // If not, then the daughter cells have lengths L1 = M * (L - 2 * R)
            // and L2 = (1 - M) * (L - 2 * R), where M is a random variable between
            // 0 and 1. The point of division occurs at cell body coordinate
            // L1 + R - Lh with Lh = L / 2, which has x- and y-coordinates given
            // by rx + (L1 + R - Lh) * nx and ry + (L1 + R - Lh) * ny, where
            // (rx, ry) and (nx, ny) are the position and orientation of the
            // dividing cell. The corresponding perturbations of the cell centers
            // from the point of division are given by R + L1 / 2 and R + L2 / 2
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

            // Update cell ID and lineages
            int max_id = parents.size() - 1; 
            for (int i = 0; i < n_divide; ++i)
            {
                int daughter_id1 = max_id + (2 * i + 1); 
                int daughter_id2 = max_id + (2 * i + 2);
                int parent_id = cells_total(idx_divide[i], __colidx_id); 
                cells_total(idx_divide[i], __colidx_id) = daughter_id1; 
                new_cells(i, __colidx_id) = daughter_id2; 
                parents.push_back(parent_id); 
                parents.push_back(parent_id); 
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

            // Update plasmid copy-numbers
            //
            // For each pair of daughter cells, partition the plasmid copies 
            // according to a randomly sampled log-ratio
            for (int i = 0; i < n_divide; ++i)
            {
                int j = idx_divide[i];

                // The total number of plasmids in the dividing cell should
                // be double the given value to account for DNA replication
                int num_total = 2 * cells_total(j, __colidx_plasmid);

                // Get the ratio of plasmid copy-numbers in the daughter
                // cells, then partition accordingly 
                T ratio = pow(10.0, partition_logratio_dist(rng));
                T num_i = round((ratio * num_total) / (1.0 + ratio));
                T num_j = round(num_total - num_i);
                cells_total(j, __colidx_plasmid) = num_j; 
                new_cells(i, __colidx_plasmid) = num_i;
                std::cout << "... Partitioning " << num_total << " plasmids into "
                          << num_i << " and " << num_j << " (ratio = " << ratio
                          << ")" << std::endl;  
            }

            // Copy over daughter cell data
            //
            // Note that each daughter cell inherits its mother cell's viscosity
            // and friction coefficient
            cells_total(Eigen::seqN(n, n_divide), Eigen::all) = new_cells;

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

        return cells_total; 
    }
    else 
    {
        return cells; 
    }
}

#endif
