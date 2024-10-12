/**
 * Functions for imposing a growth void within a population of cells according 
 * to a criterion based on radial distance.
 *
 * In what follows, a population of N cells is represented as a 2-D array 
 * with N rows, whose columns are as specified in `indices.hpp`.
 *
 * Additional columns may be included in the array but these are not relevant
 * for the computations implemented here. 
 *
 * Authors:
 *     Kee-Myoung Nam
 *
 * Last updated:
 *     10/12/2024
 */

#ifndef BIOFILM_GROWTH_VOID_HPP
#define BIOFILM_GROWTH_VOID_HPP

#include <cmath>
#include <algorithm>
#include <Eigen/Dense>
#include <boost/math/constants/constants.hpp>
#include <boost/multiprecision/mpfr.hpp>
#include "indices.hpp"

using namespace Eigen; 

using std::cos;
using boost::multiprecision::cos;
using std::min; 
using boost::multiprecision::min;

/**
 * TODO Complete this docstring.
 *
 * @param cells
 * @param boundary_idx
 * @param in_void_func
 * @returns 
 */
template <typename T>
Array<int, Dynamic, 1> inGrowthVoid(const Ref<const Array<T, Dynamic, Dynamic> >& cells,
                                    std::vector<int>& boundary_idx, 
                                    std::function<bool(T)>& in_void_func)
{
    const int n = cells.rows(); 

    // Identify the center of mass of the cells 
    Array<T, 2, 1> center; 
    center << cells.col(__colidx_rx).mean(), cells.col(__colidx_ry).mean(); 

    // Find the radial distance of each cell to the center 
    Array<T, Dynamic, 2> radial = cells(Eigen::all, __colseq_r).rowwise() - center.transpose(); 
    Array<T, Dynamic, 1> rdists = radial.matrix().rowwise().norm().array();
    Array<T, Dynamic, 2> rdirs = radial.colwise() / rdists;

    // For each cell ... 
    const T cos_eps = cos(5.0 * boost::math::constants::pi<T>() / 180.0);
    Array<T, Dynamic, 1> rnorm = Array<T, Dynamic, 1>::Zero(n); 
    for (int i = 0; i < n; ++i)
    {
        // Is the cell a peripheral cell? 
        if (std::find(boundary_idx.begin(), boundary_idx.end(), i) != boundary_idx.end())
        {
            // Then set the normalized radial distance to 1
            rnorm(i) = 1.0; 
        }
        // Otherwise ... 
        else 
        {
            // Search for the peripheral cell with the shortest radial distance 
            // among those whose (1) radial distance is larger than that of cell i
            // and (2) azimuth is less than 5 degrees from that of cell i
            int bound_i = -1;
            T dist_bound_i = std::numeric_limits<T>::infinity();

            // Identify all peripheral cells whose radial distance is larger  
            // than that of cell i and whose azimuth relative to cell i is
            // less than 5 degrees 
            for (const int& j : boundary_idx)
            {
                T cos_delta = rdirs.row(j).matrix().dot(rdirs.row(i).matrix()); 
                if (rdists(j) > rdists(i) && cos_delta > cos_eps)
                {
                    // If this peripheral cell has a sufficiently close azimuth, 
                    // then identify its radial distance 
                    if (dist_bound_i > rdists(j))
                    {
                        bound_i = j; 
                        dist_bound_i = rdists(j); 
                    }
                }
            }
            // If no peripheral cell matching the above description exists, 
            // then simply choose the peripheral cell with the closest azimuth
            if (bound_i == -1)
            {
                T curr_cos_delta = -1;   // Corresponding to 180 degrees
                for (const int& j : boundary_idx) 
                {
                    T cos_delta = rdirs.row(j).matrix().dot(rdirs.row(i).matrix()); 
                    if (cos_delta > curr_cos_delta)
                    {
                        bound_i = j; 
                        dist_bound_i = rdists(j);
                        curr_cos_delta = cos_delta; 
                    }
                }
            }
            // Now that the peripheral cell has been found, normalize the radial
            // distance of cell i by the radial distance of the peripheral cell
            //
            // Ensure also that this normalized distance is <= 1
            #ifdef DEBUG_CHECK_IF_PERIPHERAL_CELL_IN_RADIAL_DIRECTION_WAS_FOUND
                if (bound_i == -1)
                {
                    std::cerr << "Failed to find peripheral cell in radial "
                              << "direction from cell " << i << std::endl; 
                    std::cerr << "Cell center = (" << cells(i, __colidx_rx)
                              << ", " << cells(i, __colidx_ry) << ")"
                              << std::endl;
                    throw std::runtime_error(
                        "Failed to find peripheral cell in radial direction"
                    ); 
                }
            #endif
            rnorm(i) = min(1.0, rdists(i) / rdists(bound_i));
        } 
    }

    // Determine, from the normalized radial distance, whether the cell is 
    // in the growth void 
    Array<int, Dynamic, 1> in_void(n);
    for (int i = 0; i < n; ++i)
        in_void(i) = static_cast<int>(in_void_func(rnorm(i)));

    return in_void;
}

#endif
