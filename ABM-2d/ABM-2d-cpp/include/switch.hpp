/**
 * Functions for cell state switching. 
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

#ifndef BIOFILM_SWITCH_HPP
#define BIOFILM_SWITCH_HPP

#include <cmath>
#include <Eigen/Dense>
#include "indices.hpp"

using namespace Eigen;

/**
 * Randomly switch cells from one group to another.
 *
 * @param cells Existing population of cells.
 * @param switch_attributes Indices of attributes to switch when switching 
 *                          between the groups.
 * @param n_groups Number of distinct groups.
 * @param dt Timestep.
 * @param switch_rates Matrix of switching rates between groups. 
 * @param growth_dists Vector of growth rate distributions, one for each
 *                     group.
 * @param attribute_dists Dictionary of attribute distributions, one for each
 *                        group-attribute pair.
 * @param rng Random number generator.
 * @param uniform_dist Pre-defined uniform distribution between 0 and 1.
 */
template <typename T>
void switchGroups(Ref<Array<T, Dynamic, Dynamic> > cells,
                  std::vector<int>& switch_attributes,
                  const int n_groups, const T dt,
                  const Ref<const Array<T, Dynamic, Dynamic> >& switch_rates,
                  std::vector<std::function<T(boost::random::mt19937&)> >& growth_dists,
                  std::map<std::pair<int, int>, std::function<T(boost::random::mt19937&)> >& attribute_dists,
                  boost::random::mt19937& rng,
                  boost::random::uniform_01<>& uniform_dist)
{
    // First identify cells to switch groups within the given timestep
    Array<T, Dynamic, Dynamic> switch_probs = dt * switch_rates;
    const int n_attributes = switch_attributes.size();
    for (int i = 0; i < cells.rows(); ++i)
    {
        // Which group is the cell currently in? 
        int group = static_cast<int>(cells(i, __colidx_group)) - 1;   // Groups are indexed 1, 2, ...

        // Decide whether to switch to any other group
        T r = uniform_dist(rng);
        T total = 0;
        for (int j = 0; j < n_groups; ++j)
        {
            if (j != group)
            {
                T switch_prob = switch_probs(group, j);
                if (r > total && r < total + switch_prob)
                {
                    // If the cell is to switch to group j, sample the cell's
                    // new growth rate and attribute values 
                    cells(i, __colidx_group) = j + 1;
                    T growth_rate = growth_dists[j](rng);
                    cells(i, __colidx_growth) = growth_rate;
                    for (int k = 0; k < n_attributes; ++k)
                    {
                        auto pair = std::make_pair(j, k);
                        T attribute = attribute_dists[pair](rng);
                        cells(i, switch_attributes[k]) = attribute;
                    }
                    break;
                }
                total += switch_prob;
            }
        }
    }
}

#endif
