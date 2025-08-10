/**
 * Functions for cell state switching. 
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
 *     8/10/2025
 */

#ifndef BIOFILM_SWITCH_3D_HPP
#define BIOFILM_SWITCH_3D_HPP

#include <cmath>
#include <Eigen/Dense>
#include "indices.hpp"

using namespace Eigen;

enum class SwitchMode
{
    NONE = 0,
    MARKOV = 1,
    INHERIT = 2
};

/**
 * Randomly switch cells from one group to another.
 *
 * @param cells Existing population of cells.
 * @param n_groups Number of distinct groups.
 * @param dt Timestep. 
 * @param switch_rates Matrix of switching rates between groups. 
 * @param growth_dists Vector of growth rate distributions, one for each
 *                     group.
 * @param rng Random number generator. 
 * @param uniform_dist Pre-defined uniform distribution between 0 and 1.
 */
template <typename T>
void switchGroupsMarkov(Ref<Array<T, Dynamic, Dynamic> > cells,
                        const int n_groups, const T dt,
                        const Ref<const Array<T, Dynamic, Dynamic> >& switch_rates,
                        std::vector<std::function<T(boost::random::mt19937&)> >& growth_dists,
                        boost::random::mt19937& rng,
                        boost::random::uniform_01<>& uniform_dist)
{
    // First identify cells to switch groups within the given timestep
    Array<T, Dynamic, Dynamic> switch_probs = dt * switch_rates;
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
                    // new growth rate
                    cells(i, __colidx_group) = j + 1;
                    T growth_rate = growth_dists[j](rng);
                    cells(i, __colidx_growth) = growth_rate;
                    break;
                }
                total += switch_prob;
            }
        }
    }
}

/**
 * Randomly switch a subset of cells from one group to another, based on 
 * whether they have just divided.
 *
 * @param cells Existing population of cells.
 * @param daughter_pairs Indices of pairs of daughter cells that have just 
 *                       divided. 
 * @param n_groups Number of distinct groups.
 * @param switch_probs Matrix of switching probabilities between groups. 
 * @param growth_dists Vector of growth rate distributions, one for each
 *                     group.
 * @param rng Random number generator.
 * @param uniform_dist Pre-defined uniform distribution between 0 and 1.
 */
template <typename T>
void switchGroupsInherit(Ref<Array<T, Dynamic, Dynamic> > cells,
                         std::vector<std::pair<int, int> >& daughter_pairs,
		         const int n_groups, 
		         const Ref<const Array<T, Dynamic, Dynamic> >& switch_probs,
		         std::vector<std::function<T(boost::random::mt19937&)> >& growth_dists,
		         boost::random::mt19937& rng,
		         boost::random::uniform_01<>& uniform_dist)
{
    // For each pair of daughter cells ... 
    for (const std::pair<int, int>& pair : daughter_pairs)
    {
	int i = pair.first;
	int j = pair.second; 

        // Which group are the cells currently in? (Same as the mother cell) 
        int group = static_cast<int>(cells(i, __colidx_group)) - 1;   // Groups are indexed 1, 2, ...

        // Decide whether to switch to any other group
        T r = uniform_dist(rng);
        T total = switch_probs(group, group);
	if (r > total)    // In this case, we are switching to some other group  
	{
	    for (int k = 0; k < n_groups; ++k)
	    {
		if (k != group)
		{
		    T switch_prob = switch_probs(group, k);
		    if (r > total && r < total + switch_prob)
		    {
			// If one of the daughter cells is to switch to group k, 
			// first choose the daughter cell to switch ...
			T s = uniform_dist(rng); 
			int idx_switch = (s < 0.5 ? i : j); 
			// ... then sample the cell's new growth rate 
			cells(idx_switch, __colidx_group) = k + 1;
			T growth_rate = growth_dists[k](rng);
			cells(idx_switch, __colidx_growth) = growth_rate;
			break;
		    }
		    total += switch_prob;
		}
	    }
	}
    }
}

#endif
