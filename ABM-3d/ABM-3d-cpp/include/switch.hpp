/**
 * Functions for cell state switching. 
 * 
 * In what follows, a population of N cells is represented as a 2-D array of 
 * size (N, 14+), where each row represents a cell and stores the following data:
 * 
 * 0) x-coordinate of cell center
 * 1) y-coordinate of cell center
 * 2) z-coordinate of cell center
 * 3) x-coordinate of cell orientation vector
 * 4) y-coordinate of cell orientation vector
 * 5) z-coordinate of cell orientation vector
 * 6) cell length (excluding caps)
 * 7) half of cell length (excluding caps) 
 * 8) timepoint at which the cell was formed
 * 9) cell growth rate
 * 10) cell's ambient viscosity with respect to surrounding fluid
 * 11) cell-surface friction coefficient
 * 12) cell-surface adhesion energy density
 * 13) cell group identifier (integer, optional)
 *
 * Additional features may be included in the array but these are not 
 * relevant for the computations implemented here.
 *
 * Authors:
 *     Kee-Myoung Nam
 *
 * Last updated:
 *     2/23/2024
 */

#ifndef BIOFILM_SWITCH_3D_HPP
#define BIOFILM_SWITCH_3D_HPP

#include <cmath>
#include <Eigen/Dense>

using namespace Eigen;

/**
 * TODO Complete this docstring.
 *
 * @param cells Existing population of cells.
 * @param switch_attribute
 * @param n_groups
 * @param dt
 * @param switch_rates
 * @param growth_dists
 * @param attribute_dists
 * @param rng
 * @param uniform_dist
 */
template <typename T>
void switchGroups(Ref<Array<T, Dynamic, Dynamic> > cells,
                  const int switch_attribute, const int n_groups, const T dt,
                  const Ref<const Array<T, Dynamic, Dynamic> >& switch_rates,
                  std::vector<std::function<T(boost::random::mt19937&)> >& growth_dists,
                  std::vector<std::function<T(boost::random::mt19937&)> >& attribute_dists,
                  boost::random::mt19937& rng,
                  boost::random::uniform_01<>& uniform_dist)
{
    // First identify cells to switch groups within the given timestep
    Array<T, Dynamic, Dynamic> switch_probs = dt * switch_rates;
    for (int i = 0; i < cells.rows(); ++i)
    {
        // Which group is the cell currently in? 
        int group = static_cast<int>(cells(i, 13)) - 1;   // Groups are indexed 1, 2, ...

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
                    // new growth rate and attribute value 
                    T growth_rate = growth_dists[j](rng);
                    T attribute = attribute_dists[j](rng);
                    cells(i, 9) = growth_rate;
                    cells(i, switch_attribute) = attribute; 
                    cells(i, 13) = j + 1;
                    break;
                }
                total += switch_prob;
            }
        }
    }
}

#endif
