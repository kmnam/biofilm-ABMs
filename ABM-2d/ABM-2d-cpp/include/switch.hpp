/**
 * Functions for cell state switching. 
 * 
 * In what follows, a population of N cells is represented as a 2-D array of 
 * size (N, 11+), where each row represents a cell and stores the following data:
 * 
 * 0) x-coordinate of cell center
 * 1) y-coordinate of cell center
 * 2) x-coordinate of cell orientation vector
 * 3) y-coordinate of cell orientation vector
 * 4) cell length (excluding caps)
 * 5) half of cell length (excluding caps) 
 * 6) timepoint at which the cell was formed
 * 7) cell growth rate
 * 8) cell's ambient viscosity with respect to surrounding fluid
 * 9) cell-surface friction coefficient
 * 10) cell group identifier (integer, optional)
 *
 * Additional features may be included in the array but these are not 
 * relevant for the computations implemented here.
 *
 * Authors:
 *     Kee-Myoung Nam
 *
 * Last updated:
 *     11/8/2023
 */

#ifndef BIOFILM_SWITCH_HPP
#define BIOFILM_SWITCH_HPP

#include <cmath>
#include <Eigen/Dense>

using namespace Eigen;

/**
 * Given rates of switching from group 1 to group 2 and vice versa and a
 * small time increment, randomly sample a subset of cells to switch from
 * one group to the other within the time increment.
 *
 * @param cells Existing population of cells.
 * @param rate_12 Rate of switching from group 1 to group 2. 
 * @param rate_21 Rate of switching from group 2 to group 1.
 * @param dt Time increment for switching.
 * @param rng Random number generator.
 * @param dist Pre-defined instance of a standard uniform distribution.
 * @returns Boolean index indicating which cells are to switch from one
 *          group to the other. 
 */
template <typename T>
Array<int, Dynamic, 1> chooseCellsToSwitch(const Ref<const Array<T, Dynamic, Dynamic> >& cells, 
                                           const T rate_12, const T rate_21,
                                           const T dt, boost::random::mt19937& rng,
                                           boost::random::uniform_01<>& dist)
{
    // Switching from group 1 to group 2 (resp. group 2 to group 1) occurs
    // with probability rate_12 * dt (resp. rate_21 * dt) within the time
    // increment dt
    int n = cells.rows(); 
    T prob_12 = rate_12 * dt;
    T prob_21 = rate_21 * dt;
    Array<bool, Dynamic, 1> in_group_1 = (cells.col(10) == 1);
    Array<int, Dynamic, 1> to_switch = Array<int, Dynamic, 1>::Zero(n);
    for (int i = 0; i < n; ++i)
    {
        if (in_group_1(i))
            to_switch(i) = static_cast<int>(dist(rng) < prob_12);   // Switch from 1 to 2? 
        else
            to_switch(i) = static_cast<int>(dist(rng) < prob_21);   // Switch from 2 to 1?
    } 

    return to_switch;
}

/**
 * Switch the indicated cells on the basis of the given feature from one 
 * distribution to the other.
 *
 * The given array of cell data is updated in place. 
 *
 * The cells are assumed to each exist in one of two states, each of which 
 * has a distribution for the given feature. 
 *
 * The feature should be one of growth rate (7), ambient viscosity (8), 
 * surface friction coefficient (9), or an additionally specified feature
 * (11+). 
 *
 * New feature values are chosen using the given distribution functions, 
 * each of which must take a random number generator as its single input.
 *
 * @param cells Existing population of cells.
 * @param feature_idx Index of column containing the given feature.
 * @param to_switch Boolean index indicating which cells are to switch from
 *                  one distribution to the other. 
 * @param dist1 Feature distribution function for group 1.
 * @param dist2 Feature distribution function for group 2.
 * @param rng Random number generator.
 */
template <typename T>
void switchGroups(Ref<Array<T, Dynamic, Dynamic> > cells, const int feature_idx,
                  const Ref<const Array<int, Dynamic, 1> >& to_switch, 
                  std::function<T(boost::random::mt19937&)>& dist1, 
                  std::function<T(boost::random::mt19937&)>& dist2,
                  boost::random::mt19937& rng)
{
    int n_switch = to_switch.sum(); 

    // If there are cells to switch ... 
    if (n_switch > 0)
    {
        // For each cell to be switched from 1 to 2 or vice versa, sample 
        // and assign new feature value 
        int n = cells.rows();
        for (int i = 0; i < n; ++i)
        {
            if (to_switch(i) && cells(i, 10) == 1)    // Switching from 1 to 2
            {
                T value = dist2(rng);
                cells(i, 10) = 2;
                cells(i, feature_idx) = value; 
            }
            else if (to_switch(i))                    // Switching from 2 to 1
            {
                T value = dist1(rng);
                cells(i, 10) = 1;
                cells(i, feature_idx) = value;
            }
        }
    }
}

/**
 * Switch the indicated cells on the basis of growth rate and cell-surface
 * friction coefficient.
 *
 * The given array of cell data is updated in place. 
 *
 * The cells are assumed to each exist in one of two states, each of which 
 * has a distribution for both features. 
 *
 * New feature values are chosen using the given distribution functions, 
 * each of which must take a random number generator as its single input.
 *
 * @param cells Existing population of cells.
 * @param feature_idx Index of column containing the given feature.
 * @param to_switch Boolean index indicating which cells are to switch from
 *                  one distribution to the other. 
 * @param growth_dist1 Growth rate distribution function for group 1.
 * @param growth_dist2 Growth rate distribution function for group 2.
 * @param friction_dist1 Cell-surface friction coefficient distribution
 *                       function for group 1.
 * @param friction_dist2 Cell-surface friction coefficient distribution
 *                       function for group 2.
 * @param rng Random number generator.
 */
template <typename T>
void switchByGrowthRateAndFrictionCoeff(Ref<Array<T, Dynamic, Dynamic> > cells,
                                        const Ref<const Array<int, Dynamic, 1> >& to_switch, 
                                        std::function<T(boost::random::mt19937&)>& growth_dist1, 
                                        std::function<T(boost::random::mt19937&)>& growth_dist2,
                                        std::function<T(boost::random::mt19937&)>& friction_dist1,
                                        std::function<T(boost::random::mt19937&)>& friction_dist2,
                                        boost::random::mt19937& rng)
{
    int n_switch = to_switch.sum(); 

    // If there are cells to switch ... 
    if (n_switch > 0)
    {
        // For each cell to be switched from 1 to 2 or vice versa, sample 
        // and assign new feature value 
        int n = cells.rows();
        for (int i = 0; i < n; ++i)
        {
            if (to_switch(i) && cells(i, 10) == 1)    // Switching from 1 to 2
            {
                cells(i, 10) = 2;
                cells(i, 7) = growth_dist2(rng);      // New growth rate
                cells(i, 9) = friction_dist2(rng);    // New friction coefficient
            }
            else if (to_switch(i))                    // Switching from 2 to 1
            {
                cells(i, 10) = 1;
                cells(i, 7) = growth_dist1(rng);      // New growth rate
                cells(i, 9) = friction_dist1(rng);    // New friction coefficient
            }
        }
    }
}

#endif
