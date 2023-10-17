"""
Functions for cell state switching. 

Authors:
    Kee-Myoung Nam

Last updated:
    10/16/2023
"""
import numpy as np
from numba import njit

#######################################################################
# In what follows, a population of N cells is represented as a 2-D array of 
# size (N, 10), where each row represents a cell and stores the following data:
# 
# 0) x-coordinate of cell center
# 1) y-coordinate of cell center
# 2) x-coordinate of cell orientation vector
# 3) y-coordinate of cell orientation vector
# 4) cell length (excluding caps) 
# 5) timepoint at which the cell was formed
# 6) cell growth rate
# 7) cell's ambient viscosity with respect to surrounding fluid
# 8) cell-surface friction coefficient
# 9) cell group identifier (1 or 2)
#######################################################################
@njit(fastmath=True)
def choose_cells_to_switch(cells, rate_12, rate_21, dt, rng):
    """
    Given rates of switching from group 1 to group 2 and vice versa and a 
    small time increment, randomly sample a subset of cells to switch from
    one group to the other within the time increment.

    Parameters
    ----------
    cells : `numpy.ndarray`
        Existing population of cells. 
    rate_12 : float
        Rate of switching from group 1 to group 2.
    rate_21 : float
        Rate of switching from group 2 to group 1.
    dt : float
        Time increment for switching.
    rng : `numpy.random.Generator`
        Random number generator.

    Returns
    -------
    Boolean index indicating which cells are to switch from one group to
    the other. 
    """
    # Switching from group 1 to group 2 (resp. group 2 to group 1) occurs 
    # with probability rate_12 * dt (resp. rate_21 * dt) within the time 
    # increment dt
    prob_12 = rate_12 * dt
    prob_21 = rate_21 * dt
    in_group_1 = (cells[:, 9] == 1)
    to_switch = np.zeros((cells.shape[0],), dtype=np.int32)
    for i in range(cells.shape[0]):
        if in_group_1[i]:
            to_switch[i] = (rng.uniform() < prob_12)   # Switch from 1 to 2?
        else:
            to_switch[i] = (rng.uniform() < prob_21)   # Switch from 2 to 1?

    return to_switch

#######################################################################
@njit(fastmath=True)
def switch_features(cells, feature_idx, to_switch, dist1, dist2, rng):
    """
    Switch the indicated cells on the basis of the given feature from one 
    distribution to the other.

    The cells are assumed to each exist in one of two states, each of which
    has a distribution for the given feature.

    The feature should be one of growth rate (6), ambient viscosity (7), 
    surface friction coefficient (8), or an additionally specified feature
    (10+).

    New feature values are chosen using the given distribution functions,
    each of which must take the random number generator `rng` as its single
    input.

    Parameters
    ----------
    cells : `numpy.ndarray`
        Existing population of cells.
    feature_idx : int
        Index of column containing the given feature. 
    to_switch : `numpy.ndarray`
        Boolean index indicating which cells are to switch from one
        distribution to the other.
    dist1 : function
        Feature distribution function for group 1. Must take
        `numpy.random.Generator` as its single argument.
    dist2 : function
        Feature distribution function for group 2. Must take
        `numpy.random.Generator` as its single argument.
    rng : `numpy.random.Generator`
        Random number generator.

    Returns
    -------
    Updated population array.
    """
    n_switch = to_switch.sum()

    # If there are cells to switch ...
    if n_switch > 0:
        # Get indices of cells to be switched from 1 to 2 and vice versa
        to_switch_from_1_to_2 = (to_switch & (cells[:, 9] == 1))
        to_switch_from_2_to_1 = (to_switch & (cells[:, 9] == 2))
        n_switch_from_1_to_2 = to_switch_from_1_to_2.sum()
        n_switch_from_2_to_1 = to_switch_from_2_to_1.sum()
        idx_switch_from_1_to_2 = np.where(to_switch_from_1_to_2)[0]
        idx_switch_from_2_to_1 = np.where(to_switch_from_2_to_1)[0]

        # Sample new feature values for each such cell
        feature_new_1 = np.array([dist1(rng) for _ in idx_switch_from_2_to_1])
        feature_new_2 = np.array([dist2(rng) for _ in idx_switch_from_1_to_2])

        # Switch groups and assign new growth rates
        if n_switch_from_1_to_2 > 0:
            cells[idx_switch_from_1_to_2, 9] = 2
            cells[idx_switch_from_1_to_2, feature_idx] = feature_new_2
        if n_switch_from_2_to_1 > 0:
            cells[idx_switch_from_2_to_1, 9] = 1
            cells[idx_switch_from_2_to_1, feature_idx] = feature_new_1

    return cells

