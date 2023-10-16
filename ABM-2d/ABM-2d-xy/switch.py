"""
Functions for cell state switching. 

Authors:
    Kee-Myoung Nam

Last updated:
    10/16/2023
"""

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
        idx_switch_from_1_to_2 = np.where(to_switch_from_1_to_2)[0]
        idx_switch_from_2_to_1 = np.where(to_switch_from_2_to_1)[0]

        # Sample new feature values for each such cell
        feature_new_1 = np.array([dist1(rng) for _ in idx_switch_from_2_to_1])
        feature_new_2 = np.array([dist2(rng) for _ in idx_switch_from_1_to_2])

        # Switch groups and assign new growth rates
        cells[to_switch_from_1_to_2, 9] = 2
        cells[to_switch_from_1_to_2, feature_idx] = feature_new_2
        cells[to_switch_from_2_to_1, 9] = 1
        cells[to_switch_from_2_to_1, feature_idx] = feature_new_1

    return cells

