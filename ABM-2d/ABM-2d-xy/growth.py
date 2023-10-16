"""
Functions for modeling cell growth and division. 

Authors:
    Kee-Myoung Nam, JP Nijjer

Last updated:
    10/16/2023
"""

import numpy as np

#######################################################################
# In what follows, a population of N cells is represented as a 2-D array of 
# size (N, 10+), where each row represents a cell and stores the following data:
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
# 9) cell group identifier (integer, optional)
#
# Additional features may be included in the array but these are not 
# relevant for the computations implemented here
#######################################################################
def grow_cells(cells, dt, R):
    """
    Grow the cells in the given population according to the exponential 
    volume growth law. 

    Parameters
    ----------
    cells : `numpy.ndarray`
        Existing population of cells.
    dt : float
        Timestep. 
    R : float
        Cell radius.

    Returns
    -------
    Updated population of cells.
    """
    # Each cell grows in length according to an exponential growth law
    cells[:, 4] += cells[:, 6] * (4 * R / 3 + cells[:, 4]) * dt

    return cells

########################################################################
def divide_max_length(cells, Ldiv):
    """
    Identify the cells that exceed the given division length.

    Parameters
    ----------
    cells : `numpy.ndarray`
        Existing population of cells.
    Ldiv : float
        Cell division length.

    Returns
    -------
    Boolean index indicating which cells are to divide.
    """
    return (cells[:, 4] > Ldiv)

########################################################################
def divide_adder(cells, Linc, idx=-1):
    """
    Identify the cells that have grown the given length increment. (This
    corresponds to the adder model.)

    By default, the initial length of each cell is assumed to be stored 
    in the last column of the array, unless an index (`idx`) is specified.

    Parameters
    ----------
    cells : `numpy.ndarray`
        Existing population of cells.
    Linc : float
        Length increment.
    idx : int
        Index of column storing cell initial lengths. -1 by default. 

    Returns
    -------
    Boolean index indicating which cells are to divide.
    """
    return (cells[:, 4] - cells[:, idx] > Linc)

########################################################################
def divide_cells(cells, t, R, to_divide, growth_dist, rng, daughter_length_std=0,
                 orientation_conc=None):
    """
    Divide the cells that exceed the division length at the given time.

    If `i` is the index of a dividing cell, row `i` is updated to store 
    one daughter cell, and a new row is appended to store the other 
    daughter cell.

    The growth rate of each daughter cell is chosen using the given 
    distribution function, which must take the random number generator
    `rng` as its single input. 

    The initial length of each daughter cell is recorded if the input 
    array has the capacity.

    The parameter `daughter_length_std` controls the degree of asymmetry
    in cell division: if `daughter_length_std > 0`, then the daughter
    cells are determined to have lengths `M * (L - 2 * R)` and
    `(1 - M) * (L - 2 * R)`, where `L` is the length of the dividing cell,
    `R` is the cell radius, and `M` is a normally distributed variable with
    mean 0.5 and standard deviation `daughter_length_std`.

    The parameter `orientation_conc` controls the orientations of the 
    daughter cells: if `orientation_conc is None`, then the daughter cells
    have the same orientation as the dividing cell; if not, then each
    daughter cell's orientation is obtained by rotating the dividing cell's
    orientation by theta, which is sampled from a von Mises distribution with
    mean 0 and concentration `orientation_conc`.

    Parameters
    ----------
    cells : `numpy.ndarray`
        Existing population of cells.
    t : float
        Current time.
    R : float
        Cell radius.
    to_divide : `numpy.ndarray`
        Boolean index indicating which cells are to divide.
    growth_dist : function
        Growth rate sampling distribution function. Must take
        `numpy.random.Generator` as its single argument.
    rng : `numpy.random.Generator`
        Random number generator.
    daughter_length_std : float
        Standard deviation of daughter cell length distribution. Zero by 
        default (in which case cell division is symmetric).
    orientation_conc : float
        Von Mises concentration parameter for sampling daughter cell
        orientations. None by default (in which case the daughter cells
        simply have the same orientation as the dividing cell). 
    
    Returns
    -------
    Updated population array.
    """
    # If there are cells to be divided ...
    n_divide = to_divide.sum()
    if n_divide > 0:
        # Get indices of cells to be divided
        idx_divide = np.where(to_divide)[0]

        # Get a copy of the corresponding sub-array
        new_cells = cells[to_divide, :]

        # Update cell orientations ...
        # 
        # If orientation_conc is not None, then the daughter cells have 
        # orientations that are obtained by rotating the dividing cell's 
        # orientation counterclockwise by theta, where theta follows a 
        # von Mises distribution with mean 0 and concentration parameter
        # given by orientation_conc
        if orientation_conc is not None:
            dividing_orientations = cells[to_divide, 2:4]
            theta1 = rng.vonmises(0, orientation_conc, size=(n_divide,))
            for i, j in enumerate(idx_divide):
                rot = np.array(
                    [
                        [np.cos(theta1[i]), -np.sin(theta1[i])],
                        [np.sin(theta1[i]), np.cos(theta1[i])]
                    ],
                    dtype=np.float64
                )
                cells[j, 2:4] = np.dot(rot, dividing_orientations[i, :])
            theta2 = rng.vonmises(0, orientation_conc, size=(n_divide,))
            for i in range(n_divide):
                rot = np.array(
                    [
                        [np.cos(theta2[i]), -np.sin(theta2[i])],
                        [np.sin(theta2[i]), np.cos(theta2[i])]
                    ],
                    dtype=np.float64
                )
                new_cells[i, 2:4] = np.dot(rot, dividing_orientations[i, :])

        # Update cell lengths and positions ...
        #
        # If cell division is symmetric, then the daughter cells both have
        # length L / 2 - R, where L is the length of the dividing cell, 
        # the point of division occurs at the center of the dividing cell,
        # and their x- and y-coordinates are perturbed by L / 4 + R / 2
        # along the daughter cells' orientation vectors
        #
        # If not, then the daughter cells have lengths L1 = M * (L - 2 * R)
        # and L2 = (1 - M) * (L - 2 * R), where M is a normally distributed
        # variable with mean 0.5 and standard deviation given by
        # daughter_length_std. The point of division occurs at cell body 
        # coordinate L1 + R - Lh with Lh = L / 2, which has x- and y-coordinates
        # given by rx + (L1 + R - Lh) * nx and ry + (L1 + R - Lh) * ny, where
        # (rx, ry) and (nx, ny) are the position and orientation of the dividing
        # cell. The corresponding perturbations of the cell centers from the
        # point of division are given by R + M * (L - 2 * R) / 2 and
        # R + (1 - M) * (L - 2 * R) / 2
        if daughter_length_std == 0:
            delta = cells[to_divide, 4] / 4 + R / 2
            cells[to_divide, 4] = cells[to_divide, 4] / 2 - R
            new_cells[:, 4] = new_cells[:, 4] / 2 - R
            cells[to_divide, 0] = cells[to_divide, 0] - delta * cells[to_divide, 2]
            new_cells[:, 0] = new_cells[:, 0] + delta * new_cells[:, 2]
            cells[to_divide, 1] = cells[to_divide, 1] - delta * cells[to_divide, 3]
            new_cells[:, 1] = new_cells[:, 1] + delta * new_cells[:, 3]
        else:
            # Sample a normally distributed daughter cell length for each
            # cell to be divided 
            M = rng.normal(0.5, daughter_length_std, size=(n_divide,))
            # Locate point of division along dividing cell centerline
            div = M * (cells[to_divide, 4] - 2 * R) + R - (cells[to_divide, 4] / 2)
            # Get perturbations from point of division along cell centerline 
            # for the daughter cell centers
            delta1 = R + M * (cells[to_divide, 4] - 2 * R) / 2
            delta2 = R + (1 - M) * (new_cells[:, 4] - 2 * R) / 2
            # Define daughter cell lengths and locate daughter cell centers
            cells[to_divide, 4] = M * (cells[to_divide, 4] - 2 * R)
            new_cells[:, 4] = (1 - M) * (new_cells[:, 4] - 2 * R)
            cells[to_divide, 0] = (
                cells[to_divide, 0] + (div - delta1) * cells[to_divide, 2]
            )
            new_cells[:, 0] = (
                new_cells[:, 0] + (div + delta2) * new_cells[:, 2]
            )
            cells[to_divide, 1] = (
                cells[to_divide, 1] + (div - delta1) * cells[to_divide, 3]
            )
            new_cells[:, 1] = (
                new_cells[:, 1] + (div + delta2) * new_cells[:, 3]
            )

        # Update cell birth times 
        cells[to_divide, 5] = t
        new_cells[:, 5] = t

        # Update cell growth rates (sample from specified distribution)
        for i, j in enumerate(idx_divide):
            cells[j, 6] = growth_dist(rng)
            new_cells[i, 6] = growth_dist(rng)

        # Append new cells to the population and return
        # 
        # Note that each daughter cell inherits its mother cell's viscosity
        # and friction coefficient
        cells = np.append(cells, new_cells, axis=0)

    return cells

########################################################################
def divide_cells_by_group(cells, t, R, to_divide, growth_dists, rng, group_ids,
                          daughter_length_std=0, orientation_conc=None):
    """
    Divide the cells that exceed the division length at the given time.

    This function extends `divide_cells()` to incorporate group-dependent
    growth rates; see `divide_cells()` for further details. 

    The growth rate of each daughter cell is chosen using the given family of 
    distribution functions, `growth_dists`, where the i-th distribution 
    function corresponds to the i-th group. If there are not as many 
    distribution functions as there are groups, an error is raised.

    Parameters
    ----------
    cells : `numpy.ndarray`
        Existing population of cells.
    t : float
        Current time.
    R : float
        Cell radius.
    to_divide : `numpy.ndarray`
        Boolean index indicating which cells are to divide.
    growth_dists : list of functions
        List of growth rate sampling distribution functions. Each must take
        `numpy.random.Generator` as its single argument.
    rng : `numpy.random.Generator`
        Random number generator.
    group_ids : list of ints
        List of group identifiers. There should be as many groups as there
        are growth rate distribution functions.
    daughter_length_std : float
        Standard deviation of daughter cell length distribution. Zero by 
        default (in which case cell division is symmetric).
    orientation_conc : float
        Von Mises concentration parameter for sampling daughter cell
        orientations. None by default (in which case the daughter cells
        simply have the same orientation as the dividing cell). 
    
    Returns
    -------
    Updated population array.
    """
    # If there are cells to be divided ...
    n_divide = to_divide.sum()
    if n_divide > 0:
        # Get indices of cells to be divided
        idx_divide = np.where(to_divide)[0]

        # Check that the correct number of growth rate distribution functions
        # were specified
        groups = cells[:, 9].astype(np.int32)
        n_groups = len(group_ids)
        if len(growth_dists) != n_groups:
            raise RuntimeError(
                'Growth rate distribution functions incorrectly specified'
            )

        # Get a copy of the corresponding sub-array
        new_cells = cells[to_divide, :]

        # Update cell orientations ...
        # 
        # If orientation_conc is not None, then the daughter cells have 
        # orientations that are obtained by rotating the dividing cell's 
        # orientation counterclockwise by theta, where theta follows a 
        # von Mises distribution with mean 0 and concentration parameter
        # given by orientation_conc
        if orientation_conc is not None:
            dividing_orientations = cells[to_divide, 2:4]
            theta1 = rng.vonmises(0, orientation_conc, size=(n_divide,))
            for i, j in enumerate(idx_divide):
                rot = np.array(
                    [
                        [np.cos(theta1[i]), -np.sin(theta1[i])],
                        [np.sin(theta1[i]), np.cos(theta1[i])]
                    ],
                    dtype=np.float64
                )
                cells[j, 2:4] = np.dot(rot, dividing_orientations[i, :])
            theta2 = rng.vonmises(0, orientation_conc, size=(n_divide,))
            for i in range(n_divide):
                rot = np.array(
                    [
                        [np.cos(theta2[i]), -np.sin(theta2[i])],
                        [np.sin(theta2[i]), np.cos(theta2[i])]
                    ],
                    dtype=np.float64
                )
                new_cells[i, 2:4] = np.dot(rot, dividing_orientations[i, :])

        # Update cell lengths and positions ...
        #
        # If cell division is symmetric, then the daughter cells both have
        # length L / 2 - R, where L is the length of the dividing cell, 
        # and their x- and y-coordinates are perturbed by L / 4 + R / 2
        # along the daughter cells' orientation vectors
        #
        # If not, then the daughter cells have lengths M * (L - 2 * R) and 
        # (1 - M) * (L - 2 * R), where M is a normally distributed variable
        # with mean 0.5 and standard deviation given by daughter_length_std;
        # the corresponding perturbations of the cell centers are given by
        # R + M * (L - 2 * R) / 2 and R + (1 - M) * (L - 2 * R) / 2
        if daughter_length_std == 0:
            delta = cells[to_divide, 4] / 4 + R / 2
            cells[to_divide, 4] = cells[to_divide, 4] / 2 - R
            new_cells[:, 4] = new_cells[:, 4] / 2 - R
            cells[to_divide, 0] = cells[to_divide, 0] - delta * cells[to_divide, 2]
            new_cells[:, 0] = new_cells[:, 0] + delta * new_cells[:, 2]
            cells[to_divide, 1] = cells[to_divide, 1] - delta * cells[to_divide, 3]
            new_cells[:, 1] = new_cells[:, 1] + delta * new_cells[:, 3]
        else:
            # Sample a normally distributed daughter cell length for each
            # cell to be divided 
            M = rng.normal(0.5, daughter_length_std, size=(n_divide,))
            delta1 = R + M * (cells[to_divide, 4] - 2 * R) / 2
            delta2 = R + (1 - M) * (new_cells[:, 4] - 2 * R) / 2
            cells[to_divide, 4] = M * (cells[to_divide, 4] - 2 * R)
            new_cells[:, 4] = (1 - M) * (new_cells[:, 4] - 2 * R)
            cells[to_divide, 0] = cells[to_divide, 0] - delta1 * cells[to_divide, 2]
            new_cells[:, 0] = new_cells[:, 0] + delta2 * new_cells[:, 2]
            cells[to_divide, 1] = cells[to_divide, 1] - delta1 * cells[to_divide, 3]
            new_cells[:, 1] = new_cells[:, 1] + delta2 * new_cells[:, 3]

        # Update cell birth times 
        cells[to_divide, 5] = t
        new_cells[:, 5] = t

        # Update cell growth rates
        groups_idx = {group_ids[i]: i for i in range(n_groups)}
        for i, j in enumerate(idx_divide):
            gj = groups[j]
            cells[j, 6] = growth_dists[groups_idx[gj]](rng)
            new_cells[i, 6] = growth_dists[groups_idx[gj]](rng)

        # Append new cells to the population and return
        # 
        # Note that each daughter cell inherits its mother cell's viscosity
        # and friction coefficient
        cells = np.append(cells, new_cells, axis=0)

    return cells

