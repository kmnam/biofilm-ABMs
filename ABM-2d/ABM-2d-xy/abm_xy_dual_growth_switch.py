"""
Based on code originally written by JP Nijjer.

Authors:
    JP Nijjer, Kee-Myoung Nam

Last updated:
    10/15/2023
"""

import sys
import json
import numpy as np
from numba import njit
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from growth import (
    grow_cells,
    divide_cells,
    divide_cells_by_group,
    divide_max_length,
)
from mechanics import (
    get_cell_neighbors,
    step_RK_adaptive_from_neighbors
)
from plot import plot_simulation
from utils import write_cells

#######################################################################
# Butcher tableau for order 3(2) Runge-Kutta method by Bogacki and Shampine
A = np.array(
    [
        [0, 0, 0, 0],
        [1/2, 0, 0, 0],
        [0, 3/4, 0, 0],
        [2/9, 1/3, 4/9, 0]
    ],
    dtype=np.float64
)
b = np.array([2/9, 1/3, 4/9, 0], dtype=np.float64)
bs = np.array([7/24, 1/4, 1/3, 1/8], dtype=np.float64)
c = np.array([0, 1/2, 3/4, 1], dtype=np.float64)
error_order = 2

#######################################################################
# In what follows, a population of N cells is represented as a 2-D array of 
# size (N, 12), where each row represents a cell and stores the following data:
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
# 9) cell group identifier (1 for slow-growing, 2 for fast-growing)
# 10) total lifetime of cell in its current group
# 11) elapsed lifetime of cell in its current group
#######################################################################
def switch_groups(cells, to_switch, lifetime_dist1, lifetime_dist2, growth_dist1, 
                  growth_dist2, rng):
    """
    Switch the indicated cells from one group to the other.

    New lifetimes and growth rates are chosen using the given distribution
    functions, each of which must take the random number generator `rng` as
    its single input.

    Parameters
    ----------
    cells : `numpy.ndarray`
        Existing population of cells.
    to_switch : `numpy.ndarray`
        Boolean index indicating which cells are to switch from one group
        to the other. 
    lifetime_dist1 : function
        Lifetime distribution function for group 1. Must take 
        `numpy.random.Generator` as its single argument.
    lifetime_dist2 : function
        Lifetime distribution function for group 2. Must take
        `numpy.random.Generator` as its single argument.
    growth_dist1 : function
        Growth rate distribution function for group 1. Must take
        `numpy.random.Generator` as its single argument.
    growth_dist2 : function
        Growth rate distribution function for group 2. Must take
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

        # Sample new lifetimes and growth rates for each such cell 
        lifetimes_new_1 = np.array([lifetime_dist1(rng) for _ in idx_switch_from_2_to_1])
        lifetimes_new_2 = np.array([lifetime_dist2(rng) for _ in idx_switch_from_1_to_2])
        growth_new_1 = np.array([growth_dist1(rng) for _ in idx_switch_from_2_to_1])
        growth_new_2 = np.array([growth_dist2(rng) for _ in idx_switch_from_1_to_2])

        # Switch groups and assign new lifetimes and growth rates
        cells[to_switch_from_1_to_2, 9] = 2
        cells[to_switch_from_1_to_2, 6] = growth_new_2
        cells[to_switch_from_1_to_2, 10] = lifetimes_new_2
        cells[to_switch_from_1_to_2, 11] = 0
        cells[to_switch_from_2_to_1, 9] = 1
        cells[to_switch_from_2_to_1, 6] = growth_new_1
        cells[to_switch_from_2_to_1, 10] = lifetimes_new_1
        cells[to_switch_from_2_to_1, 11] = 0

    return cells

#######################################################################
if __name__ == '__main__':
    # Parse input json file
    with open(sys.argv[1]) as f:
        params = json.load(f)

    # Define required input parameters
    R = params['R']                         # Outer cell radius (including EPS)
    L0 = params['L0']                       # Initial cell length (excluding caps)
    Ldiv = 2 * L0 + 2 * R                   # Division length (excluding caps)
    growth_mean1 = params['growth_mean1']   # Mean growth rate of group 1
    growth_mean2 = params['growth_mean2']   # Mean growth rate of group 2
    lifetime_mean1 = params['lifetime_mean1']   # Mean lifetime of cells in group 1
    lifetime_mean2 = params['lifetime_mean2']   # Mean lifetime of cells in group 2
    E0 = params['E0']                       # Matrix contact stiffness
    sigma0 = params['sigma0']               # Adhesion energy density (force per unit length)
    eta_ambient = params['eta_ambient']     # Ambient viscosity with surrounding fluid
    eta_surface = params['eta_surface']     # Friction with surface
    dt = params['dt']                       # Timestep
    iter_write = params['iter_write']       # Write simulation data to file every so often
    iter_video = params['iter_video']       # Compile simulation data into output video every so often
    iter_update_stepsize = params['iter_update_stepsize']      # Update stepsize every so often
    iter_update_neighbors = params['iter_update_neighbors']    # Update neighbors every so often
    neighbor_threshold = 2 * (2 * R + L0)   # Radius for neighboring cells
    t_final = params['t_final']             # Final timepoint
    n_cells = params['n_cells']             # Desired number of cells

    # Surface contact area density
    surface_contact_density = (sigma0 * R * R / (4 * E0)) ** (1 / 3)

    # Define optional input parameters
    try:    # Inner cell radius (excluding EPS)
        Rcell = params['Rcell']
    except KeyError:
        Rcell = 0.8 * R
        params['Rcell'] = Rcell
    try:    # Standard deviation of slow growth rate
        growth_std1 = params['growth_std1']
    except KeyError:
        growth_std1 = 0.1 * growth_mean1
        params['growth_std1'] = growth_std1
    try:    # Standard deviation of fast growth rate
        growth_std2 = params['growth_std2']
    except KeyError:
        growth_std2 = 0.1 * growth_mean2
        params['growth_std2'] = growth_std2
    try:    # Inner cell (rigid) contact stiffness
        Ecell = params['Ecell']
    except KeyError:
        Ecell = E0
        params['Ecell'] = Ecell
    try:    # Scale of symmetry-breaking noise in equations of motion
        noise_scale = params['noise_scale']
    except KeyError:
        noise_scale = 1e-7 * E0 * R * R
        params['noise_scale'] = noise_scale
    try:    # Standard deviation for daughter lengths
        daughter_length_std = params['daughter_length_std']
    except KeyError:
        daughter_length_std = 0.0
        params['daughter_length_std'] = daughter_length_std
    try:    # Concentration parameter for daughter orientations
        orientation_conc = params['orientation_conc']
    except KeyError:
        orientation_conc = None
        params['orientation_conc'] = orientation_conc

    t = 0     # Current time
    i = 0     # Current iteration
    rng = np.random.default_rng(1234567890)

    # Growth rate distribution functions: normal distributions with given
    # mean and standard deviation
    growth_dist1 = lambda gen: gen.normal(growth_mean1, growth_std1)
    growth_dist2 = lambda gen: gen.normal(growth_mean2, growth_std2)

    # Group lifetime distribution functions: exponential distributions with
    # given mean lifetimes
    lifetime_dist1 = lambda gen: gen.exponential(scale=lifetime_mean1)
    lifetime_dist2 = lambda gen: gen.exponential(scale=lifetime_mean2)

    # Output file prefix
    prefix = sys.argv[2]

    # Define a founder cell at the origin at time zero, parallel to x-axis,
    # with mean growth rate and default viscosity and friction coefficients
    #
    # Additional entries include the following:
    # 9) Group identifier (1 or 2)
    # 10) Total lifetime of cell in given group
    # 11) Elapsed lifetime of cell in given group
    cells = np.array(
        [[
            0, 0, 1, 0, L0, 0, growth_mean1, eta_ambient, eta_surface,
            1, lifetime_dist1(rng), 0
        ]],
        dtype=np.float64
    )

    # Compute initial array of neighboring cells (should be empty)
    neighbors = get_cell_neighbors(cells, neighbor_threshold, R, Ldiv)

    # Write the founder cell to file
    paths = []
    path = '{}_init.txt'.format(prefix)
    write_cells(cells, path, params=params)
    paths.append(path)

    # Run the simulation ...
    while t < t_final and cells.shape[0] < n_cells:
        # Divide the cells that have reached division length
        to_divide = divide_max_length(cells, Ldiv)
        cells = divide_cells_by_group(
            cells, t, R, to_divide, [growth_dist1, growth_dist2], rng, [1, 2],
            daughter_length_std=daughter_length_std,
            orientation_conc=orientation_conc
        )

        # Update neighboring cells if division has occurred
        if to_divide.sum() > 0:
            neighbors = get_cell_neighbors(cells, neighbor_threshold, R, Ldiv)

        # Re-sample group lifetimes (and set elapsed lifetimes to zero)
        # for all daughter cells
        #
        # This is fine because the lifetimes are assumed to be exponentially
        # distributed and therefore memoryless
        if to_divide.sum() > 0:
            idx_daughters = np.concatenate((
                np.where(to_divide)[0], np.arange(to_divide.size, cells.shape[0])
            ))
            for j in idx_daughters:
                lifetime = lifetime_dist1(rng) if cells[j, 9] == 1 else lifetime_dist2(rng)
                cells[j, 10] = lifetime
                cells[j, 11] = 0
        
        # Update cell positions and orientations
        cells_new, errors = step_RK_adaptive_from_neighbors(
            A, b, bs, c, cells, neighbors, dt, R, Rcell, E0, Ecell,
            surface_contact_density, rng, noise_scale
        )

        # If the error is big, retry the step with a smaller stepsize (up to 
        # a given maximum number of iterations)
        if i % iter_update_stepsize == 0:
            max_error = np.max([np.abs(errors).max(), 1e-100])
            max_tries = 5
            j = 0
            while max_error > 1e-8 and j < max_tries:
                dt *= (1e-8 / max_error) ** (1 / (error_order + 1))
                cells_new, errors = step_RK_adaptive_from_neighbors(
                    A, b, bs, c, cells, neighbors, dt, R, Rcell, E0, Ecell,
                    surface_contact_density, rng, noise_scale
                )
                max_error = np.max([np.abs(errors).max(), 1e-100])
                j += 1
            # If the error is small, increase the stepsize up to a maximum stepsize
            if max_error < 1e-8:
                dt = np.min([dt * ((1e-8 / max_error) ** (1 / (error_order + 1))), 1e-4])
        cells = cells_new
        
        # Grow the cells
        cells = grow_cells(cells, dt, R)
        
        # Update current time
        t += dt
        i += 1

        # Update elapsed lifetimes of cells in each group and switch groups 
        # for all cells whose lifetimes have completely elapsed
        cells[:, 11] += dt
        to_switch = (cells[:, 10] < cells[:, 11])
        cells = switch_groups(
            cells, to_switch, lifetime_dist1, lifetime_dist2, growth_dist1,
            growth_dist2, rng
        )

        # Update neighboring cells 
        if i % iter_update_neighbors == 0:
            neighbors = get_cell_neighbors(cells, neighbor_threshold, R, Ldiv)
        
        # Write the current population to file 
        if i % iter_write == 0:
            print(
                'Iteration {}: {} cells, time = {:.10e}, max_error = {:.10e}, '
                'dt = {:.10e}'.format(i, cells.shape[0], t, np.abs(errors).max(), dt)
            )
            path = '{}_iter{}.txt'.format(prefix, i)
            params['t_curr'] = t
            write_cells(cells, path, params=params)
        if i % iter_video == 0:
            paths.append(path)

    # Write final population to file
    path = '{}_final.txt'.format(prefix)
    params['t_curr'] = t
    write_cells(cells, path, params=params)
    paths.append(path)

    # Generate video of simulation 
    plot_simulation(paths, '{}.avi'.format(prefix), R, fps=10)

