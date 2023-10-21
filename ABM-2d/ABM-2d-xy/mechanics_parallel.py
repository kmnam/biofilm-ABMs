"""
Parallelized implementations of cell-cell and cell-surface interaction forces. 

Authors:
    Kee-Myoung Nam

Last updated:
    10/20/2023
"""

import numpy as np
from numba import njit, prange
from mechanics import (
    cell_point_nearest_to_point,
    cell_cell_distance,
)

#######################################################################
# In what follows, a population of N cells is represented as a 2-D array of 
# size (N, 9+), where each row represents a cell and stores the following data:
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
#
# Additional features may be included in the array but these are not 
# relevant for the computations implemented here
#######################################################################
@njit(parallel=True, fastmath=True)
def composite_viscosity_force_prefactors_parallel(cells, R, surface_contact_density):
    """
    Compute the derivatives of the dissipation due to bulk viscosity and
    surface friction for each cell with respect to the cell's translational
    and orientational velocities, *divided by* the cell's translational and 
    orientational velocities.

    Namely, if drx, dry, dnx, dny are the translational and orientational 
    velocities of cell i, then the derivative of the total dissipation, P, 
    due to bulk viscosity and surface friction with respect to these velocities
    are given by 

    dP/d(drx) = K * drx
    dP/d(dry) = K * dry
    dP/d(dnx) = L * dnx
    dP/d(dny) = L * dny,

    where K and L are prefactors. This function returns these prefactors for 
    each cell in the given population.

    NOTE: This function is slower than the serial version. 

    Parameters
    ----------
    cells : `numpy.ndarray`
        Existing population of cells.
    R : float
        Cell radius.
    surface_contact_density : float
        Cell-surface contact area density.

    Returns
    -------
    The prefactors K and L defined above for each cell. 
    """
    KL = np.zeros((cells.shape[0], 2), dtype=np.float64)
    a = surface_contact_density / R
    for i in prange(cells.shape[0]):
        composite_drag = cells[i, 7] + cells[i, 8] * a
        KL[i, 0] = cells[i, 4] * composite_drag
        KL[i, 1] = (cells[i, 4] ** 3) * composite_drag / 12

    return KL

########################################################################
@njit(parallel=True, fastmath=True)
def get_cell_neighbors_parallel(cells, neighbor_threshold, R, Ldiv):
    """
    Get all pairs of cells whose distances are within the given threshold.

    This function should be used over the serial version when there are
    many cells (e.g., speedups are apparent for > 100 cells and 4 threads).  

    Parameters
    ----------
    cells : `numpy.ndarray`
        Existing population of cells.
    neighbor_threshold : float
        Threshold for distinguishing between neighboring and non-neighboring
        cells.
    R : float
        Cell radius.
    Ldiv : float
        Cell division length. 

    Returns
    -------
    Array of indices of pairs of neighboring cells. 
    """
    n = cells.shape[0]    # Number of cells

    # If there is only one cell, return an empty array
    if n == 1:
        return np.zeros((0, 6), dtype=np.float64)

    # Maintain arrays of neighboring cell data
    #
    # neighbors[k] == 1 if cells i and j are neighboring (and zero otherwise),
    # where k = i * (i - 1) / 2 + j
    #
    # If cells i and j are neighboring (neighbors[k] == 1):
    # - distances[k, :] is the distance vector between cells i and j
    # - sij[k, 0] is the cell-body coordinate of the contact point with 
    #   cell j along the centerline of cell i
    # - sij[k, 1] is the cell-body coordinate of the contact point with 
    #   cell i along the centerline of cell j
    ij = np.zeros((n * (n - 1) // 2, 2), dtype=np.int32)
    neighbors = np.zeros((n * (n - 1) // 2,), dtype=np.int32)
    distances = np.zeros((n * (n - 1) // 2, 2), dtype=np.float64)
    sij = np.zeros((n * (n - 1) // 2, 2), dtype=np.float64)

    # For each pair of cells in the population ...
    for i in prange(1, n):
        for j in range(i):    # Note that j < i
            # For two cells to be within neighbor_threshold of each other,
            # their centers must be within neighbor_threshold + Ldiv + 2 * R
            dist_rij = np.linalg.norm(cells[i, :2] - cells[j, :2])
            if dist_rij < neighbor_threshold + Ldiv + 2 * R:
                # In this case, compute their actual distance and check that
                # they lie within neighbor_threshold of each other 
                dist_ij, si, sj = cell_cell_distance(
                    cells[i, :2], cells[i, 2:4], cells[i, 4],
                    cells[j, :2], cells[j, 2:4], cells[j, 4]
                )
                if np.linalg.norm(dist_ij) < neighbor_threshold:
                    k = i * (i - 1) // 2 + j
                    neighbors[k] = 1
                    ij[k, 0] = i
                    ij[k, 1] = j
                    distances[k, :] = dist_ij
                    sij[k, 0] = si
                    sij[k, 1] = sj

    # Pick out neighboring pairs 
    are_neighbors = (neighbors != 0)
    ij = ij[are_neighbors, :]
    distances = distances[are_neighbors, :]
    sij = sij[are_neighbors, :]

    # Define output array ... 
    # 
    # Each row contains the following information about each pair of
    # neighboring cells:
    # 0) Index i of first cell in neighboring pair
    # 1) Index j of second cell in neighboring pair
    # 2) x-coordinate of distance vector from cell i to cell j
    # 3) y-coordinate of distance vector from cell i to cell j
    # 4) Cell-body coordinate of contact point along centerline of cell i
    # 5) Cell-body coordinate of contact point along centerline of cell j
    n_neighbors = are_neighbors.sum()
    outarr = np.zeros((n_neighbors, 6), dtype=np.float64)
    for k in prange(n_neighbors):
        outarr[k, :2] = ij[k, :]
        outarr[k, 2:4] = distances[k, :]
        outarr[k, 4:6] = sij[k, :]

    return outarr

########################################################################
@njit(parallel=True, fastmath=True)
def update_neighbor_distances_parallel(cells, neighbors):
    """
    Update the cell-cell distances in the given array of neighboring pairs
    of cells.

    Parameters
    ----------
    cells : `numpy.ndarray`
        Existing population of cells.
    neighbors : `numpy.ndarray`
        Array of neighboring pairs of cells.

    Returns
    -------
    Updated array of neighboring pairs of cells.
    """
    # Each row of neighbors contains the following information about
    # each pair of neighboring cells:
    # 0) Index i of first cell in neighboring pair
    # 1) Index j of second cell in neighboring pair
    # 2) x-coordinate of distance vector from cell i to cell j
    # 3) y-coordinate of distance vector from cell i to cell j
    # 4) Cell-body coordinate of contact point along centerline of cell i
    # 5) Cell-body coordinate of contact point along centerline of cell j
    #
    # Columns 2, 3, 4, 5 are updated here
    for k in prange(neighbors.shape[0]):
        i = np.int32(neighbors[k, 0])
        j = np.int32(neighbors[k, 1])
        dist_ij, si, sj = cell_cell_distance(
            cells[i, :2], cells[i, 2:4], cells[i, 4],
            cells[j, :2], cells[j, 2:4], cells[j, 4]
        )
        neighbors[k, 2:4] = dist_ij
        neighbors[k, 4] = si
        neighbors[k, 5] = sj

    return neighbors
 
########################################################################
@njit(parallel=True, fastmath=True)
def cell_cell_forces_parallel(cells, neighbor_threshold, R, Rcell, Ldiv, E0, Ecell):
    """
    Compute the derivatives of the cell-cell interaction energies for each 
    cell with respect to the cell's position and orientation coordinates.

    NOTE: The speed of this function relative to the serial version is
    undetermined. 

    Parameters
    ----------
    cells : `numpy.ndarray`
        Existing population of cells.
    neighbor_threshold : float
        Threshold for defining a pair of cells as neighbors.
    R : float
        Cell radius, including the EPS.
    Rcell : float
        Cell radius, excluding the EPS. 
    Ldiv : float
        Cell division length.
    E0 : float
        Elastic modulus of EPS.
    Ecell : float
        Elastic modulus of cell.

    Returns
    -------
    Derivatives of the cell-cell interaction energies with respect to cell 
    positions and orientations.
    """
    n = cells.shape[0]    # Number of cells

    # If there is only one cell, return zero
    if n == 1:
        return np.zeros((n, 4), dtype=np.float64)

    # Maintain array of partial derivatives of the interaction energies 
    # with respect to x-position, y-position, x-orientation, y-orientation
    dEdq = np.zeros((n, 4), dtype=np.float64)

    # Identify all pairs of neighboring cells
    neighbors = get_cell_neighbors(cells, neighbor_threshold, R, Ldiv)

    # Maintain array of forces to be computed for each pair of neighboring cells
    #
    # Columns 0 and 1 store the derivatives of the cell-cell interaction energy
    # due to the interaction between cells i and j w.r.t the position of cell i
    # (the derivatives of this energy w.r.t the position of cell j is simply 
    # the negatives of these derivatives)
    # 
    # Columns 2 and 3 store the derivatives of this energy w.r.t the orientation
    # of cell i
    #
    # Columns 4 and 5 store the derivatives of this energy w.r.t the orientation
    # of cell j
    forces = np.zeros((neighbors.shape[0], 6), dtype=np.float64)

    # For each pair of neighboring cells ...
    for k in prange(neighbors.shape[0]):
        dist_ij = neighbors[k, 2:4]    # Distance vector from i to j
        si = neighbors[k, 4]           # Cell-body coordinate along cell i
        sj = neighbors[k, 5]           # Cell-body coordinate along cell j
        delta_ij = np.linalg.norm(dist_ij)    # Magnitude of distance vector
        dir_ij = dist_ij / delta_ij           # Normalized distance vector from i to j
        
        # Get the overlapping distance between cells i and j (this distance
        # is negative if the cells are not overlapping)
        overlap = 2 * R - delta_ij

        # Define prefactors that determine the magnitudes of the interaction
        # forces, depending on the size of the overlap
        #
        # Case 1: the overlap is positive but less than R - Rcell (i.e., it
        # is limited to within the EPS coating)
        prefactor = 0
        if overlap > 0 and overlap < R - Rcell:
            prefactor = 2.5 * E0 * np.sqrt(R) * (overlap ** 1.5)
        # Case 2: the overlap is instead greater than R - Rcell (i.e., it
        # encroaches into the bodies of the two cells)
        elif overlap >= R - Rcell:
            prefactor1 = E0 * (R - Rcell) ** 1.5
            prefactor2 = Ecell * (overlap - R + Rcell) ** 1.5
            prefactor = 2.5 * np.sqrt(R) * (prefactor1 + prefactor2)

        if overlap > 0:
            # Derivative of cell-cell interaction energy w.r.t position of cell i
            vij = prefactor * dir_ij
            forces[k, :2] = vij
            # Derivative of cell-cell interaction energy w.r.t orientation of cell i
            forces[k, 2:4] = vij * si
            # Derivative of cell-cell interaction energy w.r.t orientation of cell j
            forces[k, 4:6] = -vij * sj

    # Compute net forces acting on each cell
    idx1 = neighbors[:, 0].astype(np.int32)
    idx2 = neighbors[:, 1].astype(np.int32)
    for i, j, k in zip(idx1, idx2, range(neighbors.shape[0])):   # Cannot be parallelized
        dEdq[i, :2] += forces[k, :2]
        dEdq[i, 2:4] += forces[k, 2:4]
        dEdq[j, :2] -= forces[k, :2]
        dEdq[j, 2:4] += forces[k, 4:6]

    return dEdq

########################################################################
@njit(parallel=True, fastmath=True)
def cell_cell_forces_from_neighbors_parallel(cells, neighbors, R, Rcell, E0, Ecell):
    """
    Compute the derivatives of the cell-cell interaction energies for each 
    cell with respect to the cell's position and orientation coordinates.

    In this function, the pairs of neighboring cells in the population have
    been pre-computed.
    
    NOTE: This function is slower than the serial version. 

    Parameters
    ----------
    cells : `numpy.ndarray`
        Existing population of cells.
    neighbors : `numpy.ndarray`
        Array specifying pairs of neighboring cells in the population. 
    R : float
        Cell radius, including the EPS.
    Rcell : float
        Cell radius, excluding the EPS.
    E0 : float
        Elastic modulus of EPS.
    Ecell : float
        Elastic modulus of cell.

    Returns
    -------
    Derivatives of the cell-cell interaction energies with respect to cell 
    positions and orientations.
    """
    n = cells.shape[0]    # Number of cells

    # If there is only one cell, return zero
    if n == 1:
        return np.zeros((n, 4), dtype=np.float64)

    # Maintain array of partial derivatives of the interaction energies 
    # with respect to x-position, y-position, x-orientation, y-orientation
    dEdq = np.zeros((n, 4), dtype=np.float64)

    # Maintain array of forces to be computed for each pair of neighboring cells
    #
    # Columns 0 and 1 store the derivatives of the cell-cell interaction energy
    # due to the interaction between cells i and j w.r.t the position of cell i
    # (the derivatives of this energy w.r.t the position of cell j is simply 
    # the negatives of these derivatives)
    # 
    # Columns 2 and 3 store the derivatives of this energy w.r.t the orientation
    # of cell i
    #
    # Columns 4 and 5 store the derivatives of this energy w.r.t the orientation
    # of cell j
    forces = np.zeros((neighbors.shape[0], 6), dtype=np.float64)

    # For each pair of neighboring cells ...
    for k in prange(neighbors.shape[0]):
        dist_ij = neighbors[k, 2:4]    # Distance vector from i to j
        si = neighbors[k, 4]           # Cell-body coordinate along cell i
        sj = neighbors[k, 5]           # Cell-body coordinate along cell j
        delta_ij = np.linalg.norm(dist_ij)    # Magnitude of distance vector
        dir_ij = dist_ij / delta_ij           # Normalized distance vector from i to j
        
        # Get the overlapping distance between cells i and j (this distance
        # is negative if the cells are not overlapping)
        overlap = 2 * R - delta_ij

        # Define prefactors that determine the magnitudes of the interaction
        # forces, depending on the size of the overlap
        #
        # Case 1: the overlap is positive but less than R - Rcell (i.e., it
        # is limited to within the EPS coating)
        prefactor = 0
        if overlap > 0 and overlap < R - Rcell:
            prefactor = 2.5 * E0 * np.sqrt(R) * (overlap ** 1.5)
        # Case 2: the overlap is instead greater than R - Rcell (i.e., it
        # encroaches into the bodies of the two cells)
        elif overlap >= R - Rcell:
            prefactor1 = E0 * (R - Rcell) ** 1.5
            prefactor2 = Ecell * (overlap - R + Rcell) ** 1.5
            prefactor = 2.5 * np.sqrt(R) * (prefactor1 + prefactor2)

        if overlap > 0:
            # Derivative of cell-cell interaction energy w.r.t position of cell i
            vij = prefactor * dir_ij
            forces[k, :2] = vij
            # Derivative of cell-cell interaction energy w.r.t orientation of cell i
            forces[k, 2:4] = vij * si
            # Derivative of cell-cell interaction energy w.r.t orientation of cell j
            forces[k, 4:6] = -vij * sj

    # Compute net forces acting on each cell
    idx1 = neighbors[:, 0].astype(np.int32)
    idx2 = neighbors[:, 1].astype(np.int32)
    for i, j, k in zip(idx1, idx2, range(neighbors.shape[0])):
        dEdq[i, :2] += forces[k, :2]
        dEdq[i, 2:4] += forces[k, 2:4]
        dEdq[j, :2] -= forces[k, :2]
        dEdq[j, 2:4] += forces[k, 4:6]

    return dEdq

