"""
Parallelized implementations of cell-cell and cell-surface interaction forces. 

Authors:
    Kee-Myoung Nam

Last updated:
    10/14/2023
"""

import numpy as np
from numba import njit, prange

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
@njit(parallel=True)
def composite_viscosity_force_prefactors(cells, R, surface_contact_density):
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
    for i in prange(cells.shape[0]):
        composite_drag = cells[i, 7] + cells[i, 8] * surface_contact_density / R
        KL[i, 0] = cells[i, 4] * composite_drag
        KL[i, 1] = (cells[i, 4] ** 3) * composite_drag / 12

    return KL

########################################################################
@njit
def cell_point_nearest_to_point(r, n, l, q):
    """
    Compute the point along the centerline of the given cell to the point q.

    Parameters
    ----------
    r : `numpy.ndarray`
        x- and y-coordinates of cell position vector.
    n : `numpy.ndarray`
        x- and y-coordinates of cell orientation vector.
    l : float
        Cell length.
    q : `numpy.ndarray`
        Input point.

    Returns
    -------
    Cell-body coordinate corresponding to nearest point along the cell. 
    """
    s = (-np.dot(r, n) + np.dot(q, n)) / np.dot(n, n)
    if np.abs(s) <= l / 2:
        return s
    elif s > l / 2:
        return l / 2
    else:    # s < -l / 2
        return -l / 2

########################################################################
@njit
def cell_cell_distance(r1, n1, l1, r2, n2, l2):
    """
    Compute the distance vector from cell 1 to cell 2.

    This is an implementation of Vega and Lago's algorithm for computing 
    the distance between rods (Vega & Lago, Comput Chem, 1994).

    Parameters
    ----------
    r1 : `numpy.ndarray`
        x- and y-coordinates of position vector of cell 1. 
    n1 : `numpy.ndarray`
        x- and y-coordinates of orientation vector of cell 1. 
    l1 : float
        Length of cell 1 (excluding caps).
    r2 : `numpy.ndarray`
        x- and y-coordinates of position vector of cell 2.
    n2 : `numpy.ndarray`
        x- and y-coordinates of orientation vector of cell 2.
    l2 : float
        Length of cell 2 (excluding caps).

    Returns
    -------
    Distance vector between the centerlines of cell 1 and cell 2, together
    with the cell-body coordinates of the contact points along the centerlines
    of the two cells. 
    """
    half_l1 = l1 / 2
    half_l2 = l2 / 2
    r12 = r2 - r1              # Vector running from r1 to r2

    # We are looking for the values of s in [-l1/2, l1/2] and t in [-l2/2, l2/2]
    # such that the norm of r12 + t*n2 - s*n1 is minimized
    s_numer = np.dot(r12, n1) - np.dot(n1, n2) * np.dot(r12, n2)
    t_numer = np.dot(n1, n2) * np.dot(r12, n1) - np.dot(r12, n2)
    denom = 1 - np.dot(n1, n2) ** 2

    # If the two centerlines are not parallel ...
    if np.abs(denom) > 1e-6:
        s = s_numer / denom
        t = t_numer / denom
        # Check that the unconstrained minimum values of s and t lie within 
        # the desired ranges
        if np.abs(s) > half_l1 or np.abs(t) > half_l2:
            # If not, find the side of the square [-l1/2, l1/2] by [-l2/2, l2/2]
            # in which the unconstrained minimum values is nearest
            # 
            # Region 1 (above top side): between t = s + X and t = -s + X
            # Region 2 (right of right side): between t = s + X and t = -s + Y
            # Region 3 (below bottom side): between t = -s + Y and t = s + Y
            # Region 4 (left or left side): between t = s + Y and t = -s + X,
            # where X = (l2 - l1) / 2 and Y = (l1 - l2) / 2
            X = half_l2 - half_l1
            Y = -X
            if t >= s + X and t >= -s + X:    # In region 1
                # In this case, set t = l2 / 2 and find s
                q = r2 + half_l2 * n2
                s = cell_point_nearest_to_point(r1, n1, l1, q)
                t = half_l2
            elif t < s + X and t >= -s + Y:   # In region 2
                # In this case, set s = l1 / 2 and find t
                q = r1 + half_l1 * n1
                t = cell_point_nearest_to_point(r2, n2, l2, q)
                s = half_l1
            elif t < -s + Y and t < s + Y:    # In region 3
                # In this case, set t = -l2 / 2 and find s
                q = r2 - half_l2 * n2
                s = cell_point_nearest_to_point(r1, n1, l1, q)
                t = -half_l2
            else:    # t >= s + Y and t < s + X, in region 4
                # In this case, set s = -l1 / 2 and find t
                q = r1 - half_l1 * n1
                t = cell_point_nearest_to_point(r2, n2, l2, q)
                s = -half_l1
        # Compute distance vector from cell 1 to cell 2
        dist_12 = r12 + t * n2 - s * n1
    # Otherwise, take cap centers of cell 1 and compare their distances to cell 2
    #
    # TODO Is choosing either cap center of cell 1 a suitable choice for determining
    # the cell-cell interaction force?
    else:
        p1 = r1 - half_l1 * n1    # Endpoint for s = -l1 / 2
        q1 = r1 + half_l1 * n1    # Endpoint for s = l1 / 2
        t_p1 = cell_point_nearest_to_point(r2, n2, l2, p1)
        t_q1 = cell_point_nearest_to_point(r2, n2, l2, q1)
        dist_to_p1 = p1 - (r2 + t_p1 * n2)    # Vector running towards p1
        dist_to_q1 = q1 - (r2 + t_q1 * n2)    # Vector running towards q1
        if np.linalg.norm(dist_to_p1) < np.linalg.norm(dist_to_q1):   # Here, s = -l1 / 2
            dist_12 = -dist_to_p1
            s = -half_l1
            t = t_p1
        else:                                                         # Here, s = l1 / 2
            dist_12 = -dist_to_q1
            s = half_l1
            t = t_q1

    return dist_12, s, t

########################################################################
@njit(parallel=True)
def get_cell_neighbors(cells, neighbor_threshold, R, Ldiv):
    """
    Get all pairs of cells whose distances are within the given threshold.

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
    # neighbors[i, j] == 1 if cells i and j are neighboring (and zero 
    # otherwise)
    #
    # If cells i and j are neighboring (neighbors[i, j] == 1):
    # - distances[i, j, :] is the distance vector between cells i and j
    # - sij[i, j, 0] is the cell-body coordinate of the contact point with 
    #   cell j along the centerline of cell i
    # - sij[i, j, 1] is the cell-body coordinate of the contact point with 
    #   cell i along the centerline of cell j
    neighbors = np.zeros((n, n), dtype=np.int32)
    distances = np.zeros((n, n, 2), dtype=np.float64)
    sij = np.zeros((n, n, 2), dtype=np.float64)

    # For each pair of cells in the population ...
    for i in prange(n):
        for j in range(i+1, n):
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
                    neighbors[i, j] = 1
                    distances[i, j, :] = dist_ij
                    sij[i, j, 0] = si
                    sij[i, j, 1] = sj

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
    outarr = np.zeros((neighbors.sum(), 6), dtype=np.float64)
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            if neighbors[i, j] == 1:
                outarr[idx, 0] = i
                outarr[idx, 1] = j
                outarr[idx, 2:4] = distances[i, j, :]
                outarr[idx, 4:6] = sij[i, j, :]
                idx += 1

    return outarr
 
########################################################################
@njit(parallel=True)
def cell_cell_forces(cells, neighbor_threshold, R, Rcell, Ldiv, E0, Ecell):
    """
    Compute the derivatives of the cell-cell interaction energies for each 
    cell with respect to the cell's position and orientation coordinates.

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
        elif overlap > R - Rcell:
            prefactor1 = E0 * (R - Rcell) ** 1.5
            prefactor2 = Ecell * (overlap - R + Rcell) ** 1.5
            prefactor = 2.5 * np.sqrt(R) * (prefactor1 + prefactor2)

        # Derivative of cell-cell interaction energy w.r.t position of cell i
        forces[k, :2] = prefactor * dir_ij
        # Derivative of cell-cell interaction energy w.r.t orientation of cell i
        forces[k, 2:4] = prefactor * dir_ij * si
        # Derivative of cell-cell interaction energy w.r.t orientation of cell j
        forces[k, 4:6] = -prefactor * dir_ij * sj

    # Compute net forces acting on each cell
    for k in range(neighbors.shape[0]):
        i = np.int32(neighbors[k, 0])
        j = np.int32(neighbors[k, 1])
        dEdq[i, :2] += forces[k, :2]
        dEdq[i, 2:4] += forces[k, 2:4]
        dEdq[j, :2] -= forces[k, :2]
        dEdq[j, 2:4] += forces[k, 4:6]

    return dEdq

########################################################################
@njit(parallel=True)
def cell_cell_forces_from_neighbors(cells, neighbors, R, Rcell, E0, Ecell):
    """
    Compute the derivatives of the cell-cell interaction energies for each 
    cell with respect to the cell's position and orientation coordinates.

    In this function, the pairs of neighboring cells in the population have
    been pre-computed. 

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
        elif overlap > R - Rcell:
            prefactor1 = E0 * (R - Rcell) ** 1.5
            prefactor2 = Ecell * (overlap - R + Rcell) ** 1.5
            prefactor = 2.5 * np.sqrt(R) * (prefactor1 + prefactor2)

        # Derivative of cell-cell interaction energy w.r.t position of cell i
        forces[k, :2] = prefactor * dir_ij
        # Derivative of cell-cell interaction energy w.r.t orientation of cell i
        forces[k, 2:4] = prefactor * dir_ij * si
        # Derivative of cell-cell interaction energy w.r.t orientation of cell j
        forces[k, 4:6] = -prefactor * dir_ij * sj

    # Compute net forces acting on each cell
    for k in range(neighbors.shape[0]):
        i = np.int32(neighbors[k, 0])
        j = np.int32(neighbors[k, 1])
        dEdq[i, :2] += forces[k, :2]
        dEdq[i, 2:4] += forces[k, 2:4]
        dEdq[j, :2] -= forces[k, :2]
        dEdq[j, 2:4] += forces[k, 4:6]

    return dEdq

########################################################################
@njit(parallel=True)
def get_velocities(cells, neighbor_threshold, R, Rcell, Ldiv, E0, Ecell,
                   surface_contact_density):
    """
    Given the current positions, orientations, lengths, viscosity coefficients,
    and surface friction coefficients for the given population of cells, compute
    their translational and orientational velocities.

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
    surface_contact_density : float
        Cell-surface contact area density.

    Returns
    -------
    Array of translational and orientational velocities. 
    """
    # For each cell, the relevant Lagrangian mechanics are given by 
    # 
    # dP/d(dq) = -dE/dq + lambda * d(nx^2 + ny^2 - 1)/dq, 
    #
    # where:
    # - P is the total dissipation due to bulk viscosity and surface friction,
    # - E is the total cell-cell interaction energy involving the given cell,
    # - q is a generalized coordinate (x-position, y-position, x-orientation, 
    #   y-orientation),
    # - nx and ny are the x-orientation and y-orientation, respectively,
    # - dq is the corresponding velocity, and 
    # - lambda is a Lagrange multiplier. 
    #
    # Note that dP/d(dq) = K * dq for some prefactor K, which is given in 
    # `composite_viscosity_force_prefactors()`. Moreover, the constraint 
    # 
    # nx^2 + ny^2 - 1 == 0
    # 
    # implies the constraint 
    #
    # 2 * nx * dnx + 2 * ny * dny == 0
    #
    # where dnx and dny are the orientational velocities. This yields the 
    # following value of the Lagrange multiplier:
    #
    # lambda = -0.5 * (nx * dE/dnx + ny * dE/dny)
    # 
    n = cells.shape[0]
    velocities = np.zeros((n, 4), dtype=np.float64)
    prefactors = composite_viscosity_force_prefactors(cells, R, surface_contact_density)
    dEdq = cell_cell_forces(cells, neighbor_threshold, R, Rcell, Ldiv, E0, Ecell)
    for i in prange(n):
        mult = cells[i, 2] * dEdq[i, 2] + cells[i, 3] * dEdq[i, 3]
        velocities[i, :2] = -dEdq[i, :2] / prefactors[i, 0]
        velocities[i, 2:4] = (-dEdq[i, 2:4] - mult * cells[i, 2:4]) / prefactors[i, 1]

    return velocities

########################################################################
@njit(parallel=True)
def get_velocities_from_neighbors(cells, neighbors, R, Rcell, E0, Ecell, 
                                  surface_contact_density):
    """
    Given the current positions, orientations, lengths, viscosity coefficients,
    and surface friction coefficients for the given population of cells, compute
    their translational and orientational velocities.

    In this function, the pairs of neighboring cells in the population have
    been pre-computed. 

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
    surface_contact_density : float
        Cell-surface contact area density.

    Returns
    -------
    Array of translational and orientational velocities. 
    """
    # For each cell, the relevant Lagrangian mechanics are given by 
    # 
    # dP/d(dq) = -dE/dq + lambda * d(nx^2 + ny^2 - 1)/dq, 
    #
    # where:
    # - P is the total dissipation due to bulk viscosity and surface friction,
    # - E is the total cell-cell interaction energy involving the given cell,
    # - q is a generalized coordinate (x-position, y-position, x-orientation, 
    #   y-orientation),
    # - nx and ny are the x-orientation and y-orientation, respectively,
    # - dq is the corresponding velocity, and 
    # - lambda is a Lagrange multiplier. 
    #
    # Note that dP/d(dq) = K * dq for some prefactor K, which is given in 
    # `composite_viscosity_force_prefactors()`. Moreover, the constraint 
    # 
    # nx^2 + ny^2 - 1 == 0
    # 
    # implies the constraint 
    #
    # 2 * nx * dnx + 2 * ny * dny == 0
    #
    # where dnx and dny are the orientational velocities. This yields the 
    # following value of the Lagrange multiplier:
    #
    # lambda = -0.5 * (nx * dE/dnx + ny * dE/dny)
    # 
    n = cells.shape[0]
    velocities = np.zeros((n, 4), dtype=np.float64)
    prefactors = composite_viscosity_force_prefactors(cells, R, surface_contact_density)
    dEdq = cell_cell_forces_from_neighbors(cells, neighbors, R, Rcell, E0, Ecell)
    for i in prange(n):
        mult = cells[i, 2] * dEdq[i, 2] + cells[i, 3] * dEdq[i, 3]
        velocities[i, :2] = -dEdq[i, :2] / prefactors[i, 0]
        velocities[i, 2:4] = (-dEdq[i, 2:4] - mult * cells[i, 2:4]) / prefactors[i, 1]

    return velocities

########################################################################
@njit
def normalize_orientations(cells):
    """
    Normalize the orientation vectors of all cells in the given population.

    Parameters
    ----------
    cells : `numpy.ndarray`
        Existing population of cells.

    Returns
    -------
    Updated population of cells.
    """
    # Note that np.linalg.norm(..., axis=1) is not supported in numba
    norms = np.sqrt(np.power(cells[:, 2], 2) + np.power(cells[:, 3], 2))
    cells[:, 2:4] /= norms.reshape((-1, 1))
    
    return cells

########################################################################
@njit
def step_RK(A, b, c, cells, neighbor_threshold, dt, R, Rcell, Ldiv, E0, Ecell, 
            surface_contact_density, rng, noise_scale):
    """
    Run one step of an explicit Runge-Kutta method with the given Butcher
    tableau for the given timestep.

    Parameters
    ----------
    A : `numpy.ndarray`
        Runge-Kutta matrix of Butcher tableau. Should be lower triangular 
        with zero diagonal. 
    b : `numpy.ndarray`
        Weights of Butcher tableau. Entries should sum to one.
    c : `numpy.ndarray`
        Nodes of Butcher tableau. First entry should be zero. 
    cells : `numpy.ndarray`
        Existing population of cells.
    neighbor_threshold : float
        Threshold for defining a pair of cells as neighbors.
    dt : float
        Timestep.
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
    surface_contact_density : float
        Cell-surface contact area density.
    rng : `numpy.random.Generator`
        Random number generator (for symmetry-breaking noise).
    noise_scale : float
        Scale for symmetry-breaking noise (uniformly sampled from the range 
        `[noise_scale, noise_scale]`).

    Returns
    -------
    Updated population of cells.
    """
    # Compute velocities at given partial timesteps 
    s = b.size
    velocities = np.zeros((cells.shape[0], 4, s), dtype=np.float64)
    velocities[:, :, 0] = get_velocities(
        cells, neighbor_threshold, R, Rcell, Ldiv, E0, Ecell, surface_contact_density
    )
    for i in range(1, s):
        multipliers = np.array((cells.shape[0], 4), dtype=np.float64)
        for j in range(i):
            multipliers += A[i, j] * velocities[:, :, j]
        cells_i = cells.copy()
        cells_i[:, :4] += multipliers * dt
        velocities[:, :, i] = get_velocities(
            cells_i, neighbor_threshold, R, Rcell, Ldiv, E0, Ecell,
            surface_contact_density
        )

    # Compute Runge-Kutta update from computed velocities
    velocities_final = np.zeros((cells.shape[0], 4), dtype=np.float64)
    for i in range(s):
        velocities_final += velocities[:, :, i] * b[i]
    cells[:, :4] += velocities_final * dt

    # Add symmetry-breaking noise to orientations
    #cells[:, 2:4] += rng.uniform(-noise_scale, noise_scale, (cells.shape[0], 2)) * dt

    # Renormalize orientations
    cells = normalize_orientations(cells)

    return cells

########################################################################
@njit
def step_RK_adaptive(A, b, bs, c, cells, neighbor_threshold, dt, R, Rcell, Ldiv,
                     E0, Ecell, surface_contact_density, rng, noise_scale):
    """
    Run one step of an explicit Runge-Kutta method with the given Butcher
    tableau for the given timestep.

    Parameters
    ----------
    A : `numpy.ndarray`
        Runge-Kutta matrix of Butcher tableau. Should be lower triangular 
        with zero diagonal. 
    b : `numpy.ndarray`
        Weights of Butcher tableau. Entries should sum to one.
    bs : `numpy.ndarray`
        Error weights of Butcher tableau. Entries should sum to one.
    c : `numpy.ndarray`
        Nodes of Butcher tableau. First entry should be zero. 
    cells : `numpy.ndarray`
        Existing population of cells.
    neighbor_threshold : float
        Threshold for defining a pair of cells as neighbors.
    dt : float
        Timestep.
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
    surface_contact_density : float
        Cell-surface contact area density.
    rng : `numpy.random.Generator`
        Random number generator (for symmetry-breaking noise).
    noise_scale : float
        Scale for symmetry-breaking noise (uniformly sampled from the range 
        `[noise_scale, noise_scale]`).

    Returns
    -------
    Updated population of cells.
    """
    # Compute velocities at given partial timesteps 
    s = b.size
    velocities = np.zeros((cells.shape[0], 4, s), dtype=np.float64)
    velocities[:, :, 0] = get_velocities(
        cells, neighbor_threshold, R, Rcell, Ldiv, E0, Ecell, surface_contact_density
    )
    for i in range(1, s):
        multipliers = np.array((cells.shape[0], 4), dtype=np.float64)
        for j in range(i):
            multipliers += A[i, j] * velocities[:, :, j]
        cells_i = cells.copy()
        cells_i[:, :4] += multipliers * dt
        velocities[:, :, i] = get_velocities(
            cells_i, neighbor_threshold, R, Rcell, Ldiv, E0, Ecell,
            surface_contact_density
        )

    # Compute Runge-Kutta update from computed velocities
    velocities_final1 = np.zeros((cells.shape[0], 4), dtype=np.float64)
    velocities_final2 = np.zeros((cells.shape[0], 4), dtype=np.float64)
    for i in range(s):
        velocities_final1 += velocities[:, :, i] * b[i]
        velocities_final2 += velocities[:, :, i] * bs[i]
    delta1 = velocities_final1 * dt
    delta2 = velocities_final2 * dt
    cells[:, :4] += delta1
    errors = delta1 - delta2

    # Add symmetry-breaking noise to orientations
    #cells[:, 2:4] += rng.uniform(-noise_scale, noise_scale, (cells.shape[0], 2)) * dt

    # Renormalize orientations
    cells = normalize_orientations(cells)

    return cells

########################################################################
@njit
def step_RK_from_neighbors(A, b, c, cells, neighbors, dt, R, Rcell, E0, Ecell, 
                           surface_contact_density, rng, noise_scale):
    """
    Run one step of an explicit Runge-Kutta method with the given Butcher
    tableau for the given timestep.

    In this function, the pairs of neighboring cells in the population have
    been pre-computed. 

    Parameters
    ----------
    A : `numpy.ndarray`
        Runge-Kutta matrix of Butcher tableau. Should be lower triangular 
        with zero diagonal. 
    b : `numpy.ndarray`
        Weights of Butcher tableau. Entries should sum to one.
    c : `numpy.ndarray`
        Nodes of Butcher tableau. First entry should be zero. 
    cells : `numpy.ndarray`
        Existing population of cells.
    neighbors : `numpy.ndarray`
        Array specifying pairs of neighboring cells in the population. 
    dt : float
        Timestep.
    R : float
        Cell radius, including the EPS.
    Rcell : float
        Cell radius, excluding the EPS.
    E0 : float
        Elastic modulus of EPS.
    Ecell : float
        Elastic modulus of cell.
    surface_contact_density : float
        Cell-surface contact area density.
    rng : `numpy.random.Generator`
        Random number generator (for symmetry-breaking noise).
    noise_scale : float
        Scale for symmetry-breaking noise (uniformly sampled from the range 
        `[noise_scale, noise_scale]`).

    Returns
    -------
    Updated population of cells.
    """
    # Compute velocities at given partial timesteps 
    s = b.size
    velocities = np.zeros((cells.shape[0], 4, s), dtype=np.float64)
    velocities[:, :, 0] = get_velocities_from_neighbors(
        cells, neighbors, R, Rcell, E0, Ecell, surface_contact_density
    )
    for i in range(1, s):
        multipliers = np.zeros((cells.shape[0], 4), dtype=np.float64)
        for j in range(i):
            multipliers += A[i, j] * velocities[:, :, j]
        cells_i = cells.copy()
        cells_i[:, :4] += multipliers * dt
        velocities[:, :, i] = get_velocities_from_neighbors(
            cells_i, neighbors, R, Rcell, E0, Ecell, surface_contact_density
        )

    # Compute Runge-Kutta update from computed velocities
    velocities_final = np.zeros((cells.shape[0], 4), dtype=np.float64)
    for i in range(s):
        velocities_final += velocities[:, :, i] * b[i]
    cells[:, :4] += velocities_final * dt

    # Add symmetry-breaking noise to orientations
    #cells[:, 2:4] += rng.uniform(-noise_scale, noise_scale, (cells.shape[0], 2)) * dt

    # Renormalize orientations
    cells = normalize_orientations(cells)

    return cells

########################################################################
@njit
def step_RK_adaptive_from_neighbors(A, b, bs, c, cells, neighbors, dt, R, Rcell,
                                    E0, Ecell, surface_contact_density, rng,
                                    noise_scale):
    """
    Run one step of an adaptive Runge-Kutta method with the given Butcher
    tableau for the given timestep.

    In this function, the pairs of neighboring cells in the population have
    been pre-computed. 

    Parameters
    ----------
    A : `numpy.ndarray`
        Runge-Kutta matrix of Butcher tableau. Should be lower triangular 
        with zero diagonal. 
    b : `numpy.ndarray`
        Weights of Butcher tableau. Entries should sum to one.
    bs : `numpy.ndarray`
        Error weights of Butcher tableau. Entries should sum to one. 
    c : `numpy.ndarray`
        Nodes of Butcher tableau. First entry should be zero. 
    cells : `numpy.ndarray`
        Existing population of cells.
    neighbors : `numpy.ndarray`
        Array specifying pairs of neighboring cells in the population. 
    dt : float
        Timestep.
    R : float
        Cell radius, including the EPS.
    Rcell : float
        Cell radius, excluding the EPS.
    E0 : float
        Elastic modulus of EPS.
    Ecell : float
        Elastic modulus of cell.
    surface_contact_density : float
        Cell-surface contact area density.
    rng : `numpy.random.Generator`
        Random number generator (for symmetry-breaking noise).
    noise_scale : float
        Scale for symmetry-breaking noise (uniformly sampled from the range 
        `[noise_scale, noise_scale]`).

    Returns
    -------
    Updated population of cells.
    """
    # Compute velocities at given partial timesteps 
    s = b.size
    velocities = np.zeros((cells.shape[0], 4, s), dtype=np.float64)
    velocities[:, :, 0] = get_velocities_from_neighbors(
        cells, neighbors, R, Rcell, E0, Ecell, surface_contact_density
    )
    for i in range(1, s):
        multipliers = np.zeros((cells.shape[0], 4), dtype=np.float64)
        for j in range(i):
            multipliers += A[i, j] * velocities[:, :, j]
        cells_i = cells.copy()
        cells_i[:, :4] += multipliers * dt
        velocities[:, :, i] = get_velocities_from_neighbors(
            cells_i, neighbors, R, Rcell, E0, Ecell, surface_contact_density
        )

    # Compute Runge-Kutta update from computed velocities
    velocities_final1 = np.zeros((cells.shape[0], 4), dtype=np.float64)
    velocities_final2 = np.zeros((cells.shape[0], 4), dtype=np.float64)
    for i in range(s):
        velocities_final1 += velocities[:, :, i] * b[i]
        velocities_final2 += velocities[:, :, i] * bs[i]
    delta1 = velocities_final1 * dt
    delta2 = velocities_final2 * dt
    cells[:, :4] += delta1
    errors = delta1 - delta2

    # Add symmetry-breaking noise to orientations
    #cells[:, 2:4] += rng.uniform(-noise_scale, noise_scale, (cells.shape[0], 2)) * dt

    # Renormalize orientations
    cells = normalize_orientations(cells)

    return cells, errors

