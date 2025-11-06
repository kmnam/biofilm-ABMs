"""
Miscellaneous functions.

Authors:
    Kee-Myoung Nam

Last updated:
    10/22/2025
"""
import os
import glob
import re
import numpy as np
import gudhi

#######################################################################
_colidx_id = 0
_colidx_rx = 1
_colidx_ry = 2
_colidx_rz = 3
_colidx_nx = 4
_colidx_ny = 5
_colidx_nz = 6
_colidx_drx = 7
_colidx_dry = 8
_colidx_drz = 9
_colidx_dnx = 10
_colidx_dny = 11
_colidx_dnz = 12
_colidx_l = 13
_colidx_half_l = 14
_colidx_t0 = 15
_colidx_growth = 16
_colidx_eta0 = 17
_colidx_eta1 = 18
_colidx_sigma0 = 19
_colidx_group = 20
_ncols_required = 21

#######################################################################
def read_cells(path):
    """
    Read the cells in the given file, together with the simulation parameters.

    Parameters
    ----------
    path : str 
        Input file path.

    Returns
    -------
    Population of cells, together with a dict containing the simulation 
    parameters. 
    """
    # First read the parameters
    params = {}
    with open(path) as f:
        for line in f:
            if line.startswith('#'):
                m = re.match(r'# ([A-Za-z0-9_]+) = ([0-9eE+-\.]+)', line)
                try:
                    params[m.group(1)] = float(m.group(2))
                except AttributeError:
                    print(
                        '[WARN] Skipping comment with unexpected format:',
                        line.strip()
                    )

    # Then read the cells that are stored in the file 
    cells = np.loadtxt(path, comments='#', delimiter='\t', skiprows=0)
    if len(cells.shape) == 1:
        cells = cells.reshape((1, -1))

    return cells, params

#######################################################################
def read_simplicial_complex(filename):
    """
    Read the given file, specifying a pre-computed simplicial complex. 

    This function outputs four arrays: the point coordinates, the edges, 
    the triangles, and the tetrahedra. 
    """
    tree = gudhi.SimplexTree()
    with open(filename) as f:
        for line in f:
            if line.startswith('VERTEX'):
                tree.insert([int(line.split()[1])])
            elif line.startswith('EDGE'):
                _, v, w, _ = line.split()
                tree.insert([int(v), int(w)])
            elif line.startswith('TRIANGLE'):
                tree.insert([int(x) for x in line.split()[1:]])
            elif line.startswith('TETRAHEDRON'):
                tree.insert([int(x) for x in line.split()[1:]])
   
    return tree

#######################################################################
def parse_dir(paths):
    """
    Get the files stored in the given directory and sort them by 
    iteration.

    Parameters
    ----------
    path : str
        Path to input directory. 

    Returns
    -------
    List of files in order of iteration. 
    """
    filenames = [
        filename for filename in glob.glob(paths) if filename.endswith('.txt')
    ]
    filenames_sorted = []

    # Find the initial file 
    for filename in filenames:
        if filename.endswith('init.txt'):
            filenames_sorted.append(filename)
            break

    # Run through all intermediate files and sort them in order of iteration
    filenames_iter = []
    idx = []
    for filename in filenames:
        m = re.search(r'iter(\d+)\.txt$', filename)
        if m is not None:
            filenames_iter.append(filename)
            idx.append(int(m.group(1)))
    sorted_idx = np.argsort(idx)
    filenames_sorted += [filenames_iter[i] for i in sorted_idx]
    
    # Find the final file (if one exists)
    for filename in filenames:
        if filename.endswith('final.txt'):
            filenames_sorted.append(filename)
            break

    return filenames_sorted

#######################################################################
def write_cells(cells, path, fmt=None, params={}):
    """
    Write the cells in the given population to the given path. 

    Parameters
    ----------
    cells : `numpy.ndarray`
        Existing population of cells.
    path : str
        File path.
    fmt : list of str
        Formatting string for all floats. 
    params : dict
        Parameter values that will be written as comments. The values are 
        formatted by default as strings. 
    """
    if fmt is None:
        fmt = '{:.10g}'
    
    with open(path, 'w') as f:
        # Stitch together the header ... 
        header = ''
        for key in params:
            header += '# {} = {}\n'.format(key, params[key])
        f.write(header)

        # ... then write each cell as a separate row 
        for i in range(cells.shape[0]):
            line = '{:d}\t'.format(int(cells[i, _colidx_id]))         # Cell ID is an integer
            line += '\t'.join([fmt.format(cells[i, j]) for j in range(1, _colidx_group)])
            line += '\t{:d}\t'.format(int(cells[i, _colidx_group]))   # Group is an integer
            if cells.shape[1] > _ncols_required:
                line += '\t'.join([
                    fmt.format(cells[i, j]) for j in range(_colidx_group + 1, cells.shape[1])
                ])
            line += '\n'
            f.write(line)

#######################################################################
def simulate_cell_births_with_switching(cells_init, n_final, rng, R=0.8, L0=1.0,
                                        Ldiv=3.6, lifetime1=4.0, lifetime2=2.0,
                                        growth_mean=1.12, growth_std=0.224,
                                        sample_both_growth_rates=True):
    """
    Simulate the proliferation of a population of cells.

    The population is maintained as a 2-D array whose columns indicate
    the following:

    0) cell ID
    1) cell length 
    2) growth rate 
    3) birth time 
    4) group (1 or 2)
    5) previous switching time for the cell. 

    Parameters
    ----------
    cells_init : `numpy.ndarray`
        Initial population of cells. 
    """
    _colidx_id = 0
    _colidx_l = 1
    _colidx_growth = 2
    _colidx_t0 = 3
    _colidx_group = 4
    _colidx_group_t0 = 5

    # If the initial population has not been given, initialize as one cell
    if cells_init is None:
        cells = np.array([
            [0, L0, rng.normal(loc=growth_mean, scale=growth_std), 0.0, 1, 0.0]
        ])
    else:
        cells = cells_init

    # Initialize the simulation 
    t_curr = 0.0
    growth_rates = list(cells[:, _colidx_growth])
    birth_times = [0.0]
    lifetimes = {1: [], 2: []}
    prefactor = np.log((10 * R + 6 * L0) / (4 * R + 3 * L0))

    # Run until the population reaches the desired size ... 
    while cells.shape[0] < n_final:
        # Calculate the time at which each cell is scheduled to divide 
        tdiv = cells[:, _colidx_t0] + prefactor / cells[:, _colidx_growth]

        # Define an exponentially distributed waiting time for each cell
        #
        # This waiting time does not need to account for the previous 
        # amount of time spent by each cell in its current state, since 
        # the exponential distribution is memoryless 
        mean_lifetimes = [
            lifetime1 if cells[i, _colidx_group] == 1 else lifetime2
            for i in range(cells.shape[0])
        ]
        tswitch = t_curr + rng.exponential(scale=mean_lifetimes)

        # What happens first, division or switching? 
        min_tdiv = np.min(tdiv)
        min_tswitch = np.min(tswitch)

        # If switching were to happen first ... 
        if min_tswitch < min_tdiv:
            # ... then switch the chosen cell and grow every cell by the
            # appropriate length
            min_idx = np.argmin(tswitch)
            lifetime = min_tswitch - cells[min_idx, _colidx_group_t0]
            lifetimes[int(cells[min_idx, _colidx_group])].append(lifetime)
            cells[min_idx, _colidx_group] = (
                2 if cells[min_idx, _colidx_group] == 1 else 1
            )
            cells[min_idx, _colidx_group_t0] = min_tswitch

            # The cell lengths should be updated according to the growth law:
            # 
            # l(t) = -(a / b) + (l(t0) + a / b) * exp(b * (t - t0)),
            #
            # where t0 is the current time, a = (4/3) * (growth rate) * R,
            # and b = growth rate
            dt = min_tswitch - t_curr
            for i in range(cells.shape[0]):
                a = (4. / 3.) * cells[i, _colidx_growth] * R
                b = cells[i, _colidx_growth]
                l0 = cells[i, _colidx_l]
                cells[i, _colidx_l] = -(a / b) + (l0 + (a / b)) * np.exp(b * dt)

            # Finally, update current time 
            t_curr = min_tswitch
        # Otherwise ... 
        else:
            # ... divide the cell with the minimum birth time 
            min_idx = np.argmin(tdiv)

            # Sample new growth rates for the daughter cells 
            if sample_both_growth_rates:
                growth1 = rng.normal(loc=growth_mean, scale=growth_std)
                growth2 = rng.normal(loc=growth_mean, scale=growth_std)
            else:
                growth1 = cells[min_idx, _colidx_growth]
                growth2 = rng.normal(loc=growth_mean, scale=growth_std)

            # Divide the chosen cell 
            max_id = np.max(cells[:, _colidx_id])
            cells[min_idx, _colidx_id] = max_id + 1
            cells[min_idx, _colidx_l] = L0
            cells[min_idx, _colidx_growth] = growth1
            cells[min_idx, _colidx_t0] = min_tdiv
            new_cell = np.array([
                max_id + 2, L0, growth2, min_tdiv,
                cells[min_idx, _colidx_group],
                min_tdiv
            ])
            cells = np.vstack((cells, new_cell.reshape(1, -1)))
            growth_rates.append(growth1)
            growth_rates.append(growth2)
            birth_times.append(min_tdiv)

            # The remaining cell lengths should be updated according to the
            # growth law:
            # 
            # l(t) = -(a / b) + (l(t0) + a / b) * exp(b * (t - t0)),
            #
            # where t0 is the current time, a = (4/3) * (growth rate) * R,
            # and b = growth rate
            dt = min_tdiv - t_curr
            for i in range(cells.shape[0] - 1):
                if i != min_idx:
                    a = (4. / 3.) * cells[i, _colidx_growth] * R
                    b = cells[i, _colidx_growth]
                    l0 = cells[i, _colidx_l]
                    cells[i, _colidx_l] = -(a / b) + (l0 + (a / b)) * np.exp(b * dt)

            # Update current time 
            t_curr = min_tdiv

    return cells, growth_rates, birth_times, lifetimes, t_curr

#######################################################################
def nearest_cell_body_coord_to_point(r, n, half_l, q):
    """
    Return the cell-body coordinate along the given cell centerline that is 
    nearest to the given point. 
    """
    s = np.dot(q - r, n)
    if np.abs(s) <= half_l:
        return s
    elif s > half_l:
        return half_l
    else:    # s < -half_l
        return -half_l

########################################################################
def dist_between_cells(r1, n1, half_l1, r2, n2, half_l2):
    """
    Return the shortest distance between the centerlines of two cells, along
    with the cell-body coordinates at which the shortest distance is achieved.

    The distance vector returned by this function runs from cell 1 to cell 2.
    """
    d = np.zeros((3,), dtype=np.float64)
    s = 0.0
    t = 0.0

    # Are the two cells nearly parallel? 
    #
    # We say that two cells are nearly parallel if they are at an angle of 
    # theta <= 0.01 radians, which translates to cos(theta) >= 0.9999
    cos_theta = np.dot(n1, n2)
    if cos_theta >= 0.9999 or cos_theta <= -0.9999:
        # Identify the four endpoint vectors
        p1 = r1 - half_l1 * n1
        q1 = r1 + half_l1 * n1
        p2 = r2 - half_l2 * n2
        q2 = r2 + half_l2 * n2

        # Get the distance vectors between the endpoints of cell 1 and the
        # body of cell 2
        s_p1_to_cell2 = nearest_cell_body_coord_to_point(r2, n2, half_l2, p1)
        s_q1_to_cell2 = nearest_cell_body_coord_to_point(r2, n2, half_l2, q1)
        d_p1_to_cell2 = r2 + s_p1_to_cell2 * n2 - p1
        d_q1_to_cell2 = r2 + s_q1_to_cell2 * n2 - q1
        dist_p1_to_cell2 = np.linalg.norm(d_p1_to_cell2)
        dist_q1_to_cell2 = np.linalg.norm(d_q1_to_cell2)

        # If the two distance vectors point to the same point along cell 2
        # (which should be an endpoint), return the shorter of the two
        if s_p1_to_cell2 == s_q1_to_cell2:
            if dist_p1_to_cell2 < dist_q1_to_cell2:
                s = -half_l1
                t = s_p1_to_cell2
                d = d_p1_to_cell2
            else:    # dist_p1_to_cell2 >= dist_q1_to_cell2
                s = half_l1
                t = s_q1_to_cell2
                d = d_q1_to_cell2
        # Otherwise, get the distance vectors between the endpoints of cell 2
        # and the body of cell 1
        else:
            s_p2_to_cell1 = nearest_cell_body_coord_to_point(r1, n1, half_l1, p2)
            s_q2_to_cell1 = nearest_cell_body_coord_to_point(r1, n1, half_l1, q2)
            d_p2_to_cell1 = r1 + s_p2_to_cell1 * n1 - p2
            d_q2_to_cell1 = r1 + s_q2_to_cell1 * n1 - q2
            dist_p2_to_cell1 = np.linalg.norm(d_p2_to_cell1)
            dist_q2_to_cell1 = np.linalg.norm(d_q2_to_cell1)

            # Get the two shortest distance vectors among the four 
            sortidx = np.argsort([
                dist_p1_to_cell2, dist_q1_to_cell2, dist_p2_to_cell1, dist_q2_to_cell1
            ])
            minidx = set(sortidx[:2])
            if 0 in minidx:
                if 1 in minidx:      # Average between d_p1_to_cell2 and d_q1_to_cell2
                    s = 0.0
                    t = nearest_cell_body_coord_to_point(r2, n2, half_l2, r1)
                    d = r2 + t * n2 - r1
                elif 2 in minidx:    # Average between d_p1_to_cell2 and d_p2_to_cell1
                    s = (-half_l1 + s_p2_to_cell1) / 2
                    t = nearest_cell_body_coord_to_point(r2, n2, half_l2, r1 + s * n1)
                    d = r2 + t * n2 - r1 - s * n1
                else:   # 3 in minidx; average between d_p1_to_cell2 and d_q2_to_cell1
                    s = (-half_l1 + s_q2_to_cell1) / 2
                    t = nearest_cell_body_coord_to_point(r2, n2, half_l2, r1 + s * n1)
                    d = r2 + t * n2 - r1 - s * n1
            elif 1 in minidx:
                if 2 in minidx:      # Average between d_q1_to_cell2 and d_p2_to_cell1
                    s = (half_l1 + s_p2_to_cell1) / 2
                    t = nearest_cell_body_coord_to_point(r2, n2, half_l2, r1 + s * n1)
                    d = r2 + t * n2 - r1 - s * n1
                else:   # 3 in minidx; average between d_q1_to_cell2 and d_q2_to_cell1
                    s = (half_l1 + s_q2_to_cell1) / 2
                    t = nearest_cell_body_coord_to_point(r2, n2, half_l2, r1 + s * n1)
                    d = r2 + t * n2 - r1 - s * n1
            else:   # 2 and 3 in minidx; average between d_p2_to_cell1 and d_q2_to_cell1
                t = 0.0
                s = nearest_cell_body_coord_to_point(r1, n1, half_l1, r2)
                d = r2 - r1 - s * n1
    # Otherwise, compute the distance vector 
    else:
        # We are looking for the values of s, t such that the norm of 
        # r2 + t * n2 - r1 - s * n1 is minimized
        r12 = r2 - r1
        r12_dot_n1 = np.dot(r12, n1)
        r12_dot_n2 = np.dot(r12, n2)
        n1_dot_n2 = np.dot(n1, n2)
        s_numer = r12_dot_n1 - n1_dot_n2 * r12_dot_n2
        t_numer = n1_dot_n2 * r12_dot_n1 - r12_dot_n2
        denom = 1.0 - n1_dot_n2 * n1_dot_n2
        s = s_numer / denom
        t = t_numer / denom

        # If the unconstrained minimum values do not fall within the square 
        # [-half_l1, half_l1] x [-half_l2, half_l2] ... 
        if np.abs(s) > half_l1 or np.abs(t) > half_l2:
            # Find the side of the square to which the unconstrained minimum
            # value is nearest
            #
            # Region 1 (above top side):      between t = -s - X and t = s - X
            # Region 2 (right of right side): between t = s - X and t = -s + X
            # Region 3 (below bottom side):   between t = -s + X and t = s + X
            # Region 4 (left of left side):   between t = s + X and t = -s - X
            #
            # where X = (l1 - l2) / 2
            X = half_l1 - half_l2
            Y = s + X
            Z = s - X
            if t >= -Y and t >= Z:    # In region 1
                # Set t = half_l2 and find s
                q = r2 + half_l2 * n2
                s = nearest_cell_body_coord_to_point(r1, n1, half_l1, q)
                t = half_l2
            elif t < Z and t >= -Z:   # In region 2
                # Set s = half_l1 and find t
                q = r1 + half_l1 * n1
                t = nearest_cell_body_coord_to_point(r2, n2, half_l2, q)
                s = half_l1
            elif t < -Z and t < Y:    # In region 3
                # Set t = -half_l2 and find s
                q = r2 - half_l2 * n2
                s = nearest_cell_body_coord_to_point(r1, n1, half_l1, q)
                t = -half_l2
            else:   # t >= s + X and t < -s - X, in region 4
                # Set s = -half_l1 and find t
                q = r1 - half_l1 * n1
                t = nearest_cell_body_coord_to_point(r2, n2, half_l2, q)
                s = -half_l1
        d = r12 + t * n2 - s * n1

    return d, s, t

