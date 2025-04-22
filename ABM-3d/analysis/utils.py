"""
Miscellaneous functions.

Authors:
    Kee-Myoung Nam

Last updated:
    4/22/2025
"""
import os
import glob
import re
import numpy as np
import gudhi

#######################################################################
# In what follows, a population of N cells is represented as a 2-D array of 
# size (N, 13+), where each row represents a cell and the columns are 
# specified as defined in `include/indices.hpp`.
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
            elif line.startswith('TETRAHEDRA'):
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
            line = '{:d}\t'.format(cells[i, 0])       # Cell ID is an integer
            line += '\t'.join([fmt.format(cells[i, j]) for j in range(1, 21)])
            line += '\t{:d}\n'.format(cells[i, 21])   # Group is an integer 
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

