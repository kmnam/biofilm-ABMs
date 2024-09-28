"""
Miscellaneous functions.

Authors:
    Kee-Myoung Nam

Last updated:
    9/28/2024
"""
import os
import glob
import re
import numpy as np

#######################################################################
# In what follows, a population of N cells is represented as a 2-D array of 
# size (N, 10+), where each row represents a cell and stores the following data:
# 
# 0) cell ID 
# 1) x-coordinate of cell center
# 2) y-coordinate of cell center
# 3) x-coordinate of cell orientation vector
# 4) y-coordinate of cell orientation vector
# 5) x-coordinate of cell velocity 
# 6) y-coordinate of cell velocity
# 7) x-coordinate of cell orientational velocity 
# 8) y-coordinate of cell orientational velocity 
# 9) cell length (excluding caps)
# 10) half of cell length (excluding caps)
# 11) timepoint at which the cell was formed
# 12) cell growth rate
# 13) cell's ambient viscosity with respect to surrounding fluid
# 14) cell-surface friction coefficient
# 15, 16, ...) additional features
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
        Formatting strings for any additional features (columns 10, 11, ...) 
        in `cells`. This should be either None (in which case any additional 
        features are assumed to be floats) or a list of strings that has as
        many elements as there are additional features.
    params : dict
        Parameter values that will be written as comments. The values are 
        formatted by default as strings. 
    """
    if fmt is None:
        header = ''
        for key in params:
            header += '{} = {}\n'.format(key, params[key])
        header = header.strip()
        np.savetxt(
            path, cells, fmt='%.10g', delimiter='\t', header=header,
            comments='# '
        )
    elif len(fmt) != cells.shape[1] - 10:   # There are 10 canonical columns
        raise ValueError('Incorrect number of formatting strings specified')
    else:
        with open(path, 'w') as f:
            header = ''
            for key in params:
                header += '# {} = {}\n'.format(key, params[key])
            f.write(header)
            for i in range(cells.shape[0]):
                line = '\t'.join(['{:.10g}'.format(cells[i, j]) for j in range(10)])
                for j, k in enumerate(range(10, cells.shape[1])):
                    line += '\t' + fmt[j].format(cells[i, k])
                f.write(line + '\n')

