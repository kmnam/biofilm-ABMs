"""
Miscellaneous functions.

Authors:
    Kee-Myoung Nam

Last updated:
    1/11/2024
"""
import os
import glob
import re
import numpy as np

#######################################################################
# In what follows, a population of N cells is represented as a 2-D array of 
# size (N, 12+), where each row represents a cell and stores the following data:
# 
# 0) x-coordinate of cell center
# 1) y-coordinate of cell center
# 2) z-coordinate of cell center
# 3) x-coordinate of cell orientation vector
# 4) y-coordinate of cell orientation vector
# 5) z-coordinate of cell orientation vector
# 6) cell length (excluding caps)
# 7) half of cell length (excluding caps)
# 8) timepoint at which the cell was formed
# 9) cell growth rate
# 10) cell's ambient viscosity with respect to surrounding fluid
# 11) cell-surface friction coefficient
# 12, 13, 14, ...) additional features
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
                params[m.group(1)] = float(m.group(2))

    # Then read the cells that are stored in the file 
    cells = np.loadtxt(path, comments='#', delimiter='\t', skiprows=0)
    if len(cells.shape) == 1:
        cells = cells.reshape((1, -1))

    return cells, params

#######################################################################
def parse_dir(path):
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
    filenames = glob.glob(os.path.join(path, '*'))
    filenames_sorted = []

    # Find the initial file 
    for filename in filenames:
        if 'init' in filename:
            filenames_sorted.append(filename)
            break

    # Run through all intermediate files and sort them in order of iteration
    filenames_iter = [filename for filename in filenames if 'iter' in filename]
    idx = []
    for filename in filenames_iter:
        m = re.search(r'iter([0-9]+)\.txt', filename)
        idx.append(int(m.group(1)))
    sorted_idx = np.argsort(idx)
    filenames_sorted += [filenames_iter[i] for i in sorted_idx]
    
    # Find the final file (if one exists)
    for filename in filenames:
        if 'final' in filename:
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
        Formatting strings for any additional features (columns 12, 13, ...) 
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
    elif len(fmt) != cells.shape[1] - 12:
        raise ValueError('Incorrect number of formatting strings specified')
    else:
        with open(path, 'w') as f:
            header = ''
            for key in params:
                header += '# {} = {}\n'.format(key, params[key])
            f.write(header)
            for i in range(cells.shape[0]):
                line = '\t'.join(['{:.10g}'.format(cells[i, j]) for j in range(12)])
                for j, k in enumerate(range(12, cells.shape[1])):
                    line += '\t' + fmt[j].format(cells[i, k])
                f.write(line + '\n')

