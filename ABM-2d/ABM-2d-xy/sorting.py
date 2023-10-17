"""
Given a directory containing the simulation of a biofilm, plot the 
trajectory of radial sortedness over time. 

Authors:
    Kee-Myoung Nam

Last updated:
    10/17/2023
"""

import os
import sys
import glob
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils import read_cells
from metrics import radial_sortedness

#####################################################################
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
    filenames = glob.glob(path)
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
    
    # Find the final file
    for filename in filenames:
        if 'final' in filename:
            filenames_sorted.append(filename)
            break
    
    return filenames_sorted

#####################################################################
if __name__ == '__main__':
    rng = np.random.default_rng(1234567890)
    filenames = parse_dir(os.path.join(sys.argv[1], '*'))

    # Minimum number of cells for sortedness to be measured
    min_cells = 100

    # For each file ...
    times = []
    sortedness = []
    for filename in filenames:
        # Parse the cells and timepoint associated with the file
        cells, params = read_cells(filename)
        try:
            t = params['t_curr']
        except KeyError:    # Timepoint not stored in initial file
            t = 0.0

        # Compute the radial sortedness of the population, given that 
        # there are more than the minimum number of cells
        if cells.shape[0] > min_cells:
            times.append(t)
            scores = np.array(
                [0 if cells[i, 9] == 2 else 1 for i in range(cells.shape[0])],
                dtype=np.int32
            )
            sortedness.append(radial_sortedness(cells, scores, rng))

    # Plot the sortedness profile over time
    color = sns.color_palette()[0]
    plt.plot(times, sortedness, c=color)

    # Plot dashed horizontal line at final sortedness value
    ax = plt.gca()
    xlim = ax.get_xlim()
    color = sns.color_palette()[1]
    plt.plot(xlim, [sortedness[-1], sortedness[-1]], linestyle='--', c=color)
    plt.annotate(
        '{:.4f}'.format(sortedness[-1]),
        (xlim[1] - (xlim[1] - xlim[0]) * 0.05, sortedness[-1] - 0.1),
        horizontalalignment='right'
    )

    # Configure axes
    ax = plt.gca()
    ax.set_xlabel('Time')
    ax.set_ylabel('Sortedness')
    ax.set_xlim(xlim)
    ax.set_ylim([-1, 1])
    plt.savefig(sys.argv[2])

