"""
Given a directory containing the simulation of a biofilm, plot the 
trajectory of radial Spearman correlation coefficient over time.

Authors:
    Kee-Myoung Nam

Last updated:
    10/22/2023
"""

import os
import sys
import glob
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils import read_cells
from metrics import (
    radial_sortedness, radial_spearman_coeff, radial_kendall_tau
)

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

#####################################################################
if __name__ == '__main__':
    rng = np.random.default_rng(1234567890)
    filenames = parse_dir(sys.argv[1])

    # Minimum number of cells for sortedness to be measured
    min_cells = 100

    # Maximum number of cells for sortedness to be measured
    max_cells = 5000

    # Array of population sizes to demarcate on the plot
    sizes_to_plot = [500, 1000, 2000]

    # For each file ...
    times = []
    sizes = []
    sortedness = []
    spearman = []
    kendall = []
    for filename in filenames:
        # Parse the cells, population size, and timepoint associated with
        # the file
        cells, params = read_cells(filename)
        try:
            t = params['t_curr']
        except KeyError:    # Timepoint not stored in initial file
            t = 0.0

        # Compute the radial Spearman correlation coefficient of the
        # population, given that there are more than the minimum number
        # of cells
        size = cells.shape[0]
        if size > min_cells and size <= max_cells:
            times.append(t)
            sizes.append(size)
            scores = np.array(
                [0 if cells[i, 9] == 2 else 1 for i in range(size)],
                dtype=np.int32
            )
            sortedness.append(radial_sortedness(cells, scores, rng))
            spearman.append(radial_spearman_coeff(cells, scores))
            kendall.append(radial_kendall_tau(cells, scores))

    # Plot the sortedness profile over time
    c1, c2, c3 = sns.color_palette()[:3]
    plt.plot(times, sortedness, c=c1, zorder=0)
    plt.plot(times, spearman, c=c2, zorder=0)
    plt.plot(times, kendall, c=c3, zorder=0)

    # Plot dashed vertical lines for sortedness values at chosen
    # population sizes
    color = sns.color_palette()[4]
    for size in sizes_to_plot:
        try:
            idx = sizes.index(size)
        except ValueError:
            for i, s in enumerate(sizes):
                if s > size:
                    idx = i
                    break
        plt.plot(
            [times[idx], times[idx]], [-1, 1],
            linestyle='--', c=color, zorder=1
        )
        plt.annotate(
            r'$n = {}$'.format(size),
            (times[idx] - (times[-1] - times[0]) * 0.01, 0.95),
            verticalalignment='top',
            horizontalalignment='right'
        )
    plt.plot(
        [times[-1], times[-1]], [-1, 1],
        linestyle='--', c=color, zorder=1
    )
    plt.annotate(
        r'$n = {}$'.format(sizes[-1]),
        (times[-1] - (times[-1] - times[0]) * 0.01, 0.95),
        verticalalignment='top',
        horizontalalignment='right'
    )

    # Annotate final sortedness values
    color = sns.color_palette()[3]
    plt.scatter(
        [times[-1], times[-1], times[-1]],
        [sortedness[-1], spearman[-1], kendall[-1]],
        marker='X', s=30, color=color, zorder=1
    )
    plt.annotate(
        '{:.4f}'.format(spearman[-1]),
        (times[-1], spearman[-1] - 0.05),
        verticalalignment='top',
        horizontalalignment='right'
    )

    # Configure axes
    ax = plt.gca()
    ax.set_xlabel('Time')
    ax.set_ylabel('Sortedness')
    ax.set_ylim([-1, 1])
    plt.savefig(sys.argv[2])

