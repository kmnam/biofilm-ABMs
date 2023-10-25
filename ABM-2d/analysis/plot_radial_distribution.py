"""
Given a directory containing the simulation of a biofilm containing 
two groups of cells, plot the distribution of the two groups as 
a function of radial distance. 

Authors:
    Kee-Myoung Nam

Last updated:
    10/25/2023
"""

import os
import sys
import glob
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils import read_cells
from metrics import radial_group_distribution

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

    # Array of population sizes for which to measure the distributions 
    sizes_to_plot = [500, 1000, 2000, 5000]

    # First parse each file ... 
    times = []
    sizes = []
    for filename in filenames:
        # Parse the population size, and timepoint associated with
        # the file
        cells, params = read_cells(filename)
        try:
            t = params['t_curr']
        except KeyError:    # Timepoint not stored in initial file
            t = 0.0
        times.append(t)
        sizes.append(cells.shape[0])

    # Identify the frames that best match the population sizes of interest
    R = 0.8
    L0 = 1.0
    delta = 2 * (R + L0)
    sizes = np.array(sizes, dtype=np.int64)
    radii = []
    distributions = []
    sizes_plotted = []
    times_plotted = []
    for size in sizes_to_plot:
        i = np.argmin(np.abs(sizes - size))
        cells, params = read_cells(filenames[i])
        t = times[i]
        s = cells.shape[0]
        r, dist = radial_group_distribution(cells, delta)
        radii.append(r)
        distributions.append(dist)
        sizes_plotted.append(s)
        times_plotted.append(t)

    # Plot the distributions
    colors = sns.color_palette()[:len(sizes_to_plot)]
    for i in range(len(sizes_plotted)):
        plt.plot(radii[i], distributions[i], c=colors[i], zorder=0)

    # Configure axes and legend
    ax = plt.gca()
    ax.set_xlabel('Radial distance')
    ax.set_ylabel('Fraction of blue cells')
    ax.set_ylim([0, 1])
    plt.legend([r'$n = {}$'.format(s) for s in sizes_plotted])
    plt.savefig(sys.argv[2])
