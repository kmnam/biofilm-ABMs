"""
Authors:
    Kee-Myoung Nam

Last updated:
    12/11/2023
"""

import sys
import os
import glob
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils import read_cells

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
    # Parse input directory 
    filenames = parse_dir(sys.argv[1])
    absmeans1 = []
    absmeans2 = []
    radmeans1 = []
    radmeans2 = []
    ncells = []

    # Minimum number of cells to plot  
    min_cells = int(sys.argv[3])

    # For each file in the directory ... 
    for filename in filenames:
        # Parse the cells in the file 
        cells, params = read_cells(filename)
        n = cells.shape[0]

        # If the number of cells exceeds the given minimum, get the 
        # mean absolute and radial velocity of the two groups of cells
        if n >= min_cells:
            # Get the center of mass of the cells 
            center_x = np.mean(cells[:, 0])
            center_y = np.mean(cells[:, 1])
            
            # Get the radial orientation of each cell relative to the center
            dists_x = cells[:, 0] - center_x
            dists_y = cells[:, 1] - center_y
            orientations = np.hstack((dists_x.reshape(-1, 1), dists_y.reshape(-1, 1)))
            orientations /= np.sqrt(dists_x ** 2 + dists_y ** 2).reshape((-1, 1))

            # Get the absolute velocity of each cell 
            idx1 = np.where(cells[:, 10] == 1)[0]
            idx2 = np.where(cells[:, 10] == 2)[0]
            absvel1 = np.linalg.norm(cells[idx1, 11:13], axis=1)
            absvel2 = np.linalg.norm(cells[idx2, 11:13], axis=1)

            # Get the radial velocity of each cell 
            radvel = np.zeros(n, dtype=np.float64)
            for i in range(n):
                radvel[i] = np.dot(cells[i, :2], orientations[i, :])
            radvel1 = radvel[idx1]
            radvel2 = radvel[idx2]

            # Get the mean absolute velocity and radial velocity over cells
            # of each group 
            ncells.append(n)
            absmeans1.append(np.mean(absvel1))
            absmeans2.append(np.mean(absvel2))
            radmeans1.append(np.mean(radvel1))
            radmeans2.append(np.mean(radvel2))

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(7, 5))
    axes[0].plot(ncells, absmeans1)
    axes[0].plot(ncells, absmeans2)
    axes[1].plot(ncells, radmeans1)
    axes[1].plot(ncells, radmeans2)
    axes[1].set_xlabel('Number of cells')
    axes[0].set_ylabel('Absolute velocity ($\mu$m/h)')
    axes[1].set_ylabel('Radial velocity ($\mu$m/h)')
    plt.tight_layout()
    plt.savefig(sys.argv[2])

