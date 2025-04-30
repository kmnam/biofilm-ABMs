"""
Authors:
    Kee-Myoung Nam

Last updated:
    4/29/2025
"""

import sys
import os
import re
import numpy as np
from utils import read_cells, parse_dir

#######################################################################
if __name__ == '__main__':
    indir = sys.argv[1]
    nframes_total = int(sys.argv[2])
    min_cells = int(sys.argv[3])
    max_cells = int(sys.argv[4])
    filenames = parse_dir(os.path.join(indir, '*.txt'))
 
    # Run through the files and (1) filter our all files that exceed the given
    # number of cells, and (2) identify their corresponding timepoints
    filenames_filtered = []
    file_timepoints = []    # Timepoints for all files 
    for filename in filenames:
        cells, params = read_cells(filename)
        if cells.shape[0] > min_cells and cells.shape[0] <= max_cells:
            filenames_filtered.append(filename)
            file_timepoints.append(params['t_curr'])

    # Generate mesh of timepoints
    _, params = read_cells(filenames_filtered[0])
    t_init = params['t_curr']
    _, params = read_cells(filenames_filtered[-1])
    t_final = params['t_curr']
    t_mesh = np.linspace(t_init, t_final, nframes_total)

    # Run through the files and identify, for each timepoint in the mesh,
    # the file whose timepoint is closest
    filenames_nearest = []
    plot_timepoints = []      # Timepoints for all files to be plotted
    for t in t_mesh:
        nearest_idx = np.argmin(np.abs(file_timepoints - t))
        filenames_nearest.append(filenames_filtered[nearest_idx])
        plot_timepoints.append(file_timepoints[nearest_idx])

    print(' '.join(filenames_nearest))

