"""
Plot simulated biofilms from all (or a subset of) files within a given
directory.

Authors:
    Kee-Myoung Nam

Last updated:
    12/8/2023
"""

import matplotlib
matplotlib.use('agg')
import sys
import os
import glob
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils import read_cells
from plot import plot_simulation

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
    filenames = parse_dir(sys.argv[1])
    outpath = sys.argv[2]

    # Minimum number of cells to be plotted
    min_cells = 0

    # Maximum number of cells to be plotted
    max_cells = 5000

    # Maximum number of frames to be plotted
    max_frames = 600    # 600 frames at 10 fps = 60 sec

    # Cell radius and line width
    R = 0.35
    linewidth = 0.8

    # Colors for all groups being tracked in the simulation
    colors = sns.color_palette()[:3]

    # Parse all files up to the given maximum number of cells
    filenames_to_plot = []
    for filename in filenames:
        cells, _ = read_cells(filename)
        size = cells.shape[0]
        if size >= min_cells and size < max_cells:
            filenames_to_plot.append(filename)
        elif size == max_cells:
            filenames_to_plot.append(filename)
            break
    print(
        '... parsed {} frames in {}'.format(
            len(filenames_to_plot), sys.argv[1]
        )
    )

    # If the number of frames exceeds the maximum number, skip over
    # frames in constant increments (excluding the initial and final 
    # frames, which should be kept)
    if len(filenames_to_plot) > max_frames:
        increment = len(filenames_to_plot) // max_frames + 1
        filename_final = filenames_to_plot[-1]
        filenames_to_plot = filenames_to_plot[:-1:increment]
        filenames_to_plot.append(filename_final)
        
    # Plot simulation to the given output file
    print(
        '... plotting {} frames in {}'.format(
            len(filenames_to_plot), sys.argv[1]
        )
    )
    plot_simulation(
        filenames_to_plot, outpath, R=R, fps=10, colors=colors,
        figsize=(12.8, 9.6), linewidth=linewidth
    )

