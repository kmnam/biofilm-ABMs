"""
Functions for visualizing spherocylindrical cells in 3-D with Pyvista. 

Authors:
    Kee-Myoung Nam

Last updated:
    7/11/2024
"""

import sys
import numpy as np
import pyvista as pv
pv.start_xvfb()
import seaborn as sns
from utils import read_cells, parse_dir
from plot3d import plot_simulation

#######################################################################
if __name__ == '__main__':
    inprefix = sys.argv[1]
    outprefix = sys.argv[2]
    nframes_total = int(sys.argv[3])
    nframes_per_video = int(sys.argv[4])
    extra_args = sys.argv[5:]
    uniform_color = ('--uniform-color' in extra_args)
    overwrite_frames = ('--overwrite-frames' in extra_args)
    filenames = parse_dir(inprefix)
    print('Parsing {} files ...'.format(len(filenames)))

    # Get cell radius, final dimensions, and final timepoint from final file
    cells, params = read_cells(filenames[-1])
    R = params['R']
    L0 = params['L0']
    E0 = params['E0']
    sigma0 = params['sigma0']
    rz = R - (1 / R) * ((R * R * sigma0) / (4 * E0)) ** (2 / 3)
    xmin = np.floor(cells[:, 0].min() - 4 * L0)
    xmax = np.ceil(cells[:, 0].max() + 4 * L0)
    ymin = np.floor(cells[:, 1].min() - 4 * L0)
    ymax = np.ceil(cells[:, 1].max() + 4 * L0)
    zmin = rz - R
    zmax = rz + R
    t_final = params['t_curr']

    # Generate array of timepoints
    timepoints = np.linspace(0, t_final, nframes_total)
    
    # Run through the files and identify, for each timepoint, the file 
    # whose timepoint is closest
    file_timepoints = []
    for filename in filenames:
        _, params = read_cells(filename)
        file_timepoints.append(params['t_curr'])
    filenames_nearest = []
    for t in timepoints:
        nearest_idx = np.argmin(np.abs(file_timepoints - t))
        filenames_nearest.append(filenames[nearest_idx])

    # Plot the simulation in fixed increments
    start = 0
    end = nframes_per_video
    i = 0
    while end <= nframes_total:
        plot_simulation(
            filenames_nearest[start:end], outprefix + '_{}.avi'.format(i), R,
            rz, xmin, xmax, ymin, ymax, zmin, zmax, view='xy', res=20, fps=20,
            uniform_color=uniform_color, overwrite_frames=overwrite_frames
        )
        print('Saving video: {}_{}.avi'.format(outprefix, i))
        start += nframes_per_video
        end += nframes_per_video
        i += 1

