"""
Functions for visualizing spherocylindrical cells in 3-D with Pyvista. 

Authors:
    Kee-Myoung Nam

Last updated:
    1/15/2025
"""

import sys
import re
import numpy as np
from utils import read_cells, parse_dir
from plot3d import plot_simulation

#######################################################################
if __name__ == '__main__':
    inprefix = sys.argv[1]
    outprefix = sys.argv[2]
    nframes_total = int(sys.argv[3])
    nframes_per_video = int(sys.argv[4])
    args = sys.argv[5:]
    uniform_color = ('--uniform' in args)
    plot_boundary = ('--bound' in args)
    plot_membrane = ('--membrane' in args)
    plot_arrested = ('--arrested' in args)
    plot_3d = ('--3d' in args)
    overwrite_frames = ('--overwrite' in args)
    filenames = parse_dir(inprefix)
    print('Parsing {} files ...'.format(len(filenames)))

    # Get cell radius, final dimensions, and final timepoint from final file
    cells, params = read_cells(filenames[-1])
    R = params['R']
    L0 = params['L0']
    E0 = params['E0']
    sigma0 = params['sigma0']
    rz = R - (1 / R) * ((R * R * sigma0) / (4 * E0)) ** (2 / 3)
    _colidx_rx = 1
    _colidx_ry = 2
    xmin = np.floor(cells[:, _colidx_rx].min() - 4 * L0)
    xmax = np.ceil(cells[:, _colidx_rx].max() + 4 * L0)
    ymin = np.floor(cells[:, _colidx_ry].min() - 4 * L0)
    ymax = np.ceil(cells[:, _colidx_ry].max() + 4 * L0)
    zmin = rz - R
    zmax = rz + R
    t_final = params['t_curr']

    # Generate array of timepoints
    timepoints = np.linspace(0, t_final, nframes_total)
  
    # Run through the files and identify their corresponding timepoints 
    file_timepoints = []    # Timepoints for all files 
    for filename in filenames:
        _, params = read_cells(filename)
        file_timepoints.append(params['t_curr'])

    # Run through the files and identify, for each timepoint, the file 
    # whose timepoint is closest
    filenames_nearest = []
    plot_timepoints = []      # Timepoints for all files to be plotted
    for t in timepoints:
        nearest_idx = np.argmin(np.abs(file_timepoints - t))
        filenames_nearest.append(filenames[nearest_idx])
        plot_timepoints.append(file_timepoints[nearest_idx])

    # Plot the simulation in fixed increments
    start = 0
    end = nframes_per_video
    i = 0
    while end <= nframes_total:
        plot_simulation(
            filenames_nearest[start:end], outprefix + '_{}.avi'.format(i),
            xmin, xmax, ymin, ymax, zmin, zmax, plot_3d=plot_3d, res=20, fps=20,
            times=plot_timepoints[start:end], uniform_color=uniform_color,
            plot_boundary=plot_boundary, plot_membrane=plot_membrane,
            plot_arrested=plot_arrested, overwrite_frames=overwrite_frames
        )
        print('Saving video: {}_{}.avi'.format(outprefix, i))
        start += nframes_per_video
        end += nframes_per_video
        i += 1

