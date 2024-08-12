"""
Functions for visualizing spherocylindrical cells in 3-D with Pyvista. 

Authors:
    Kee-Myoung Nam

Last updated:
    8/12/2024
"""

import sys
import re
import numpy as np
from utils import read_cells, parse_dir
from plot3d import plot_simulation

__colidx_id = 0
__colidx_rx = 1
__colidx_ry = 2
__colseq_r = [1, 2]
__colidx_nx = 3
__colidx_ny = 4
__colseq_n = [3, 4]
__colidx_drx = 5
__colidx_dry = 6
__colidx_dnx = 7
__colidx_dny = 8
__colidx_l = 9
__colidx_half_l = 10
__colidx_t0 = 11
__colidx_growth = 12
__colidx_eta0 = 13
__colidx_eta1 = 14
__colidx_group = 15
__colidx_bound = -1

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
    overwrite_frames = ('--overwrite' in args)
    multistage = ('--multistage' in args)
    filenames = parse_dir(inprefix, multistage=multistage)
    print('Parsing {} files ...'.format(len(filenames)))

    # Parse the initial files in each stage
    init_timepoints = {1: 0.0}
    n_stages = 1
    if multistage:
        n_stages = max(
            int(re.search(r'_stage(\d+)_(?:init|iter\d+|final)\.txt$', filename).group(1))
            for filename in filenames
        )
        for stage in range(2, n_stages + 1):
            filename_prev_final = next(
                filename for filename in filenames
                if re.search(r'_stage{}_final\.txt$'.format(stage - 1), filename) is not None
            )
            _, params = read_cells(filename_prev_final)
            init_timepoints[stage] = params['t_curr']

    # Get cell radius, final dimensions, and final timepoint from final file
    cells, params = read_cells(filenames[-1])
    R = params['R']
    L0 = params['L0']
    E0 = params['E0']
    sigma0 = params['sigma0']
    rz = R - (1 / R) * ((R * R * sigma0) / (4 * E0)) ** (2 / 3)
    xmin = np.floor(cells[:, __colidx_rx].min() - 4 * L0)
    xmax = np.ceil(cells[:, __colidx_rx].max() + 4 * L0)
    ymin = np.floor(cells[:, __colidx_ry].min() - 4 * L0)
    ymax = np.ceil(cells[:, __colidx_ry].max() + 4 * L0)
    zmin = rz - R
    zmax = rz + R
    t_final = (
        params['t_curr'] if not multistage else
        init_timepoints[n_stages] + params['t_curr']
    )

    # Generate array of timepoints
    timepoints = np.linspace(0, t_final, nframes_total)
  
    # Run through the files and identify their corresponding timepoints 
    file_timepoints = []    # Timepoints for all files 
    for filename in filenames:
        _, params = read_cells(filename)
        if not multistage:
            file_timepoints.append(params['t_curr'])
        # If processing a multi-stage simulation, we must update the timepoint
        # in each file
        else:
            stage = int(
                re.search(r'_stage(\d+)_(?:init|iter\d+|final)\.txt$', filename).group(1)
            )
            file_timepoints.append(params['t_curr'] + init_timepoints[stage])

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
            xmin, xmax, ymin, ymax, zmin, zmax, res=20, fps=20,
            times=plot_timepoints[start:end], uniform_color=uniform_color,
            plot_boundary=plot_boundary, plot_membrane=plot_membrane,
            plot_arrested=plot_arrested, overwrite_frames=overwrite_frames
        )
        print('Saving video: {}_{}.avi'.format(outprefix, i))
        start += nframes_per_video
        end += nframes_per_video
        i += 1

