"""
Authors:
    Kee-Myoung Nam

Last updated:
    4/29/2025
"""

import gc
import sys
from time import sleep
import pyvista as pv
from utils import read_cells
from trajectories import parse_boundary, parse_trajectory, save_trajectory_frame_to_file

#######################################################################
if __name__ == '__main__':
    filename = sys.argv[1]
    outfilename = sys.argv[2]
    boundary_prefix = sys.argv[3]
    idx = int(sys.argv[4])
    position = None
    if len(sys.argv) == 6:
        values = [float(x) for x in sys.argv[5].strip('()').split(',')]
        position = [values[:3], values[3:6], values[6:]]

    # Parse the trajectory and extract the frame 
    trajectory, params = parse_trajectory(filename)
    R = params['R']

    # Parse the corresponding boundary 
    iteration = int(trajectory[idx, 0])
    boundary_filename = boundary_prefix + '_iter{}_boundary.txt'.format(iteration)
    boundary = parse_boundary(boundary_filename)

    # Plot the corresponding trajectory frame 
    save_trajectory_frame_to_file(
        trajectory, idx, boundary, R, outfilename, plot_prev=True, res=50,
        image_scale=5, position=position
    )
    gc.collect()
    sleep(1)

