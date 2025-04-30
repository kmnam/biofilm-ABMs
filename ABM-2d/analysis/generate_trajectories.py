"""
Authors:
    Kee-Myoung Nam

Last updated:
    4/29/2025
"""

import sys
import os
import glob
import re
import numpy as np
from utils import read_cells, parse_dir
from trajectories import sample_trajectories, write_trajectories

#######################################################################
if __name__ == '__main__':
    indir = sys.argv[1]
    n_cells = int(sys.argv[2])
    nframes_total = int(sys.argv[3])
    outprefix = os.path.join(indir, 'trajectories', indir.strip('/').split('/')[-1])
    seed = int(sys.argv[4])
    if len(sys.argv) >= 6:
        tmin = float(sys.argv[5])
    else:
        tmin = 0.0
    if len(sys.argv) == 7:
        tmax = float(sys.argv[6])
    else:
        tmax = None

    # Sample trajectories ...
    rng = np.random.default_rng(seed)
    trajectories = sample_trajectories(
        indir, rng, n_cells=n_cells, min_rdist=0.9, tmin=tmin, tmax=tmax,
        max_frames=nframes_total
    )

    # ... and write them to file
    cells, params = read_cells(parse_dir(os.path.join(indir, '*'))[-1])
    R = params['R']
    outfilenames = write_trajectories(trajectories, outprefix, R)

    print(' '.join(outfilenames))

