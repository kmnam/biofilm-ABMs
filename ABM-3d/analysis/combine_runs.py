"""
Combine multiple iterations of a simulation for post-processing.

Authors:
    Kee-Myoung Nam

Last updated:
    10/17/2025
"""

import sys
import os
import glob
import shutil
import re
import json
from utils import parse_dir, read_cells, write_cells

######################################################################
if __name__ == '__main__':
    # Collect input directory  
    indir = sys.argv[1]

    # Check if there indeed were additional runs performed
    if not os.path.isdir(os.path.join(indir, 'run1')):
        sys.exit()

    # Copy over all files from the initial simulation into a new 'combined' 
    # directory
    try:
        os.mkdir(os.path.join(indir, 'combined'))
        for filename in glob.glob(os.path.join(indir, '*.txt')):
            shutil.copy(filename, os.path.join(indir, 'combined'))
    except FileExistsError:
        pass

    # For each subsequent run ...
    maxrun = None
    i = 1
    while maxrun is None:
        if not os.path.isdir(os.path.join(indir, 'run{}'.format(i))):
            maxrun = i - 1
        i += 1

    # For each run after the initial run ... 
    rundirs = [os.path.join(indir, 'run{}'.format(i)) for i in range(1, maxrun + 1)]
    t_init = 0
    idx_init = 0
    for rundir in rundirs:
        # Find the .json file
        json_filename = next(
            filename for filename in glob.glob(os.path.join(rundir, '*.json'))
        )
        
        # Find the initial file that was used from the previous run
        with open(json_filename) as f:
            json_data = json.load(f)
        init_filename = json_data['init_filename']
        
        # Get the iteration number and timepoint for the corresponding frame
        m = re.search(r'_iter([0-9]+)', os.path.basename(init_filename))
        if m is None:
            raise RuntimeError('Initial filename does not contain iteration number')
        idx_init += int(m[1])
        _, params = read_cells(init_filename)
        t_init += params['t_curr']

        # Read each file in the current directory (other than the initial and
        # final files) ... 
        filenames = parse_dir(os.path.join(rundir, '*'))
        for filename in filenames[1:]:
            # Get the iteration number and timepoint for the corresponding
            # frame
            m = re.search(r'_iter([0-9]+)', os.path.basename(filename))
            if m is None:
                idx_curr = -1   # In this case, this file should be a final file
            else:
                idx_curr = int(m[1])
            cells, params = read_cells(filename)
            t_curr = params['t_curr']

            # Copy over the file with an updated iteration number and timepoint
            params['t_curr'] = t_init + t_curr
            new_filename = filename.replace(
                '_iter{}'.format(idx_curr), '_iter{}'.format(idx_init + idx_curr)
            )
            new_filename = os.path.join(
                indir, 'combined', os.path.basename(new_filename)
            )
            write_cells(cells, new_filename, params=params)

    # Copy over the final lineage file
    lineage_filename = next(
        filename for filename in glob.glob(os.path.join(rundirs[-1], '*.txt'))
        if filename.endswith('_lineage.txt')
    )
    shutil.copy(
        lineage_filename,
        os.path.join(indir, 'combined', os.path.basename(lineage_filename))
    )

