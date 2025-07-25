"""
Authors:
    Kee-Myoung Nam

Last updated:
    4/30/2025
"""

import gc
import sys
from time import sleep
import pyvista as pv
from utils import read_cells
from plot3d import save_simplicial_complex_to_file

#######################################################################
if __name__ == '__main__':
    # Parse input arguments 
    filename = sys.argv[1]
    cells_filename = sys.argv[2]
    outfilename = sys.argv[3]
    maxdim = int(sys.argv[4])
    xmin = None
    xmax = None
    ymin = None
    ymax = None
    zmin = None
    zmax = None
    position = None
    view = 'xyz'
    if len(sys.argv) == 6 and sys.argv[5] == '--basal':
        view = 'xy_bottom'
    elif len(sys.argv) == 11:
        xmin, xmax, ymin, ymax, zmin, zmax = [float(x) for x in sys.argv[5:11]]
    elif len(sys.argv) == 12:
        xmin, xmax, ymin, ymax, zmin, zmax = [float(x) for x in sys.argv[5:11]]
        if sys.argv[11] == '--basal':
            view = 'xy_bottom'
        else:
            values = [float(x) for x in sys.argv[11].strip('()').split(',')]
            position = [values[:3], values[3:6], values[6:]]
    elif len(sys.argv) == 13:
        xmin, xmax, ymin, ymax, zmin, zmax = [float(x) for x in sys.argv[5:11]]
        values = [float(x) for x in sys.argv[11].strip('()').split(',')]
        position = [values[:3], values[3:6], values[6:]]
        if sys.argv[12] == '--basal':
            view = 'xy_bottom'

    _, params = read_cells(cells_filename)
    time = params['t_curr']
    if xmin is not None:
        xmin -= params['L0']
        ymin -= params['L0']
        zmin -= params['L0']
        xmax += params['L0']
        ymax += params['L0']
        zmax += params['L0']
    save_simplicial_complex_to_file(
        filename, cells_filename, outfilename, xmin=xmin, xmax=xmax, ymin=ymin,
        ymax=ymax, zmin=zmin, zmax=zmax, maxdim=maxdim, view=view, time=time,
        image_scale=5, position=position, show=False
    )
    gc.collect()
    sleep(1)

